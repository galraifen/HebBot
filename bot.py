#!/usr/bin/env python3
"""
yt-translator — Extract, translate, and play YouTube videos with burned-in Hebrew subtitles.
Usage: python bot.py <youtube_url> [options]

How it works:
  Instead of transcribing audio, this tool reads the Hebrew subtitles that are
  already burned into the video frames using OCR (Tesseract). This gives exact
  timing and accurate source text, then translates to English via Helsinki-NLP.

Pipeline:
  1. Download video (yt-dlp)
  2. Extract frames at regular intervals (ffmpeg)
  3. OCR the subtitle region of each frame (Tesseract, Hebrew)
  4. Deduplicate consecutive identical frames → produces timed segments
  5. Translate Hebrew → English (Helsinki-NLP)
  6. Write .srt
  7. Optionally burn subtitles / play with mpv

OCR cache:
  Extracted frames and OCR results are cached in ocr_cache/<video_stem>/.
  Re-running on the same video skips frame extraction and re-OCR automatically.
  Use --no-ocr-cache to force a fresh run.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import pytesseract
import numpy as np
import yt_dlp
from transformers import pipeline as hf_pipeline


# ─── Config ───────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path("downloads")
OUTPUT_DIR    = Path("output")
OCR_CACHE_DIR = Path("ocr_cache")

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
OCR_CACHE_DIR.mkdir(exist_ok=True)

# How many seconds between sampled frames.
# 0.25 = 4 fps → catches fast subtitle changes without too many frames.
FRAME_INTERVAL = 0.25

# Bottom fraction of the frame to crop for subtitle detection.
# Most burned-in subs live in the bottom 22% of the frame.
# Crop the bottom strip for subtitles. 0.10 = bottom 10%.
# SUBTITLE_CROP_TOP trims the top portion of that strip to skip lower-third graphics.
SUBTITLE_CROP_FRACTION = 0.22
SUBTITLE_CROP_TOP_TRIM = 0.25  # skip the top 40% of the cropped strip

# Two consecutive frames are considered "same subtitle" if their texts are
# this similar (Levenshtein ratio). Handles minor OCR noise between frames.
SIMILARITY_THRESHOLD = 0.60


# ─── Step 1: Download ─────────────────────────────────────────────────────────

def get_video_title(url: str) -> str:
    """Fetch just the video title from YouTube without downloading."""
    try:
        result = subprocess.run(
            ["python3", "-m", "yt_dlp", "--get-filename",
             "-o", "%(title)s.mp4", "--no-playlist", url],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout.strip()
    except Exception:
        return ""


def download_video(url: str) -> Path:
    print("\n📥  Checking download cache...")

    expected_filename = get_video_title(url)
    if expected_filename:
        title_stem = Path(expected_filename).stem.lower()
        matches = [
            f for f in DOWNLOADS_DIR.iterdir()
            if f.is_file() and f.stem.lower() == title_stem
        ]
        if matches:
            print(f"   ⏭️  Already downloaded, skipping: {matches[0]}")
            return matches[0]

    print("   Downloading...")
    ydl_opts = {
        "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        path = Path(filename).with_suffix(".mp4")
        if not path.exists():
            path = Path(filename)

    aac_path = path.with_stem(path.stem + "_aac")
    print("   🔄  Re-encoding audio to AAC for compatibility...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(aac_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path.unlink()
    aac_path.rename(path)

    print(f"   ✅  Saved to: {path}")
    return path


# ─── Steps 2+3: Extract frames and OCR in one streaming pass ─────────────────

def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def crop_subtitle_region(frame: np.ndarray, crop_fraction: float = SUBTITLE_CROP_FRACTION) -> np.ndarray:
    """
    Crop a thin strip at the bottom of the frame where subtitles typically appear.
    Also trims the top portion of that strip to skip lower-third graphics.
    """
    h = frame.shape[0]
    strip_top = int(h * (1 - crop_fraction))
    strip = frame[strip_top:, :]
    # Trim the top of the strip to skip any lower-third overlay graphics
    inner_cut = int(strip.shape[0] * SUBTITLE_CROP_TOP_TRIM)
    return strip[inner_cut:, :]


def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """
    Simple preprocessing for Tesseract:
    - Convert to grayscale
    - Upscale 2× (Tesseract performs better on larger text)
    """
    h, w = region.shape[:2]
    return cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def extract_and_ocr(
    video_path: Path,
    cache_dir: Path,
    interval: float = FRAME_INTERVAL,
    crop_fraction: float = SUBTITLE_CROP_FRACTION,
    use_cache: bool = True,
) -> list[tuple[float, str]]:
    """
    Stream through the video one frame at a time: extract → OCR → discard.
    No frames are kept on disk. OCR results are saved to cache_dir/ocr_results.json.
    Returns list of (timestamp_seconds, text) — empty string if no subtitle detected.
    """
    cache_file = cache_dir / "ocr_results.json"

    if use_cache and cache_file.exists():
        print(f"\n🔍  Loading OCR results from cache ({cache_file})...")
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        print(f"   ✅  Loaded {len(data)} cached OCR results.")
        return [(item["ts"], item["text"]) for item in data]

    duration = get_video_duration(video_path)
    timestamps = [round(t * interval, 3) for t in range(int(duration / interval) + 1)]
    total = len(timestamps)

    tess_config = "--oem 1 --psm 6 -l heb"
    print(f"\n🔍  Extracting + OCR ({interval}s interval, {total} frames, Hebrew)...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []

    for i, ts in enumerate(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            results.append((ts, ""))
            continue

        region = crop_subtitle_region(frame, crop_fraction)
        processed = preprocess_for_ocr(region)
        del frame

        tsv  = pytesseract.image_to_data(processed, config=tess_config, output_type=pytesseract.Output.STRING)
        text = extract_hebrew_from_data(tsv, processed.shape[0])

        results.append((ts, text))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"   {i+1}/{total} frames done...", end="\r")

    cap.release()
    print()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([{"ts": ts, "text": text} for ts, text in results], f, ensure_ascii=False, indent=2)
    print(f"   ✅  OCR complete. Results cached to {cache_file}")
    return results


# ─── Step 4: Build timed segments from OCR results ────────────────────────────

def _normalize(text: str) -> str:
    """Strip whitespace/newlines for comparison — Tesseract adds noisy spacing."""
    return " ".join(text.split())


def extract_hebrew_from_data(tsv: str, image_height: int) -> str:
    """
    Parse Tesseract's TSV output and return Hebrew text whose words fall
    within the subtitle band, discarding bottom-fringe noise.

    From profiling real frames (358px image after 2x upscale):
      - Real subtitle lines: top = 0–200px
      - Noise (background texture, logos): top = 270px+
    So we cut anything below 75% of image height.
    No top cut — real subtitles can appear near the very top of the strip.
    """
    import re

    bottom_cut = int(image_height * 0.75)
    hebrew_re  = re.compile(r'[\u05d0-\u05ea\u05f0-\u05f4\ufb1d-\ufb4e]')

    # Group words by (block_num, line_num), filtering by position and confidence
    lines_map: dict[tuple, list[str]] = {}
    for row in tsv.splitlines()[1:]:  # skip header
        parts = row.split('\t')
        if len(parts) < 12:
            continue
        try:
            top  = int(parts[7])
            conf = int(float(parts[10]))
            text = parts[11].strip()
        except (ValueError, IndexError):
            continue
        if not text or conf < 30:
            continue
        if top > bottom_cut:
            continue
        key = (parts[2], parts[4])  # block_num, line_num
        lines_map.setdefault(key, []).append(text)

    result = []
    for words in lines_map.values():
        line         = " ".join(words)
        hebrew_count = len(hebrew_re.findall(line))
        printable    = len(re.findall(r'\S', line))
        if printable > 0 and hebrew_count / printable >= 0.50 and hebrew_count >= 2:
            result.append(line)

    return "\n".join(result)


def _text_similarity(a: str, b: str) -> float:
    """
    Simple character-level similarity ratio (like difflib.SequenceMatcher).
    Avoids importing difflib for just one function.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Count matching chars via longest common subsequence approximation
    longer = max(len(a), len(b))
    matches = sum(c1 == c2 for c1, c2 in zip(a, b))
    return matches / longer


def build_segments(ocr_results: list[tuple[float, str]]) -> list[dict]:
    """
    Collapse consecutive OCR frames with similar text into subtitle segments.
    Each segment: {"start": float, "end": float, "text": str}

    Logic:
    - Empty frames (no text) are treated as gaps.
    - Frames with text similar to the current segment extend it.
    - A sufficiently different text starts a new segment.
    """
    segments = []
    current_text = ""
    current_start = None

    for ts, text in ocr_results:
        text = _normalize(text)
        if not text:
            # No subtitle in this frame — close any open segment
            if current_text and current_start is not None:
                segments.append({"start": current_start, "end": ts, "text": current_text})
                current_text = ""
                current_start = None
            continue

        if current_start is None:
            # Start a new segment
            current_text = text
            current_start = ts
        elif _text_similarity(text, current_text) >= SIMILARITY_THRESHOLD:
            # Same subtitle — extend (keep whichever has more chars, handles OCR noise)
            if len(text) > len(current_text):
                current_text = text
        else:
            # New subtitle — close current and start fresh
            segments.append({"start": current_start, "end": ts, "text": current_text})
            current_text = text
            current_start = ts

    # Close final open segment
    if current_text and current_start is not None:
        last_ts = ocr_results[-1][0] if ocr_results else current_start + 2.0
        segments.append({"start": current_start, "end": last_ts, "text": current_text})

    print(f"\n🧩  Built {len(segments)} subtitle segments from OCR frames.")
    return segments


# ─── Step 5: Translate ────────────────────────────────────────────────────────

def translate_segments(
    segments: list[dict],
    src_lang: str = "he",
    tgt_lang: str = "en",
) -> list[dict]:
    if src_lang == tgt_lang:
        print("\n🌐  Source and target language are the same — skipping translation.")
        for seg in segments:
            seg["translated"] = seg["text"]
        return segments

    model_name = f"Helsinki-NLP/opus-mt-tc-big-{src_lang}-{tgt_lang}"
    print(f"\n🌐  Translating ({src_lang} → {tgt_lang}) using {model_name}...")

    try:
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        print(f"   ⚠️  Could not load {model_name}: {e}")
        print("   ℹ️  Falling back to original text (no translation).")
        for seg in segments:
            seg["translated"] = seg["text"]
        return segments

    texts = [seg["text"] for seg in segments]
    print(f"   Translating {len(texts)} segments in batches...")

    batch_size = 16
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        if (i + batch_size) % 160 == 0 or i + batch_size >= len(texts):
            print(f"   {min(i + batch_size, len(texts))}/{len(texts)} segments translated...", end="\r")

    print()
    for seg, t in zip(segments, translated):
        seg["translated"] = t

    print(f"   ✅  Translated {len(segments)} segments.")
    return segments


# ─── Step 6: Write SRT ────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    ms     = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


def write_srt(segments: list[dict], output_path: Path) -> Path:
    print(f"\n📝  Writing subtitles to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = seg.get("translated") or seg["text"]
            f.write(f"{i}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{text.strip()}\n\n")
    print(f"   ✅  Wrote {len(segments)} subtitle entries.")
    return output_path


# ─── Step 7: Burn subtitles (optional) ───────────────────────────────────────

def burn_subtitles(video_path: Path, srt_path: Path) -> Path:
    output_path = OUTPUT_DIR / (video_path.stem + "_subbed.mp4")
    print(f"\n🎬  Burning subtitles into video → {output_path}...")
    safe_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{safe_srt}':force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Shadow=2,BorderStyle=3,BackColour=&H80000000'",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
    )
    print(f"   ✅  Output: {output_path}")
    return output_path


# ─── Step 8: Play ─────────────────────────────────────────────────────────────

def play_video(video_path: Path, srt_path: Path = None, burn: bool = False):
    print(f"\n▶️   Launching mpv...")
    cmd = [
        "mpv", "--fs",
        "--sub-font-size=40",
        "--sub-back-color=0.0/0.0/0.0/0.6",
        "--sub-border-size=3",
        "--sub-color=1.0/1.0/1.0/1.0",
        "--sub-pos=90",
    ]
    if srt_path and not burn:
        cmd += [f"--sub-file={srt_path}"]
    cmd.append(str(video_path))
    subprocess.Popen(cmd)
    print("   ✅  mpv launched. Share its window in Discord for screen share!")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract burned-in Hebrew subtitles via OCR, translate, and play.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--src-lang", default="he",
        help="Source language of burned-in subtitles (default: he).",
    )
    parser.add_argument(
        "--tgt-lang", default="en",
        help="Target language for translation (default: en).",
    )
    parser.add_argument(
        "--frame-interval", type=float, default=FRAME_INTERVAL,
        help=f"Seconds between sampled frames (default: {FRAME_INTERVAL}).\n"
             "Lower = more accurate timing but slower OCR.",
    )
    parser.add_argument(
        "--crop-fraction", type=float, default=SUBTITLE_CROP_FRACTION,
        help=f"Bottom fraction of frame to scan for subtitles (default: {SUBTITLE_CROP_FRACTION}).\n"
             "Increase if subs are higher up; decrease to reduce noise.",
    )
    parser.add_argument(
        "--no-ocr-cache", action="store_true",
        help="Ignore cached OCR results and re-run OCR from scratch.",
    )
    parser.add_argument(
        "--burn", action="store_true",
        help="Burn English subtitles into the video (slower). Default: soft subs via mpv.",
    )
    parser.add_argument(
        "--no-play", action="store_true",
        help="Don't auto-play the video after processing.",
    )
    parser.add_argument(
        "--subs-only", action="store_true",
        help="Only generate the .srt file, don't play or burn.",
    )
    parser.add_argument(
        "--ocr-only", action="store_true",
        help="Run OCR and print extracted Hebrew segments, then stop. No translation or SRT.",
    )
    parser.add_argument(
        "--debug-frame", type=float, default=None, metavar="SECONDS",
        help="Save the preprocessed OCR input image for one timestamp and exit. "
             "Example: --debug-frame 10.0 → saves debug_frame_10.0.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Debug mode: save preprocessed image + run Tesseract on one frame and exit
    if args.debug_frame is not None:
        video_path = download_video(args.url)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.debug_frame * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"❌  Could not read frame at ts={args.debug_frame}")
            return
        region = crop_subtitle_region(frame, args.crop_fraction)
        processed = preprocess_for_ocr(region)
        out = f"debug_frame_{args.debug_frame}.png"
        cv2.imwrite(out, processed)
        tess_config = "--oem 1 --psm 6 -l heb"
        tsv      = pytesseract.image_to_data(processed, config=tess_config, output_type=pytesseract.Output.STRING)
        filtered = extract_hebrew_from_data(tsv, processed.shape[0])
        print(f"✅  Saved preprocessed image to: {out}")
        print(f"📝  Tesseract filtered output:\n{filtered if filtered else '(nothing)'}")
        return

    # 1. Download
    video_path = download_video(args.url)

    # Prepare per-video OCR cache directory
    cache_dir = OCR_CACHE_DIR / video_path.stem
    cache_dir.mkdir(exist_ok=True)

    # 2+3. Extract frames and OCR in one streaming pass (frames are never saved to disk)
    ocr_results = extract_and_ocr(
        video_path,
        cache_dir,
        interval=args.frame_interval,
        crop_fraction=args.crop_fraction,
        use_cache=not args.no_ocr_cache,
    )

    # 4. Collapse OCR results into timed subtitle segments
    segments = build_segments(ocr_results)

    if not segments:
        print("\n⚠️  No subtitle segments found. Tips:")
        print("   • Try --crop-fraction 0.30 if subs are higher than usual")
        print("   • Try --frame-interval 0.1 for videos with very fast subtitles")
        print("   • Check that the video actually has burned-in subtitles")
        return

    # OCR preview — print segments and stop before translation
    if args.ocr_only:
        print(f"\n📋  OCR preview ({len(segments)} segments):\n")
        for seg in segments:
            print(f"  [{_ts(seg['start'])} --> {_ts(seg['end'])}]  {seg['text']}")
        print("\n✅  OCR-only run complete. Re-run without --ocr-only to translate.")
        return

    # 5. Translate
    segments = translate_segments(segments, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    # 6. Write SRT
    srt_path = OUTPUT_DIR / (video_path.stem + f".{args.tgt_lang}.srt")
    write_srt(segments, srt_path)

    if args.subs_only:
        print(f"\n✅  Done! Subtitle file: {srt_path}")
        return

    # 7. Optionally burn
    final_video = video_path
    if args.burn:
        final_video = burn_subtitles(video_path, srt_path)

    # 8. Play
    if not args.no_play:
        play_video(
            final_video,
            srt_path=srt_path if not args.burn else None,
            burn=args.burn,
        )

    print("\n🎉  All done!\n")


if __name__ == "__main__":
    main()
