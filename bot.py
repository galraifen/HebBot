#!/usr/bin/env python3
"""
yt-translator — Download, transcribe, translate, subtitle, and play YouTube videos.
Usage: python bot.py <youtube_url> [options]

Translation modes:
  Default (--src-lang / --tgt-lang):
    Whisper transcribes in the source language, then Helsinki-NLP translates.
    Supports any language pair. Requires two model downloads.

  Fast Hebrew→English (--whisper-translate):
    Whisper transcribes AND translates to English in one step.
    No Helsinki-NLP model needed. Faster, and often more accurate for Hebrew.
    Only works when the target language is English.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yt_dlp
import whisper
from transformers import pipeline as hf_pipeline


# ─── Config ───────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path("downloads")
OUTPUT_DIR    = Path("output")

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Step 1: Download ─────────────────────────────────────────────────────────

def download_video(url: str) -> Path:
    print("\n📥  Downloading video...")
    ydl_opts = {
        "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
        # bestvideo + bestaudio, but prefer mp4/m4a to avoid Opus audio
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        # Re-encode audio to AAC so VLC / Windows Media Player can play it
        "postprocessors": [{
            "key": "FFmpegAudioConvertor",
            "preferedcodec": "aac",
        }],
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Normalize extension after merge
        path = Path(filename).with_suffix(".mp4")
        if not path.exists():
            # Try the raw filename in case extension didn't change
            path = Path(filename)
    print(f"   ✅  Saved to: {path}")
    return path


# ─── Step 2: Transcribe ───────────────────────────────────────────────────────

def transcribe(
    video_path: Path,
    model_size: str = "turbo",
    language: str = None,
    whisper_translate: bool = False,
) -> tuple[list[dict], str]:
    """
    Transcribe audio from video_path using Whisper.

    whisper_translate=True:
        Uses Whisper's built-in translation task — transcribes AND translates
        to English in a single pass. Faster and more accurate for languages
        like Hebrew. Skips the Helsinki-NLP step entirely.
        ⚠️  Only outputs English. Don't use if you need another target language.

    whisper_translate=False (default):
        Transcribes in the source language only. The translate_segments()
        step will handle translation via Helsinki-NLP afterwards.
    """
    task = "translate" if whisper_translate else "transcribe"
    mode_label = "transcribe + translate to English" if whisper_translate else "transcribe only"
    print(f"\n🎙️  Whisper ({model_size}) — mode: {mode_label}...")

    model = whisper.load_model(model_size)
    result = model.transcribe(
        str(video_path),
        language=language,
        task=task,
        verbose=False,
    )
    segments = result["segments"]
    detected = result.get("language", "unknown")

    if whisper_translate:
        # Whisper already put English text in seg["text"] — copy to "translated"
        # so the rest of the pipeline works uniformly
        for seg in segments:
            seg["translated"] = seg["text"]
        print(f"   ✅  Detected language: {detected} | Segments: {len(segments)} | Already in English ✓")
    else:
        print(f"   ✅  Detected language: {detected} | Segments: {len(segments)}")

    return segments, detected


# ─── Step 3: Translate ────────────────────────────────────────────────────────

def translate_segments(
    segments: list[dict],
    src_lang: str,
    tgt_lang: str,
) -> list[dict]:
    if src_lang == tgt_lang:
        print("\n🌐  Source and target language are the same — skipping translation.")
        for seg in segments:
            seg["translated"] = seg["text"]
        return segments

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    print(f"\n🌐  Translating ({src_lang} → {tgt_lang}) using {model_name}...")

    try:
        translator = hf_pipeline("translation", model=model_name)
    except Exception as e:
        print(f"   ⚠️  Could not load {model_name}: {e}")
        print("   ℹ️  Falling back to original text (no translation).")
        for seg in segments:
            seg["translated"] = seg["text"]
        return segments

    texts = [seg["text"] for seg in segments]
    results = translator(texts, batch_size=16)

    for seg, res in zip(segments, results):
        seg["translated"] = res["translation_text"]

    print(f"   ✅  Translated {len(segments)} segments.")
    return segments


# ─── Step 4: Write SRT ────────────────────────────────────────────────────────

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


# ─── Step 5: Burn subtitles (optional) ───────────────────────────────────────

def burn_subtitles(video_path: Path, srt_path: Path) -> Path:
    output_path = OUTPUT_DIR / (video_path.stem + "_subbed.mp4")
    print(f"\n🎬  Burning subtitles into video → {output_path}...")
    # Escape colons/backslashes in path for ffmpeg filtergraph on Windows/WSL
    safe_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{safe_srt}':force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
    )
    print(f"   ✅  Output: {output_path}")
    return output_path


# ─── Step 6: Play ─────────────────────────────────────────────────────────────

def play_video(video_path: Path, srt_path: Path = None, burn: bool = False):
    print(f"\n▶️   Launching mpv...")
    cmd = ["mpv", "--fs"]
    if srt_path and not burn:
        cmd += [f"--sub-file={srt_path}"]
    cmd.append(str(video_path))
    subprocess.Popen(cmd)
    print("   ✅  mpv launched. Share its window in Discord for screen share!")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, translate, subtitle, and play a YouTube video.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--src-lang", default=None,
        help="Source language code (e.g. 'en', 'es'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tgt-lang", default="en",
        help="Target language for translation (default: en).",
    )
    parser.add_argument(
        "--whisper-model", default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: turbo).",
    )
    parser.add_argument(
        "--burn", action="store_true",
        help="Burn subtitles into the video (slower). Default: soft subs via mpv.",
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
        "--whisper-translate", action="store_true",
        help=(
            "Use Whisper's built-in translation to English.\n"
            "Faster and more accurate than Helsinki-NLP for many languages\n"
            "(especially Hebrew, Arabic, Japanese, etc.).\n"
            "⚠️  Always outputs English — don't use with --tgt-lang other than 'en'."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: --whisper-translate only makes sense targeting English
    if args.whisper_translate and args.tgt_lang != "en":
        print(f"⚠️  Warning: --whisper-translate always outputs English, but --tgt-lang is '{args.tgt_lang}'.")
        print("   Ignoring --tgt-lang and outputting English subtitles.")
        args.tgt_lang = "en"

    # 1. Download
    video_path = download_video(args.url)

    # 2. Transcribe (+ optionally translate via Whisper in one shot)
    segments, detected_lang = transcribe(
        video_path,
        model_size=args.whisper_model,
        language=args.src_lang,
        whisper_translate=args.whisper_translate,
    )
    src_lang = args.src_lang or detected_lang

    # 3. Translate via Helsinki-NLP (skipped if Whisper already translated)
    if args.whisper_translate:
        print("\n🌐  Skipping Helsinki-NLP — Whisper already translated to English.")
    else:
        segments = translate_segments(segments, src_lang=src_lang, tgt_lang=args.tgt_lang)

    # 4. Write SRT
    srt_path = OUTPUT_DIR / (video_path.stem + f".{args.tgt_lang}.srt")
    write_srt(segments, srt_path)

    if args.subs_only:
        print(f"\n✅  Done! Subtitle file: {srt_path}")
        return

    # 5. Optionally burn
    final_video = video_path
    if args.burn:
        final_video = burn_subtitles(video_path, srt_path)

    # 6. Play
    if not args.no_play:
        play_video(
            final_video,
            srt_path=srt_path if not args.burn else None,
            burn=args.burn,
        )

    print("\n🎉  All done!\n")


if __name__ == "__main__":
    main()
