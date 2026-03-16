#!/usr/bin/env python3
"""
run_batch.py — Nightly batch runner for yt-translator.

Reads URLs from episodes.txt, skips already-processed videos,
and runs bot.py on each one. Logs results to logs/nightly.log.

Usage:
    python3 run_batch.py                        # uses default episodes.txt
    python3 run_batch.py --file my_list.txt     # custom episode list
    python3 run_batch.py --dry-run              # preview what would run
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("output")
LOGS_DIR     = Path("logs")
DEFAULT_LIST = Path("episodes.txt")

# These are passed through to bot.py for every run — edit to taste
BOT_DEFAULTS = [
    "--whisper-translate",
    "--whisper-model", "turbo",
    "--no-play",        # never auto-play during batch runs
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, logfile):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    logfile.write(line + "\n")
    logfile.flush()


def load_urls(list_path: Path) -> list[str]:
    if not list_path.exists():
        print(f"❌  Episode list not found: {list_path}")
        print(f"   Create it with one YouTube URL per line.")
        sys.exit(1)

    urls = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


DOWNLOADS_DIR = Path("downloads")


def get_video_info(url: str) -> tuple[str, str]:
    """
    Ask yt-dlp for the title and expected filename without downloading.
    Returns (title, expected_mp4_filename).
    """
    try:
        result = subprocess.run(
            ["python3", "-m", "yt_dlp",
             "--print", "%(title)s",
             "--print", "%(title)s.mp4",
             "--no-playlist", url],
            capture_output=True, text=True, timeout=15
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            return lines[0].strip(), lines[1].strip()
        return "", ""
    except Exception:
        return "", ""


def already_processed(url: str) -> tuple[bool, str, bool]:
    """
    Check what work has already been done for this URL.
    Returns (fully_done, title, download_cached).

    fully_done      — .srt exists in output/, skip entirely
    download_cached — video exists in downloads/, skip download step
    """
    title, filename = get_video_info(url)
    if not title:
        return False, "", False

    # Check if final .srt already exists in output/
    for f in OUTPUT_DIR.glob("*.srt"):
        if f.stem.startswith(title[:30]):
            return True, title, True

    # Check if video already downloaded — compare stem only, ignore extension
    title_stem = Path(filename).stem.lower()
    download_cached = any(
        f.stem.lower() == title_stem
        for f in DOWNLOADS_DIR.iterdir()
        if f.is_file()
    )

    return False, title, download_cached


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nightly batch runner for yt-translator.")
    parser.add_argument("--file", type=Path, default=DEFAULT_LIST, help="Path to episode list (default: episodes.txt)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would run without doing anything")
    parser.add_argument("--bot-args", nargs=argparse.REMAINDER, help="Extra args to pass to bot.py (overrides BOT_DEFAULTS)")
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    bot_args = args.bot_args if args.bot_args else BOT_DEFAULTS
    urls = load_urls(args.file)

    log_path = LOGS_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_path, "w") as logfile:
        log(f"{'[DRY RUN] ' if args.dry_run else ''}Starting batch — {len(urls)} URLs from {args.file}", logfile)
        log(f"Bot args: {' '.join(bot_args)}", logfile)
        log("-" * 60, logfile)

        success, skipped, failed = 0, 0, 0

        for i, url in enumerate(urls, 1):
            log(f"[{i}/{len(urls)}] {url}", logfile)

            # Check if already done
            done, title, download_cached = already_processed(url)
            if done:
                log(f"   ⏭️  Skipping — already processed: {title}", logfile)
                skipped += 1
                continue

            if args.dry_run:
                status = "download cached, needs transcription" if download_cached else "needs download + transcription"
                log(f"   🔍  Would process ({status}): {title or '(title unknown)'}", logfile)
                continue

            if download_cached:
                log(f"   📦  Download cached, skipping download: {title}", logfile)

            # Run bot.py
            cmd = [sys.executable, "bot.py", url] + bot_args
            log(f"   ▶️  Running: {' '.join(cmd)}", logfile)

            try:
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    log(f"   ✅  Done.", logfile)
                    success += 1
                else:
                    log(f"   ❌  bot.py exited with code {result.returncode} — skipping.", logfile)
                    failed += 1
            except Exception as e:
                log(f"   ❌  Unexpected error: {e} — skipping.", logfile)
                failed += 1

        log("-" * 60, logfile)
        log(f"Batch complete — ✅ {success} done  ⏭️ {skipped} skipped  ❌ {failed} failed", logfile)
        log(f"Full log: {log_path}", logfile)


if __name__ == "__main__":
    main()
