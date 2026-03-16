# 🎬 yt-translator

> Download any YouTube video, auto-transcribe it with Whisper, translate subtitles with AI, and play it — all in one command.

---

## 📦 Prerequisites

Install system dependencies (WSL/Ubuntu):

```bash
sudo apt update && sudo apt install -y ffmpeg mpv python3-pip python3-venv
```

---

## ⚙️ Setup

```bash
# 1. Clone / enter the project folder
cd yt-translator

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

> ⚠️ First run will download Whisper and Helsinki-NLP models (~1–3 GB depending on size). This is cached locally for future runs.

---

## 🌐 Translation Modes — Which Should I Use?

There are two ways the bot can translate subtitles. Choosing the right one matters for speed and accuracy.

### Mode 1: `--whisper-translate` (fast, Hebrew-friendly)

```
Audio → [Whisper] → English text  (one step, one model)
```

Whisper does **everything in one pass** — it listens to the audio and outputs English text directly, without a separate translation model.

**Use this when:**
- Your video is in Hebrew, Arabic, Japanese, Chinese, or any non-Latin language → English
- You want the fastest possible result
- You don't want to download a second translation model
- You only ever need English subtitles

**Limitations:**
- ⚠️ Only outputs **English**. Cannot translate to French, Spanish, etc.

```bash
# Hebrew video → English subtitles (fast path)
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate

# You can still hint the source language to help Whisper
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate --src-lang he
```

---

### Mode 2: Default — Whisper + Helsinki-NLP (flexible, any language pair)

```
Audio → [Whisper] → Hebrew text → [Helsinki-NLP] → French text  (two steps, two models)
```

Whisper transcribes in the original language, then a separate Helsinki-NLP model translates to your chosen target language.

**Use this when:**
- You need to translate to a language **other than English** (e.g. Spanish → French)
- You want to keep the original-language `.srt` as well
- The source language is well-supported by Helsinki-NLP

**Limitations:**
- Requires downloading a Helsinki-NLP model (~300MB, cached after first use)
- Not all language pairs exist — check [Helsinki-NLP on HuggingFace](https://huggingface.co/Helsinki-NLP)

```bash
# Spanish video → French subtitles
python bot.py "https://youtu.be/VIDEO_ID" --src-lang es --tgt-lang fr

# Hebrew video → English subtitles (slower alternative to --whisper-translate)
python bot.py "https://youtu.be/VIDEO_ID" --src-lang he --tgt-lang en
```

---

### Quick Decision Guide

| Your situation | Use |
|---|---|
| Hebrew / Arabic / Japanese → English | `--whisper-translate` ✅ |
| Any language → English, want speed | `--whisper-translate` ✅ |
| Spanish → French (non-English target) | Default mode (`--src-lang es --tgt-lang fr`) |
| Need subtitles in German, Italian, etc. | Default mode |
| Not sure what language the video is | `--whisper-translate` (Whisper auto-detects) |

---

## 🚀 Usage Examples

### Basic — download, transcribe, translate to English, and play:
```bash
python bot.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Hebrew video → English subtitles (recommended fast path):
```bash
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate
```

### Translate a Spanish video to French:
```bash
python bot.py "https://youtu.be/VIDEO_ID" --src-lang es --tgt-lang fr
```

### Use a larger Whisper model for better accuracy:
```bash
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate --whisper-model large
```

### Burn subtitles into the video file:
```bash
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate --burn
```

### Only generate the .srt subtitle file (no playback):
```bash
python bot.py "https://youtu.be/VIDEO_ID" --whisper-translate --subs-only
```

### Download and translate without auto-playing:
```bash
python bot.py "https://youtu.be/VIDEO_ID" --no-play
```

---

## 🎛️ All Options

| Flag | Default | Description |
|---|---|---|
| `url` | *(required)* | YouTube video URL |
| `--whisper-translate` | off | Use Whisper's built-in translation (→ English only, faster) |
| `--src-lang` | auto-detect | Source language code (`he`, `en`, `es`, `fr`, `ja`, etc.) |
| `--tgt-lang` | `en` | Target language for translation (Helsinki-NLP mode only) |
| `--whisper-model` | `turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `turbo` |
| `--burn` | off | Burn subtitles into the video (requires ffmpeg re-encode) |
| `--no-play` | off | Skip auto-playing the video |
| `--subs-only` | off | Only generate the `.srt` file |

---

## 🗂️ Output Structure

```
yt-translator/
├── bot.py
├── requirements.txt
├── downloads/          ← raw downloaded videos
└── output/             ← .srt files and (optionally) burned videos
```

---

## 🌍 Supported Language Codes

Common codes for `--src-lang` / `--tgt-lang`:

`he` Hebrew · `en` English · `es` Spanish · `fr` French · `de` German · `it` Italian · `pt` Portuguese · `ja` Japanese · `zh` Chinese · `ko` Korean · `ar` Arabic · `ru` Russian · `hi` Hindi

Helsinki-NLP supports 1,000+ pairs. If a direct pair like `ja→fr` doesn't exist, you may need to pivot through English (`ja→en`, then `en→fr`).

---

## 🖥️ Discord Screen Share

1. Run the bot — `mpv` will open the video fullscreen
2. In Discord, go to your voice channel → **Share Screen**
3. Select the `mpv` window
4. Done! Others will see the video with subtitles

> **Tip**: Use `--burn` mode if Discord's screen share compresses the video and makes soft subs look blurry.

---

## ⚡ Performance Tips

| Situation | Recommendation |
|---|---|
| Fast laptop / short video | `--whisper-model turbo` (default) |
| Best accuracy | `--whisper-model large` |
| Low-end machine | `--whisper-model small` or `tiny` |
| Hebrew/Arabic/CJK → English | Always use `--whisper-translate` |
| GPU available | Torch will auto-detect CUDA — much faster |
| Re-watching same video | SRT is cached in `output/` — edit `bot.py` to skip retranscription |
| Nightly batch, best quality | Use `turbo` or `large` model in `BOT_DEFAULTS` in `run_batch.py` — let it run overnight |
| Nightly batch, low-end machine | Use `small` model — faster but less accurate for Hebrew |

### Recommended nightly batch setup for Hebrew

If the `small` model isn't accurate enough, switch to `turbo` or `large` in `run_batch.py`:

```python
BOT_DEFAULTS = [
    "--whisper-translate",
    "--whisper-model", "turbo",   # or "large" for best accuracy
    "--no-play",
]
```

Then make sure WSL has enough memory in `C:\Users\<YourName>\.wslconfig`:
```ini
[wsl2]
memory=10GB
processors=4
```

A rough estimate of overnight processing time on CPU:

| Model | Time per 30min episode |
|---|---|
| `small` | ~5 min |
| `turbo` | ~30 min |
| `large` | ~60 min |

So with `turbo`, a batch of 10 episodes takes roughly 5 hours — easily fits in an overnight run.

---

## 🐛 Troubleshooting

**`mpv` not found**: `sudo apt install mpv`

**`ffmpeg` not found**: `sudo apt install ffmpeg`

**Helsinki model not found for language pair**: Check available models at https://huggingface.co/Helsinki-NLP — not all pairs exist directly. Try `--whisper-translate` instead if targeting English.

**WSL display issues with mpv**: Make sure you have an X server running (e.g. [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or use WSL2 with WSLg enabled on Windows 11).

---

## 🌙 Nightly Batch Runs

### Setup

1. Fill `episodes.txt` with one YouTube URL per line:
```
# comments are ignored
https://youtu.be/episode1
https://youtu.be/episode2
```

2. Run the batch manually to test:
```bash
python3 run_batch.py --dry-run    # preview without doing anything
python3 run_batch.py              # actually run
```

Already-processed episodes are automatically skipped (checks if a matching `.srt` exists in `output/`).

---

### Changing batch settings

Edit the `BOT_DEFAULTS` list at the top of `run_batch.py` to change the whisper model, language, etc. for all batch runs:

```python
BOT_DEFAULTS = [
    "--whisper-translate",
    "--whisper-model", "small",   # change to "turbo" or "large" if you want
    "--no-play",
]
```

---

### Schedule with cron (runs every night at 2am)

```bash
crontab -e
```

Add this line:
```
0 2 * * * cd /home/am/Projects/HebBot && .venv/bin/python3 run_batch.py >> logs/cron.log 2>&1
```

> ⚠️ Replace `/home/am/Projects/HebBot` with your actual project path.

Each run creates a timestamped log in `logs/` (e.g. `logs/batch_20260313_020001.log`).

---

### Project structure with batch

```
yt-translator/
├── bot.py              ← single video processor
├── run_batch.py        ← batch runner
├── episodes.txt        ← your queue of URLs
├── requirements.txt
├── downloads/          ← raw downloaded videos
├── output/             ← ready-to-watch .srt + videos
└── logs/               ← one log file per batch run
```
