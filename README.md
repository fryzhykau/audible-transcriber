# Audible Transcriber

Transcribe audio files and Audible audiobooks (AAX) to text using OpenAI Whisper.

## Features

- Transcribe general audio files (MP3, M4A, WAV, FLAC, OGG, WebM)
- Transcribe Audible AAX audiobooks (with automatic DRM handling)
- **Fast mode** - 4x faster transcription using faster-whisper backend
- GPU acceleration support (NVIDIA CUDA)
- Multiple output formats: plain text, timestamped text, SRT subtitles, Markdown, HTML
- Progress tracking during transcription
- **Detailed statistics** after transcription (timing, resource usage, output summary)

## Prerequisites

- Python 3.8+
- FFmpeg (required for AAX conversion)
  - Windows: `winget install ffmpeg` or `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or equivalent
- NVIDIA GPU (optional, for faster transcription)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fryzhykau/audible-transcriber.git
   cd audible-transcriber
   ```

2. Install dependencies:
   ```bash
   pip install openai-whisper audible tqdm psutil
   ```

3. For 4x faster transcription (recommended):
   ```bash
   pip install faster-whisper
   ```

4. For GPU acceleration:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### General Audio Transcription

```bash
python transcribe_audio.py <audio_file> [--model <model_size>] [--output <output_file>]
```

**Examples:**
```bash
# Basic usage
python transcribe_audio.py podcast.mp3

# Use a larger model for better accuracy
python transcribe_audio.py interview.wav --model medium

# Specify output file
python transcribe_audio.py lecture.m4a --output lecture_transcript.txt
```

### Audible Audiobook Transcription

```bash
python transcribe_audiobook.py <audiobook.aax> [options]
```

**Examples:**
```bash
# Basic usage (will prompt for Audible login on first run)
python transcribe_audiobook.py MyAudiobook.aax

# Fast mode - 4x faster transcription (recommended)
python transcribe_audiobook.py MyAudiobook.aax --fast

# High performance mode - maximum GPU utilization
python transcribe_audiobook.py MyAudiobook.aax --high-performance

# Use a specific model
python transcribe_audiobook.py MyAudiobook.aax --model large --fast

# If you already have activation bytes
python transcribe_audiobook.py MyAudiobook.aax --activation-bytes XXXXXXXX

# Skip conversion if you already have M4A
python transcribe_audiobook.py MyAudiobook.m4a --skip-convert --fast
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Model size: tiny, base, small, medium, large (default: medium) |
| `--fast` | Use faster-whisper backend (4x faster, same accuracy) |
| `--high-performance` | Fast mode with optimized settings (beam=10, batch=24) |
| `--beam-size` | Beam size for decoding (default: 5, higher = more GPU usage) |
| `--batch-size` | Batch size for faster-whisper (default: 16) |
| `--gpu-utilization` | Limit GPU memory usage (0.0-1.0, e.g., 0.8 for 80%) |
| `--skip-convert` | Skip AAX to M4A conversion |
| `--activation-bytes` | Provide Audible activation bytes directly |
| `--no-stats` | Skip displaying statistics after transcription |

## Model Sizes

| Model  | Parameters | Relative Speed | Accuracy |
|--------|------------|----------------|----------|
| tiny   | 39M        | ~32x           | Lower    |
| base   | 74M        | ~16x           | Basic    |
| small  | 244M       | ~6x            | Good     |
| medium | 769M       | ~2x            | Better   |
| large  | 1550M      | 1x             | Best     |

**Recommendation:** Use `medium` for audiobooks (good balance of speed and accuracy).

## Performance: Standard vs Fast Mode

The `--fast` flag uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which runs the **same Whisper models** through an optimized inference engine (CTranslate2).

| Mode | Backend | Speed | VRAM Usage | Accuracy |
|------|---------|-------|------------|----------|
| Standard | OpenAI Whisper (PyTorch) | 1x | ~5 GB | Baseline |
| `--fast` | faster-whisper (CTranslate2) | **4x** | ~2.5 GB | **Identical** |
| `--high-performance` | faster-whisper + tuned settings | **4.5x** | ~3.5 GB | Identical |

### Why faster-whisper is faster

- **Batched beam search** - Decodes multiple tokens in parallel
- **Optimized kernels** - Fused GPU operations via CTranslate2
- **VAD (Voice Activity Detection)** - Automatically skips silent portions
- **Efficient memory access** - Better GPU cache utilization

### Example: 1-hour audiobook on RTX 4070

| Mode | Transcription Time | Realtime Factor |
|------|-------------------|-----------------|
| Standard | ~4-5 minutes | 15x |
| `--fast` | ~1-1.5 minutes | 50x |
| `--high-performance` | ~45-60 seconds | 70x |

## Output Formats

The scripts generate multiple output files:

| Format | Extension | Description |
|--------|-----------|-------------|
| Plain text | `.txt` | Clean transcript without timestamps |
| Timestamped | `_timestamped.txt` | Transcript with time markers |
| SRT subtitles | `.srt` | Standard subtitle format |
| **Markdown** | `.md` | Structured document with TOC and chapters |
| **HTML** | `.html` | Interactive viewer with search and navigation |

### Structured Output Features

Both Markdown and HTML outputs include:

- **Table of Contents** - Quick navigation to sections
- **Chapter/Section Detection** - Automatically detects chapters from text patterns (e.g., "Chapter 1", "Part One", "Prologue")
- **Time-based Sections** - Falls back to time-based sections if no chapters detected
- **Metadata** - Source file, duration, model used, language detected
- **Timestamps** - Each segment includes precise timestamps

The HTML output additionally provides:
- **Search functionality** - Filter transcript by keywords
- **Sidebar navigation** - Sticky TOC for easy browsing
- **Responsive design** - Works on desktop and mobile
- **Dark theme** - Easy on the eyes

## Statistics Display

After transcription completes, detailed statistics are shown:

```
======================================================================
                    TRANSCRIPTION STATISTICS
======================================================================

┌─ TIMING ─────────────────────────────────────────────────────────┐
│  Audio Duration:            01:32:45                             │
│  Transcription Time:        00:01:15                             │
│  Speed:                       74.2x realtime                     │
│  Model Load Time:              2.1s                              │
└──────────────────────────────────────────────────────────────────┘

┌─ RESOURCE UTILIZATION ───────────────────────────────────────────┐
│  Device:               GPU (faster-whisper)                      │
│  GPU:                  NVIDIA GeForce RTX 4070 Laptop GPU        │
│  GPU Memory (Peak):         2.84 GB / 8.00 GB (35.5%)            │
│  CPU Usage (Avg):              12.3%                             │
│  RAM Usage (Peak):         14.20 GB / 63.7 GB (22.3%)            │
│  Model Size:                  medium                             │
│  Backend:             faster-whisper                             │
│  Beam Size:                       10                             │
│  Batch Size:                      24                             │
└──────────────────────────────────────────────────────────────────┘

┌─ OUTPUT SUMMARY ─────────────────────────────────────────────────┐
│  Total Words:                 18,432                             │
│  Total Characters:            98,210                             │
│  Segments:                     1,847                             │
│  Avg Segment Duration:          3.0s                             │
│  Words Per Minute:            199.5                              │
│  Detected Language:              en                              │
└──────────────────────────────────────────────────────────────────┘

┌─ FILES GENERATED ────────────────────────────────────────────────┐
│  audiobook.txt                                           124.2 KB │
│  audiobook_timestamped.txt                               198.4 KB │
│  audiobook.srt                                           215.1 KB │
│  audiobook.md                                            256.8 KB │
│  audiobook.html                                          312.5 KB │
│  ────────────────────────────────────────────────────────────────│
│  Total                                                     1.1 MB │
└──────────────────────────────────────────────────────────────────┘
```

Use `--no-stats` to skip this display.

## Notes

- First run will download the Whisper model (may take a few minutes)
- AAX transcription requires a valid Audible account (for activation bytes)
- Activation bytes are stored locally in `~/.audible_auth.json`
- This tool is intended for personal use of legally purchased audiobooks

## Disclaimer

This tool is for personal backup and accessibility purposes only. Users are responsible for ensuring their use complies with local laws and Audible's Terms of Service.

## License

MIT License - see [LICENSE](LICENSE) for details.
