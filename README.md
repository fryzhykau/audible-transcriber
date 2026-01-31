# Audible Transcriber

Transcribe audio files and Audible audiobooks (AAX) to text using OpenAI Whisper.

## Features

- Transcribe general audio files (MP3, M4A, WAV, FLAC, OGG, WebM)
- Transcribe Audible AAX audiobooks (with automatic DRM handling)
- GPU acceleration support (NVIDIA CUDA)
- Multiple output formats: plain text, timestamped text, SRT subtitles
- Progress tracking during transcription

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
   pip install -r requirements.txt
   ```

3. For GPU acceleration (optional):
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
python transcribe_audiobook.py <audiobook.aax> [--model <model_size>]
```

**Examples:**
```bash
# Basic usage (will prompt for Audible login on first run)
python transcribe_audiobook.py MyAudiobook.aax

# Use a specific model
python transcribe_audiobook.py MyAudiobook.aax --model large

# If you already have activation bytes
python transcribe_audiobook.py MyAudiobook.aax --activation-bytes XXXXXXXX

# Skip conversion if you already have M4A
python transcribe_audiobook.py MyAudiobook.m4a --skip-convert
```

## Model Sizes

| Model  | Parameters | Relative Speed | Accuracy |
|--------|------------|----------------|----------|
| tiny   | 39M        | ~32x           | Lower    |
| base   | 74M        | ~16x           | Basic    |
| small  | 244M       | ~6x            | Good     |
| medium | 769M       | ~2x            | Better   |
| large  | 1550M      | 1x             | Best     |

**Recommendation:** Use `medium` for audiobooks (good balance of speed and accuracy).

## Output Formats

The scripts generate multiple output files:

- **Plain text** (`.txt`): Clean transcript without timestamps
- **Timestamped text** (`_timestamped.txt`): Transcript with time markers
- **SRT subtitles** (`.srt`): Standard subtitle format (audiobook script only)

## Notes

- First run will download the Whisper model (may take a few minutes)
- AAX transcription requires a valid Audible account (for activation bytes)
- Activation bytes are stored locally in `~/.audible_auth.json`
- This tool is intended for personal use of legally purchased audiobooks

## Disclaimer

This tool is for personal backup and accessibility purposes only. Users are responsible for ensuring their use complies with local laws and Audible's Terms of Service.

## License

MIT License - see [LICENSE](LICENSE) for details.
