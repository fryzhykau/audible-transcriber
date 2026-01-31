"""
Audiobook Transcription Script (AAX to Text)

For personal use of purchased audiobooks.

Requirements:
    pip install openai-whisper audible tqdm

You'll also need FFmpeg installed:
    - Windows: winget install ffmpeg  OR  choco install ffmpeg
    - Or download from: https://ffmpeg.org/download.html

Usage:
    python transcribe_audiobook.py <audiobook.aax> [--model medium]
"""

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from tqdm import tqdm


def get_activation_bytes():
    """
    Get Audible activation bytes using the audible library.
    You need to authenticate with your Audible account once.
    """
    try:
        import audible
    except ImportError:
        print("Installing audible library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "audible"], check=True)
        import audible

    auth_file = Path.home() / ".audible_auth.json"

    if auth_file.exists():
        auth = audible.Authenticator.from_file(auth_file)
    else:
        print("\n=== Audible Authentication ===")
        print("You need to log in to your Audible account once to get your activation bytes.")
        print("This is stored locally and used to decrypt YOUR purchased audiobooks.\n")

        auth = audible.Authenticator.from_login_external(locale="us")
        auth.to_file(auth_file)
        print(f"Auth saved to: {auth_file}")

    # Get activation bytes
    activation_bytes = auth.get_activation_bytes()
    return activation_bytes


def convert_aax_to_m4a(aax_path: Path, activation_bytes: str) -> Path:
    """Convert AAX to M4A using FFmpeg."""

    # Validate activation bytes to prevent command injection
    if not re.match(r'^[0-9a-fA-F]{8}$', activation_bytes):
        print("Error: Activation bytes must be exactly 8 hexadecimal characters")
        sys.exit(1)

    output_path = aax_path.with_suffix('.m4a')

    print(f"Converting {aax_path.name} to M4A...")

    # Find ffmpeg executable
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Error: FFmpeg not found. Please install it:")
        print("  Windows: winget install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        sys.exit(1)

    cmd = [
        ffmpeg_path, "-y",
        "-activation_bytes", activation_bytes,
        "-i", str(aax_path),
        "-c", "copy",  # No re-encoding, just copy streams
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        sys.exit(1)

    print(f"Converted to: {output_path}")
    return output_path


def transcribe_audio(audio_path: Path, model_size: str = "medium") -> dict:
    """Transcribe audio using Whisper."""

    try:
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper

    import torch

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nUsing GPU: {gpu_name}")
    else:
        device = "cpu"
        print("\nUsing CPU (no GPU detected - this will be slower)")

    print(f"Loading Whisper model '{model_size}'...")
    print("(First run will download the model, which may take a few minutes)\n")

    model = whisper.load_model(model_size, device=device)

    # Load audio to get duration for progress bar
    print(f"Loading audio: {audio_path.name}")
    audio = whisper.load_audio(str(audio_path))
    duration_seconds = len(audio) / whisper.audio.SAMPLE_RATE
    duration_str = format_timestamp(duration_seconds)
    print(f"Audio duration: {duration_str}")

    print(f"Transcribing...")

    # Transcribe with progress tracking using a background thread
    result = [None]
    error = [None]

    def transcribe_task():
        try:
            result[0] = model.transcribe(audio, verbose=False, language="en")
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=transcribe_task)
    thread.start()

    # Estimate processing speed: GPU processes ~15-30x realtime, CPU ~1-5x realtime
    if device == "cuda":
        speed_factor = 20  # Conservative estimate for GPU
    else:
        speed_factor = 2   # Conservative estimate for CPU

    estimated_time = duration_seconds / speed_factor

    with tqdm(total=100, desc="Transcribing", unit="%", ncols=80,
              bar_format="{l_bar}{bar}| {n:.0f}% [{elapsed}<{remaining}]") as pbar:
        start_time = time.time()
        while thread.is_alive():
            thread.join(timeout=0.3)
            elapsed = time.time() - start_time
            # Progress based on elapsed time vs estimated time (cap at 99% until done)
            progress = min(99, (elapsed / estimated_time) * 100)
            pbar.n = progress
            pbar.refresh()
        pbar.n = 100
        pbar.refresh()

    if error[0]:
        raise error[0]

    return result[0]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_transcript(result: dict, audio_path: Path):
    """Save transcript to text files."""

    # Plain text transcript
    txt_path = audio_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript: {audio_path.stem}\n")
        f.write("=" * 60 + "\n\n")
        f.write(result["text"])
    print(f"\nPlain transcript saved to: {txt_path}")

    # Timestamped transcript
    ts_path = audio_path.parent / f"{audio_path.stem}_timestamped.txt"
    with open(ts_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript: {audio_path.stem} (with timestamps)\n")
        f.write("=" * 60 + "\n\n")
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} --> {end}]\n{text}\n\n")
    print(f"Timestamped transcript saved to: {ts_path}")

    # SRT subtitle format (can be used with video players)
    srt_path = audio_path.with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result["segments"], 1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"SRT subtitles saved to: {srt_path}")


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Audible audiobooks (AAX) to text"
    )
    parser.add_argument("audiobook", help="Path to the .aax audiobook file")
    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium, recommended for audiobooks)"
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion if you already have an M4A/MP3 file"
    )
    parser.add_argument(
        "--activation-bytes",
        help="Provide activation bytes directly (optional)"
    )

    args = parser.parse_args()

    audio_path = Path(args.audiobook)

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    # Convert AAX if needed
    if audio_path.suffix.lower() == '.aax' and not args.skip_convert:
        if args.activation_bytes:
            activation_bytes = args.activation_bytes
            # Validate user-provided activation bytes
            if not re.match(r'^[0-9a-fA-F]{8}$', activation_bytes):
                print("Error: Activation bytes must be exactly 8 hexadecimal characters")
                sys.exit(1)
        else:
            activation_bytes = get_activation_bytes()

        audio_path = convert_aax_to_m4a(audio_path, activation_bytes)

    # Transcribe
    result = transcribe_audio(audio_path, args.model)

    # Save transcripts
    save_transcript(result, audio_path)

    print("\n" + "=" * 60)
    print("Transcription complete!")


if __name__ == "__main__":
    main()
