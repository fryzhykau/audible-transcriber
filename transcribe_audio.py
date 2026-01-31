"""
Audio Transcription Script using OpenAI Whisper

Requirements:
    pip install openai-whisper tqdm

For GPU acceleration (optional):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Usage:
    python transcribe_audio.py <audio_file> [--model <model_size>] [--output <output_file>]

Model sizes: tiny, base, small, medium, large (larger = more accurate but slower)
"""

import argparse
import sys
import threading
from pathlib import Path

from tqdm import tqdm

def transcribe_audio(audio_path: str, model_size: str = "base", output_path: str = None):
    """Transcribe an audio file using OpenAI Whisper."""

    try:
        import whisper
    except ImportError:
        print("Error: whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    # Supported formats
    supported = {'.mp3', '.mp4', '.m4a', '.wav', '.flac', '.ogg', '.webm'}
    if audio_file.suffix.lower() not in supported:
        print(f"Error: Unsupported format '{audio_file.suffix}'")
        print(f"Supported formats: {', '.join(supported)}")
        print("\nNote: .aax files are DRM-protected and must be converted first.")
        sys.exit(1)

    # Check for GPU availability
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        print("Using CPU (no GPU detected - this will be slower)")

    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size, device=device)

    # Load audio to get duration for progress bar
    print(f"Loading audio: {audio_file.name}")
    audio = whisper.load_audio(str(audio_file))
    duration_seconds = len(audio) / whisper.audio.SAMPLE_RATE
    duration_str = format_timestamp(duration_seconds)
    print(f"Audio duration: {duration_str}")

    print(f"Transcribing...")

    # Transcribe with progress tracking using a background thread
    result = [None]
    error = [None]

    def transcribe_task():
        try:
            result[0] = model.transcribe(audio, verbose=False)
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
        import time
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

    result = result[0]

    # Determine output path
    if output_path is None:
        output_path = audio_file.with_suffix('.txt')
    else:
        output_path = Path(output_path)

    # Write transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript of: {audio_file.name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(result["text"])

    print(f"\nTranscript saved to: {output_path}")

    # Also save with timestamps if desired
    timestamps_path = output_path.with_stem(output_path.stem + "_timestamps")
    with open(timestamps_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript of: {audio_file.name} (with timestamps)\n")
        f.write("=" * 60 + "\n\n")
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} --> {end}] {text}\n")

    print(f"Timestamped transcript saved to: {timestamps_path}")

    return result

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper"
    )
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: same name with .txt extension)"
    )

    args = parser.parse_args()
    transcribe_audio(args.audio_file, args.model, args.output)

if __name__ == "__main__":
    main()
