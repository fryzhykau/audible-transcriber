"""
Audiobook Transcription Script (AAX to Text)

For personal use of purchased audiobooks.

Requirements:
    pip install openai-whisper audible tqdm psutil

    For faster transcription (recommended, 4x speed boost):
        pip install faster-whisper

You'll also need FFmpeg installed:
    - Windows: winget install ffmpeg  OR  choco install ffmpeg
    - Or download from: https://ffmpeg.org/download.html

Usage:
    python transcribe_audiobook.py <audiobook.aax> [--model medium]
    python transcribe_audiobook.py <audiobook.aax> --fast   # 4x faster with same model

Options:
    --model, -m           Model size: tiny, base, small, medium, large (default: medium)
    --skip-convert        Skip AAX conversion if you already have M4A/MP3
    --activation-bytes    Provide Audible activation bytes directly
    --no-stats            Skip displaying statistics after transcription

GPU Performance Options:
    --fast                Use faster-whisper backend (4x faster, same accuracy)
    --high-performance    Combines --fast with optimized GPU settings
    --gpu-utilization     GPU memory fraction (0.0-1.0). E.g., 0.8 for 80%
    --beam-size           Beam size for search (default: 5)
"""

import argparse
import html as html_lib
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
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


def transcribe_audio_fast(audio_path: Path, model_size: str = "medium",
                          beam_size: int = 5, batch_size: int = 16) -> tuple:
    """Transcribe audio using faster-whisper (CTranslate2). Returns (result, duration_seconds, stats)."""

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Installing faster-whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"], check=True)
        from faster_whisper import WhisperModel

    import torch

    stats = {
        'device': None,
        'gpu_name': None,
        'gpu_memory_used': None,
        'gpu_memory_total': None,
        'cpu_percent': None,
        'memory_used_gb': None,
        'memory_total_gb': None,
        'transcription_time': None,
        'model_size': model_size,
        'realtime_factor': None,
        'beam_size': beam_size,
        'batch_size': batch_size,
        'backend': 'faster-whisper',
    }

    # Check for GPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"  # Use FP16 for speed
        gpu_name = torch.cuda.get_device_name(0)
        stats['device'] = 'GPU (faster-whisper)'
        stats['gpu_name'] = gpu_name
        stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nUsing GPU: {gpu_name} (faster-whisper backend)")
    else:
        device = "cpu"
        compute_type = "int8"  # Use INT8 on CPU for speed
        stats['device'] = 'CPU (faster-whisper)'
        print("\nUsing CPU with INT8 quantization (faster-whisper backend)")

    print(f"Loading faster-whisper model '{model_size}'...")
    print("(First run will download/convert the model)\n")

    model_load_start = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    stats['model_load_time'] = time.time() - model_load_start

    # Get audio duration
    import subprocess as sp
    ffprobe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    try:
        duration_seconds = float(sp.check_output(ffprobe_cmd, stderr=sp.DEVNULL).decode().strip())
    except Exception:
        duration_seconds = 0

    print(f"Audio: {audio_path.name}")
    print(f"Duration: {format_timestamp(duration_seconds)}")
    print(f"Transcribing with batch_size={batch_size}, beam_size={beam_size}...")

    # Track resources during transcription
    cpu_samples = []
    memory_samples = []
    peak_gpu_memory = [0]

    segments_result = []
    full_text = []
    error = [None]
    detected_language = [None]

    def transcribe_task():
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language="en",
                beam_size=beam_size,
                vad_filter=True,  # Voice activity detection for faster processing
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            detected_language[0] = info.language
            for segment in segments:
                segments_result.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                })
                full_text.append(segment.text)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=transcribe_task)
    transcribe_start = time.time()
    thread.start()

    # faster-whisper is ~4x faster
    speed_factor = 80 if device == "cuda" else 8
    estimated_time = max(duration_seconds / speed_factor, 5)

    with tqdm(total=100, desc="Transcribing", unit="%", ncols=80,
              bar_format="{l_bar}{bar}| {n:.0f}% [{elapsed}<{remaining}]") as pbar:
        start_time = time.time()
        while thread.is_alive():
            thread.join(timeout=0.3)
            elapsed = time.time() - start_time
            progress = min(99, (elapsed / estimated_time) * 100)
            pbar.n = progress
            pbar.refresh()

            cpu_samples.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            memory_samples.append(mem.used / (1024**3))

            if device == "cuda":
                try:
                    current_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                    peak_gpu_memory[0] = max(peak_gpu_memory[0], current_gpu_mem)
                except Exception:
                    pass

        pbar.n = 100
        pbar.refresh()

    transcribe_end = time.time()
    stats['transcription_time'] = transcribe_end - transcribe_start

    if cpu_samples:
        stats['cpu_percent'] = sum(cpu_samples) / len(cpu_samples)
    if memory_samples:
        stats['memory_used_gb'] = max(memory_samples)
    stats['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)

    if device == "cuda":
        stats['gpu_memory_used'] = peak_gpu_memory[0]
        try:
            stats['gpu_memory_used'] = max(stats['gpu_memory_used'], torch.cuda.max_memory_allocated() / (1024**3))
        except Exception:
            pass

    if duration_seconds > 0:
        stats['realtime_factor'] = duration_seconds / stats['transcription_time']

    if error[0]:
        raise error[0]

    # Format result to match openai-whisper structure
    result = {
        "text": "".join(full_text),
        "segments": segments_result,
        "language": detected_language[0] or "en",
    }

    return result, duration_seconds, stats


def transcribe_audio(audio_path: Path, model_size: str = "medium",
                     gpu_memory_fraction: float = None, beam_size: int = 5,
                     best_of: int = 5) -> tuple:
    """Transcribe audio using OpenAI Whisper. Returns (result, duration_seconds, stats)."""

    try:
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
        import whisper

    import torch

    # Initialize stats dictionary
    stats = {
        'device': None,
        'gpu_name': None,
        'gpu_memory_used': None,
        'gpu_memory_total': None,
        'gpu_memory_fraction': gpu_memory_fraction,
        'cpu_percent': None,
        'memory_used_gb': None,
        'memory_total_gb': None,
        'transcription_time': None,
        'model_size': model_size,
        'realtime_factor': None,
        'beam_size': beam_size,
        'best_of': best_of,
    }

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        stats['device'] = 'GPU'
        stats['gpu_name'] = gpu_name
        stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Set GPU memory fraction limit if specified
        if gpu_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
            print(f"\nUsing GPU: {gpu_name} (memory capped at {gpu_memory_fraction*100:.0f}%)")
        else:
            print(f"\nUsing GPU: {gpu_name}")

        # Enable TF32 for better performance on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmark for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True

    else:
        device = "cpu"
        stats['device'] = 'CPU'
        print("\nUsing CPU (no GPU detected - this will be slower)")

    print(f"Loading Whisper model '{model_size}'...")
    print("(First run will download the model, which may take a few minutes)\n")

    model_load_start = time.time()
    model = whisper.load_model(model_size, device=device)
    stats['model_load_time'] = time.time() - model_load_start

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
    peak_gpu_memory = [0]
    cpu_samples = []
    memory_samples = []

    def transcribe_task():
        try:
            # Use higher beam_size and best_of for better GPU utilization and accuracy
            # beam_size: number of beams in beam search (more = more GPU work)
            # best_of: number of candidates when sampling (more = more GPU work)
            result[0] = model.transcribe(
                audio,
                verbose=False,
                language="en",
                beam_size=beam_size,
                best_of=best_of,
                fp16=(device == "cuda"),  # Use FP16 on GPU for speed
            )
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=transcribe_task)
    transcribe_start = time.time()
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

            # Sample resource utilization
            cpu_samples.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            memory_samples.append(mem.used / (1024**3))

            if device == "cuda":
                try:
                    current_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                    peak_gpu_memory[0] = max(peak_gpu_memory[0], current_gpu_mem)
                except Exception:
                    pass

        pbar.n = 100
        pbar.refresh()

    transcribe_end = time.time()
    stats['transcription_time'] = transcribe_end - transcribe_start

    # Calculate resource stats
    if cpu_samples:
        stats['cpu_percent'] = sum(cpu_samples) / len(cpu_samples)
    if memory_samples:
        stats['memory_used_gb'] = max(memory_samples)
    stats['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)

    if device == "cuda":
        stats['gpu_memory_used'] = peak_gpu_memory[0]
        try:
            stats['gpu_memory_used'] = max(stats['gpu_memory_used'], torch.cuda.max_memory_allocated() / (1024**3))
        except Exception:
            pass

    # Calculate realtime factor (how much faster/slower than realtime)
    if duration_seconds > 0:
        stats['realtime_factor'] = duration_seconds / stats['transcription_time']

    if error[0]:
        raise error[0]

    return result[0], duration_seconds, stats


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_chapters(segments: list, min_gap: float = 3.0) -> list:
    """
    Detect chapter breaks based on:
    1. Text patterns like "Chapter X", "Part X", etc.
    2. Long pauses between segments (silence gaps)

    Returns list of chapter dicts with 'title', 'start_time', 'start_segment_idx'
    """
    chapters = []
    chapter_patterns = [
        r'^chapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^part\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^section\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^book\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^prologue',
        r'^epilogue',
        r'^introduction',
        r'^conclusion',
        r'^preface',
        r'^afterword',
    ]

    for i, segment in enumerate(segments):
        text = segment["text"].strip().lower()

        # Check for chapter patterns in text
        for pattern in chapter_patterns:
            if re.match(pattern, text):
                # Extract a meaningful title from the segment
                title = segment["text"].strip()
                # Truncate if too long
                if len(title) > 60:
                    title = title[:57] + "..."
                chapters.append({
                    'title': title,
                    'start_time': segment["start"],
                    'start_segment_idx': i,
                    'type': 'detected'
                })
                break

        # Check for long gaps (potential chapter breaks)
        if i > 0:
            gap = segment["start"] - segments[i - 1]["end"]
            if gap >= min_gap and len(chapters) > 0:
                # Only add gap-based breaks if we've detected at least one chapter
                # to avoid false positives at the beginning
                pass  # Could add logic here for gap-based chapter detection

    # If no chapters detected, create time-based sections (every 30 min)
    if not chapters:
        total_duration = segments[-1]["end"] if segments else 0
        section_length = 1800  # 30 minutes
        section_num = 1

        for i, segment in enumerate(segments):
            if segment["start"] >= (section_num - 1) * section_length and \
               (section_num == 1 or segment["start"] >= section_num * section_length - section_length):
                if section_num == 1 or segment["start"] >= (section_num - 1) * section_length:
                    if section_num == 1:
                        chapters.append({
                            'title': f"Section {section_num}",
                            'start_time': segment["start"],
                            'start_segment_idx': i,
                            'type': 'time-based'
                        })
                        section_num += 1
                    elif segment["start"] >= (section_num - 1) * section_length:
                        chapters.append({
                            'title': f"Section {section_num}",
                            'start_time': segment["start"],
                            'start_segment_idx': i,
                            'type': 'time-based'
                        })
                        section_num += 1

    return chapters


def save_transcript(result: dict, audio_path: Path, duration_seconds: float, model_size: str) -> list:
    """Save transcript to multiple formats. Returns list of output file paths."""

    output_files = []
    segments = result["segments"]
    chapters = detect_chapters(segments)

    # Plain text transcript
    txt_path = audio_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript: {audio_path.stem}\n")
        f.write("=" * 60 + "\n\n")
        f.write(result["text"])
    print(f"\nPlain transcript saved to: {txt_path}")
    output_files.append(txt_path)

    # Timestamped transcript
    ts_path = audio_path.parent / f"{audio_path.stem}_timestamped.txt"
    with open(ts_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript: {audio_path.stem} (with timestamps)\n")
        f.write("=" * 60 + "\n\n")
        for segment in segments:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} --> {end}]\n{text}\n\n")
    print(f"Timestamped transcript saved to: {ts_path}")
    output_files.append(ts_path)

    # SRT subtitle format
    srt_path = audio_path.with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"SRT subtitles saved to: {srt_path}")
    output_files.append(srt_path)

    # Markdown with chapters and TOC
    md_path = save_markdown(result, audio_path, duration_seconds, model_size, chapters)
    output_files.append(md_path)

    # HTML with chapters and TOC
    html_path = save_html(result, audio_path, duration_seconds, model_size, chapters)
    output_files.append(html_path)

    return output_files


def segments_to_paragraphs(segments: list, sentences_per_paragraph: int = 4) -> list:
    """Convert segments into flowing paragraphs with timing data.

    Returns list of dicts with 'text', 'start_time', 'end_time' keys.
    """
    paragraphs = []
    current_paragraph = []
    current_start_time = None
    current_end_time = None
    sentence_count = 0

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue

        if current_start_time is None:
            current_start_time = segment.get("start", 0)
        current_end_time = segment.get("end", segment.get("start", 0))

        current_paragraph.append(text)

        # Count sentence endings
        sentence_count += text.count('.') + text.count('!') + text.count('?')

        # Start new paragraph after enough sentences or at natural breaks
        if sentence_count >= sentences_per_paragraph:
            paragraphs.append({
                'text': ' '.join(current_paragraph),
                'start_time': current_start_time,
                'end_time': current_end_time
            })
            current_paragraph = []
            current_start_time = None
            sentence_count = 0

    # Don't forget remaining text
    if current_paragraph:
        paragraphs.append({
            'text': ' '.join(current_paragraph),
            'start_time': current_start_time or 0,
            'end_time': current_end_time or 0
        })

    return paragraphs


def save_markdown(result: dict, audio_path: Path, duration_seconds: float, model_size: str, chapters: list):
    """Save transcript as a clean, book-like Markdown document."""
    md_path = audio_path.with_suffix('.md')
    segments = result["segments"]

    # Create a clean title from filename
    title = audio_path.stem.replace('_', ' ').replace('-', ' ')

    with open(md_path, 'w', encoding='utf-8') as f:
        # Title
        f.write(f"# {title}\n\n")

        # Table of Contents (clean, no timestamps)
        if len(chapters) > 1:
            f.write("## Contents\n\n")
            for i, chapter in enumerate(chapters):
                anchor = f"chapter-{i + 1}"
                f.write(f"{i + 1}. [{chapter['title']}](#{anchor})\n")
            f.write("\n---\n\n")

        # Chapters with content
        for i, chapter in enumerate(chapters):
            anchor = f"chapter-{i + 1}"

            # Chapter heading
            if len(chapters) > 1:
                f.write(f"## <a id=\"{anchor}\"></a>{chapter['title']}\n\n")
            else:
                # Single chapter - skip redundant heading
                pass

            # Get segments for this chapter
            start_idx = chapter['start_segment_idx']
            end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

            # Convert to paragraphs
            chapter_segments = segments[start_idx:end_idx]
            paragraphs = segments_to_paragraphs(chapter_segments)

            for paragraph in paragraphs:
                f.write(f"{paragraph['text']}\n\n")

            if i < len(chapters) - 1:
                f.write("\n---\n\n")

        # Colophon at the end
        f.write("\n---\n\n")
        f.write("<details>\n<summary>About this transcript</summary>\n\n")
        f.write(f"- **Source:** {audio_path.name}\n")
        f.write(f"- **Duration:** {format_timestamp(duration_seconds)}\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Model:** Whisper {model_size}\n")
        f.write("</details>\n")

    print(f"Markdown book saved to: {md_path}")
    return md_path


def save_html(result: dict, audio_path: Path, duration_seconds: float, model_size: str, chapters: list):
    """Save transcript as a clean, book-like HTML document."""
    html_path = audio_path.with_suffix('.html')
    segments = result["segments"]

    # Create a clean title from filename
    title = audio_path.stem.replace('_', ' ').replace('-', ' ')

    # Build TOC HTML (clean, no timestamps)
    toc_html = ""
    for i, chapter in enumerate(chapters):
        toc_html += f'''
            <li><a href="#chapter-{i + 1}">{html_lib.escape(chapter['title'])}</a></li>'''

    # Build chapters HTML with flowing paragraphs (with hidden timing data)
    chapters_html = ""
    for i, chapter in enumerate(chapters):
        start_idx = chapter['start_segment_idx']
        end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

        # Convert segments to paragraphs
        chapter_segments = segments[start_idx:end_idx]
        paragraphs = segments_to_paragraphs(chapter_segments)

        paragraphs_html = ""
        for paragraph in paragraphs:
            escaped_text = html_lib.escape(paragraph['text'])
            start_time = paragraph['start_time']
            paragraphs_html += f'<p data-time="{start_time}">{escaped_text}</p>\n'

        chapter_title = html_lib.escape(chapter['title'])
        chapter_start = chapter['start_time']
        chapters_html += f'''
        <section class="chapter" id="chapter-{i + 1}" data-time="{chapter_start}">
            <h2>{chapter_title}</h2>
            <div class="chapter-content">
                {paragraphs_html}
            </div>
        </section>'''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_lib.escape(title)}</title>
    <style>
        :root {{
            --bg-color: #faf9f7;
            --text-color: #2c2c2c;
            --chapter-color: #1a1a1a;
            --accent-color: #6b4c35;
            --border-color: #e0ddd8;
            --sidebar-bg: #f0eeeb;
            --link-color: #6b4c35;
            --highlight-bg: #fff3cd;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #1a1a1a;
                --text-color: #e0ddd8;
                --chapter-color: #f0eeeb;
                --accent-color: #c9a87c;
                --border-color: #333;
                --sidebar-bg: #242424;
                --link-color: #c9a87c;
                --highlight-bg: #3d3520;
            }}
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 18px;
            line-height: 1.8;
            background: var(--bg-color);
            color: var(--text-color);
        }}
        .container {{
            display: grid;
            grid-template-columns: 260px 1fr;
            min-height: 100vh;
        }}
        .sidebar {{
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            padding: 30px 20px;
        }}
        .sidebar h1 {{
            font-size: 1.1em;
            font-weight: normal;
            font-style: italic;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            color: var(--chapter-color);
        }}
        .toc {{
            list-style: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 12px 0;
        }}
        .toc a {{
            color: var(--text-color);
            text-decoration: none;
            font-size: 0.9em;
            transition: color 0.2s;
        }}
        .toc a:hover {{
            color: var(--link-color);
        }}
        .main {{
            padding: 60px 80px;
            max-width: 800px;
            margin: 0 auto;
        }}
        .book-title {{
            font-size: 2.2em;
            font-weight: normal;
            text-align: center;
            margin-bottom: 60px;
            padding-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            color: var(--chapter-color);
        }}
        .chapter {{
            margin-bottom: 60px;
        }}
        .chapter h2 {{
            font-size: 1.5em;
            font-weight: normal;
            color: var(--chapter-color);
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        .chapter-content p {{
            margin-bottom: 1.5em;
            text-align: justify;
            text-indent: 1.5em;
        }}
        .chapter-content p:first-child {{
            text-indent: 0;
        }}
        .chapter-content p:first-child::first-letter {{
            font-size: 3.2em;
            float: left;
            line-height: 1;
            margin-right: 8px;
            margin-top: 4px;
            color: var(--accent-color);
        }}
        .search-container {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }}
        #search {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: inherit;
            font-size: 0.85em;
        }}
        #search:focus {{
            outline: none;
            border-color: var(--accent-color);
        }}
        .highlight {{
            background: var(--highlight-bg);
            padding: 0 2px;
        }}
        .colophon {{
            margin-top: 80px;
            padding-top: 40px;
            border-top: 1px solid var(--border-color);
            font-size: 0.85em;
            color: #888;
            text-align: center;
            font-style: italic;
        }}
        .timeline-container {{
            position: sticky;
            top: 0;
            background: var(--bg-color);
            padding: 15px 0;
            z-index: 1000;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }}
        .timeline-wrapper {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .timeline-time {{
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.85em;
            color: var(--accent-color);
            min-width: 70px;
        }}
        .timeline-slider {{
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 6px;
            background: var(--border-color);
            border-radius: 3px;
            outline: none;
            cursor: pointer;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent-color);
            border-radius: 50%;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        .timeline-slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: var(--accent-color);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .timeline-duration {{
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.85em;
            color: #888;
            min-width: 70px;
            text-align: right;
        }}
        @media (max-width: 900px) {{
            .container {{ grid-template-columns: 1fr; }}
            .sidebar {{
                position: relative;
                height: auto;
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }}
            .main {{ padding: 40px 30px; }}
        }}
        @media print {{
            .sidebar {{ display: none; }}
            .main {{ padding: 0; max-width: 100%; }}
            .chapter {{ page-break-after: always; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <h1>{html_lib.escape(title)}</h1>
            <div class="search-container">
                <input type="text" id="search" placeholder="Search...">
            </div>
            <nav>
                <ul class="toc">{toc_html}
                </ul>
            </nav>
        </aside>

        <main class="main">
            <div class="timeline-container">
                <div class="timeline-wrapper">
                    <span class="timeline-time" id="current-time">00:00:00</span>
                    <input type="range" class="timeline-slider" id="timeline" min="0" max="{int(duration_seconds)}" value="0">
                    <span class="timeline-duration">{format_timestamp(duration_seconds)}</span>
                </div>
            </div>

            <h1 class="book-title">{html_lib.escape(title)}</h1>

            {chapters_html}

            <div class="colophon">
                Transcribed from {html_lib.escape(audio_path.name)}<br>
                Duration: {format_timestamp(duration_seconds)} &middot; Generated {datetime.now().strftime('%Y-%m-%d')}
            </div>
        </main>
    </div>

    <script>
        // Timeline slider functionality
        const timeline = document.getElementById('timeline');
        const currentTimeDisplay = document.getElementById('current-time');
        const allTimedElements = document.querySelectorAll('[data-time]');

        function formatTime(seconds) {{
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${{String(h).padStart(2, '0')}}:${{String(m).padStart(2, '0')}}:${{String(s).padStart(2, '0')}}`;
        }}

        function findElementAtTime(targetTime) {{
            let closest = null;
            let closestTime = -1;
            allTimedElements.forEach(el => {{
                const time = parseFloat(el.dataset.time);
                if (time <= targetTime && time > closestTime) {{
                    closestTime = time;
                    closest = el;
                }}
            }});
            return closest;
        }}

        timeline.addEventListener('input', function() {{
            const time = parseInt(this.value);
            currentTimeDisplay.textContent = formatTime(time);

            const targetElement = findElementAtTime(time);
            if (targetElement) {{
                const offset = document.querySelector('.timeline-container').offsetHeight + 20;
                const elementTop = targetElement.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({{ top: elementTop, behavior: 'smooth' }});
            }}
        }});

        // Update slider position based on scroll
        let isScrolling = false;
        window.addEventListener('scroll', function() {{
            if (isScrolling) return;

            const scrollPos = window.pageYOffset + window.innerHeight / 3;
            let currentElement = null;
            let currentTime = 0;

            allTimedElements.forEach(el => {{
                if (el.offsetTop <= scrollPos) {{
                    currentElement = el;
                    currentTime = parseFloat(el.dataset.time);
                }}
            }});

            if (currentTime > 0) {{
                timeline.value = currentTime;
                currentTimeDisplay.textContent = formatTime(currentTime);
            }}
        }});

        // Prevent scroll listener from fighting with slider
        timeline.addEventListener('mousedown', () => isScrolling = true);
        timeline.addEventListener('mouseup', () => setTimeout(() => isScrolling = false, 500));
        timeline.addEventListener('touchstart', () => isScrolling = true);
        timeline.addEventListener('touchend', () => setTimeout(() => isScrolling = false, 500));

        // Search functionality
        document.getElementById('search').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            const paragraphs = document.querySelectorAll('.chapter-content p');
            const chapters = document.querySelectorAll('.chapter');

            if (!query) {{
                paragraphs.forEach(p => {{
                    p.style.display = 'block';
                    p.innerHTML = p.textContent;
                }});
                chapters.forEach(ch => ch.style.display = 'block');
                return;
            }}

            chapters.forEach(chapter => {{
                let hasMatch = false;
                chapter.querySelectorAll('.chapter-content p').forEach(p => {{
                    const original = p.textContent;
                    if (original.toLowerCase().includes(query)) {{
                        p.style.display = 'block';
                        hasMatch = true;
                        const regex = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
                        p.innerHTML = original.replace(regex, '<span class="highlight">$1</span>');
                    }} else {{
                        p.style.display = 'none';
                    }}
                }});
                chapter.style.display = hasMatch ? 'block' : 'none';
            }});
        }});

        // Smooth scroll for TOC
        document.querySelectorAll('.toc a').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{ behavior: 'smooth' }});
            }});
        }});
    </script>
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML book saved to: {html_path}")
    return html_path


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def display_stats(result: dict, audio_path: Path, duration_seconds: float, stats: dict, output_files: list):
    """Display comprehensive transcription statistics."""

    segments = result.get("segments", [])
    full_text = result.get("text", "")

    # Calculate text statistics
    word_count = len(full_text.split())
    char_count = len(full_text)
    segment_count = len(segments)

    # Calculate average segment duration
    if segment_count > 0:
        avg_segment_duration = duration_seconds / segment_count
    else:
        avg_segment_duration = 0

    # Calculate words per minute
    if duration_seconds > 0:
        words_per_minute = (word_count / duration_seconds) * 60
    else:
        words_per_minute = 0

    print("\n")
    print("=" * 70)
    print("                    TRANSCRIPTION STATISTICS")
    print("=" * 70)

    # Timing Section
    print("\n┌─ TIMING ─────────────────────────────────────────────────────────┐")
    print(f"│  Audio Duration:        {format_timestamp(duration_seconds):>12}                        │")
    print(f"│  Transcription Time:    {format_timestamp(stats['transcription_time']):>12}                        │")
    if stats.get('realtime_factor'):
        rtf = stats['realtime_factor']
        if rtf >= 1:
            print(f"│  Speed:                 {rtf:>10.1f}x realtime                      │")
        else:
            print(f"│  Speed:                 {rtf:>10.2f}x realtime (slower than audio)  │")
    if stats.get('model_load_time'):
        print(f"│  Model Load Time:       {stats['model_load_time']:>10.1f}s                          │")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Resource Utilization Section
    print("\n┌─ RESOURCE UTILIZATION ───────────────────────────────────────────┐")
    print(f"│  Device:                {stats['device']:>12}                        │")
    if stats.get('gpu_name'):
        gpu_name_short = stats['gpu_name'][:40] if len(stats['gpu_name']) > 40 else stats['gpu_name']
        print(f"│  GPU:                   {gpu_name_short:<40} │")
    if stats.get('gpu_memory_fraction'):
        print(f"│  GPU Memory Limit:      {stats['gpu_memory_fraction']*100:>10.0f}%                        │")
    if stats.get('gpu_memory_used') is not None and stats.get('gpu_memory_total'):
        gpu_mem_pct = (stats['gpu_memory_used'] / stats['gpu_memory_total']) * 100
        print(f"│  GPU Memory (Peak):     {stats['gpu_memory_used']:>6.2f} GB / {stats['gpu_memory_total']:.2f} GB ({gpu_mem_pct:.1f}%)       │")
    if stats.get('cpu_percent') is not None:
        print(f"│  CPU Usage (Avg):       {stats['cpu_percent']:>10.1f}%                        │")
    if stats.get('memory_used_gb') is not None and stats.get('memory_total_gb'):
        mem_pct = (stats['memory_used_gb'] / stats['memory_total_gb']) * 100
        print(f"│  RAM Usage (Peak):      {stats['memory_used_gb']:>6.2f} GB / {stats['memory_total_gb']:.1f} GB ({mem_pct:.1f}%)        │")
    print(f"│  Model Size:            {stats['model_size']:>12}                        │")
    if stats.get('backend'):
        print(f"│  Backend:               {stats['backend']:>12}                        │")
    print(f"│  Beam Size:             {stats.get('beam_size', 5):>12}                        │")
    if stats.get('batch_size'):
        print(f"│  Batch Size:            {stats.get('batch_size'):>12}                        │")
    elif stats.get('best_of'):
        print(f"│  Best Of:               {stats.get('best_of', 5):>12}                        │")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Output Summary Section
    print("\n┌─ OUTPUT SUMMARY ─────────────────────────────────────────────────┐")
    print(f"│  Total Words:           {word_count:>12,}                        │")
    print(f"│  Total Characters:      {char_count:>12,}                        │")
    print(f"│  Segments:              {segment_count:>12,}                        │")
    print(f"│  Avg Segment Duration:  {avg_segment_duration:>10.1f}s                          │")
    print(f"│  Words Per Minute:      {words_per_minute:>10.1f}                          │")
    print(f"│  Detected Language:     {result.get('language', 'en'):>12}                        │")
    print("└──────────────────────────────────────────────────────────────────┘")

    # Files Generated Section
    print("\n┌─ FILES GENERATED ────────────────────────────────────────────────┐")
    total_size = 0
    for file_path in output_files:
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            size_str = format_file_size(size)
            name = file_path.name
            if len(name) > 45:
                name = "..." + name[-42:]
            print(f"│  {name:<45} {size_str:>10} │")
    print(f"│  {'─' * 56}─────────│")
    print(f"│  {'Total':45} {format_file_size(total_size):>10} │")
    print("└──────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 70)
    print("                    TRANSCRIPTION COMPLETE")
    print("=" * 70 + "\n")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


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
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip displaying statistics after transcription"
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=None,
        metavar="FRACTION",
        help="GPU memory fraction to use (0.0-1.0). E.g., 0.8 for 80%%. Default: no limit"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search. Higher = more GPU usage & accuracy (default: 5)"
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates when sampling. Higher = more GPU usage (default: 5)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster-whisper backend (4x faster, same model, same accuracy)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for faster-whisper (default: 16, higher = more GPU)"
    )
    parser.add_argument(
        "--high-performance",
        action="store_true",
        help="Enable --fast with optimized settings (beam=10, batch=24)"
    )

    args = parser.parse_args()

    # Apply high-performance preset
    if args.high_performance:
        args.fast = True
        if args.beam_size == 5:
            args.beam_size = 10
        if args.batch_size == 16:
            args.batch_size = 24

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
    if args.fast:
        result, duration_seconds, stats = transcribe_audio_fast(
            audio_path,
            model_size=args.model,
            beam_size=args.beam_size,
            batch_size=args.batch_size
        )
    else:
        result, duration_seconds, stats = transcribe_audio(
            audio_path,
            model_size=args.model,
            gpu_memory_fraction=args.gpu_utilization,
            beam_size=args.beam_size,
            best_of=args.best_of
        )

    # Save transcripts
    output_files = save_transcript(result, audio_path, duration_seconds, args.model)

    # Display statistics
    if not args.no_stats:
        display_stats(result, audio_path, duration_seconds, stats, output_files)
    else:
        print("\n" + "=" * 60)
        print("Transcription complete!")


if __name__ == "__main__":
    main()
