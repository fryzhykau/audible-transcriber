"""
Audiobook Transcription Script (AAX to Text)

For personal use of purchased audiobooks.

Requirements:
    pip install openai-whisper audible tqdm psutil

    For faster-whisper backend:
        pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12

You'll also need FFmpeg installed:
    - Windows: winget install ffmpeg  OR  choco install ffmpeg
    - Or download from: https://ffmpeg.org/download.html

Usage:
    # OpenAI Whisper (original) - processes entire file at once
    python transcribe_audiobook.py <audiobook.m4a>

    # faster-whisper - uses chunked processing for large files
    python transcribe_audiobook.py <audiobook.m4a> --fast

    # faster-whisper without chunking (process entire file at once)
    python transcribe_audiobook.py <audiobook.m4a> --fast --no-chunk

Backends:
    OpenAI Whisper (default): Original implementation, processes entire file at once.
                              May be faster for very long files due to no chunk overhead.

    faster-whisper (--fast):  CTranslate2-based, typically 4x faster for short files.
                              Uses chunked processing for files >1 hour by default.
                              Use --no-chunk to disable chunking.

Options:
    --model, -m           Model size: tiny, base, small, medium, large (default: medium)
    --skip-convert        Skip AAX conversion if you already have M4A/MP3
    --activation-bytes    Provide Audible activation bytes directly
    --no-stats            Skip displaying statistics after transcription

GPU Performance Options:
    --fast                Use faster-whisper backend
    --no-chunk            Disable chunked processing (faster-whisper only)
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


def print_gpu_info():
    """Print detailed GPU information to help diagnose which GPU is being used."""
    import torch

    print("\n" + "=" * 60)
    print("GPU DETECTION")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is NOT available!")
        print("  - PyTorch cannot see any NVIDIA GPU")
        print("  - Will fall back to CPU (very slow)")
        print("  - If on laptop: plug in power adapter")
        print("=" * 60 + "\n")
        return False, None

    device_count = torch.cuda.device_count()
    print(f"CUDA available: Yes")
    print(f"CUDA devices found: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"\n  GPU {i}: {name}")
        print(f"    Memory: {mem_gb:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")

        # Warn if it looks like an integrated GPU
        if "Intel" in name or "Iris" in name:
            print(f"    WARNING: This appears to be an integrated GPU!")

    # Check which one will be used (device 0)
    primary_gpu = torch.cuda.get_device_name(0)
    print(f"\n>>> Will use: {primary_gpu}")

    if "NVIDIA" in primary_gpu and ("RTX" in primary_gpu or "GTX" in primary_gpu):
        print(">>> Status: OK - Using dedicated NVIDIA GPU")
    else:
        print(">>> WARNING: May not be using dedicated GPU!")

    print("=" * 60 + "\n")
    return True, primary_gpu


def get_gpu_memory_nvidia_smi() -> tuple:
    """Get GPU memory usage via nvidia-smi (more accurate for CTranslate2/faster-whisper).

    Returns (used_gb, total_gb) or (None, None) if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]  # First GPU
            used_mb, total_mb = map(float, line.split(','))
            return used_mb / 1024, total_mb / 1024
    except Exception:
        pass
    return None, None


def transcribe_audio_fast(audio_path: Path, model_size: str = "medium",
                          beam_size: int = 5, batch_size: int = 16,
                          disable_chunking: bool = False) -> tuple:
    """Transcribe audio using faster-whisper (CTranslate2). Returns (result, duration_seconds, stats)."""

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Installing faster-whisper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"], check=True)
        from faster_whisper import WhisperModel

    import torch

    # Print GPU detection info
    cuda_available, gpu_name = print_gpu_info()

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
    if cuda_available and gpu_name:
        device = "cuda"
        compute_type = "float16"  # Use FP16 for speed
        stats['device'] = 'GPU (faster-whisper)'
        stats['gpu_name'] = gpu_name
        stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        device = "cpu"
        compute_type = "int8"  # Use INT8 on CPU for speed
        stats['device'] = 'CPU (faster-whisper)'
        print("Falling back to CPU with INT8 quantization (this will be slow)")

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

    # Use chunk-based processing for large files (>1 hour) for better GPU utilization
    # Can be disabled with --no-chunk flag
    CHUNK_THRESHOLD = 3600  # 1 hour in seconds
    CHUNK_DURATION = 1800   # 30 minutes per chunk

    if duration_seconds > CHUNK_THRESHOLD and not disable_chunking:
        return _transcribe_chunked(
            audio_path, model, duration_seconds, device, beam_size,
            batch_size, stats, torch, CHUNK_DURATION
        )

    # For shorter files or when chunking is disabled, use direct transcription
    if disable_chunking and duration_seconds > CHUNK_THRESHOLD:
        print(f"Chunking disabled - processing entire file at once...")
    return _transcribe_direct(
        audio_path, model, duration_seconds, device, beam_size,
        batch_size, stats, torch
    )


def _transcribe_direct(audio_path: Path, model, duration_seconds: float,
                       device: str, beam_size: int, batch_size: int,
                       stats: dict, torch) -> tuple:
    """Direct transcription for smaller files (under 1 hour)."""
    import subprocess as sp

    # Convert to WAV for faster processing (M4A decoding is very slow in faster-whisper)
    original_audio_path = audio_path
    if audio_path.suffix.lower() in ['.m4a', '.mp3', '.aac', '.ogg', '.flac']:
        wav_path = audio_path.with_suffix('.temp.wav')
        print(f"Converting to WAV for faster processing...")
        convert_start = time.time()
        convert_cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path), "-loglevel", "error"
        ]
        sp.run(convert_cmd, check=True)
        print(f"Conversion completed in {time.time() - convert_start:.1f}s")
        audio_path = wav_path
        stats['wav_conversion_time'] = time.time() - convert_start

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
                vad_filter=False,
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

            # Track GPU memory using nvidia-smi (more accurate for CTranslate2)
            if device == "cuda":
                gpu_used, _ = get_gpu_memory_nvidia_smi()
                if gpu_used is not None:
                    peak_gpu_memory[0] = max(peak_gpu_memory[0], gpu_used)
                else:
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

    if duration_seconds > 0:
        stats['realtime_factor'] = duration_seconds / stats['transcription_time']

    if error[0]:
        raise error[0]

    # Cleanup temporary WAV file if we created one
    if audio_path != original_audio_path and audio_path.exists():
        try:
            audio_path.unlink()
            print(f"Cleaned up temporary file: {audio_path.name}")
        except Exception:
            pass

    # Format result to match openai-whisper structure
    result = {
        "text": "".join(full_text),
        "segments": segments_result,
        "language": detected_language[0] or "en",
    }

    stats['processing_mode'] = 'direct'
    return result, duration_seconds, stats


def _transcribe_chunked(audio_path: Path, model, duration_seconds: float,
                        device: str, beam_size: int, batch_size: int,
                        stats: dict, torch, chunk_duration: int = 1800) -> tuple:
    """Chunk-based transcription for large files. Better GPU utilization."""
    import subprocess as sp
    import tempfile

    print(f"\nUsing chunk-based processing for better GPU utilization...")
    print(f"Chunk size: {chunk_duration // 60} minutes")

    num_chunks = int(duration_seconds // chunk_duration) + 1
    print(f"Total chunks: {num_chunks}\n")

    # Track resources
    cpu_samples = []
    memory_samples = []
    peak_gpu_memory = [0]

    all_segments = []
    all_text = []
    detected_language = "en"
    chunk_times = []  # Track time per chunk for ETA calculation

    transcribe_start = time.time()

    # Reset GPU memory stats for accurate tracking
    if device == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # Create temp directory for chunk files
    temp_dir = Path(tempfile.gettempdir()) / f"whisper_chunks_{audio_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    try:
        for chunk_idx in range(num_chunks):
            chunk_start_time = chunk_idx * chunk_duration
            chunk_process_start = time.time()

            # Calculate actual chunk duration (last chunk may be shorter)
            actual_chunk_duration = min(chunk_duration, duration_seconds - chunk_start_time)
            if actual_chunk_duration <= 0:
                break

            # Calculate progress percentage
            progress_pct = (chunk_idx / num_chunks) * 100
            audio_position = format_timestamp(chunk_start_time)
            audio_end = format_timestamp(min(chunk_start_time + chunk_duration, duration_seconds))

            # Calculate ETA based on average chunk time
            if chunk_times:
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = num_chunks - chunk_idx
                eta_seconds = avg_chunk_time * remaining_chunks
                eta_str = format_timestamp(eta_seconds)
            else:
                eta_str = "calculating..."

            # Print detailed progress line
            print(f"\r[{progress_pct:5.1f}%] Chunk {chunk_idx + 1}/{num_chunks} | "
                  f"{audio_position} -> {audio_end} | "
                  f"Segments: {len(all_segments):,} | "
                  f"ETA: {eta_str}    ", end="", flush=True)

            # Extract chunk to WAV
            chunk_wav = temp_dir / f"chunk_{chunk_idx:04d}.wav"

            extract_cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(chunk_start_time),
                "-t", str(actual_chunk_duration),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(chunk_wav), "-loglevel", "error"
            ]
            sp.run(extract_cmd, check=True)

            # Transcribe chunk
            chunk_segments = []
            try:
                segments, info = model.transcribe(
                    str(chunk_wav),
                    language="en",
                    beam_size=beam_size,
                    vad_filter=False,
                )

                if chunk_idx == 0:
                    detected_language = info.language

                # Collect segments with adjusted timestamps
                for segment in segments:
                    seg_data = {
                        "start": segment.start + chunk_start_time,
                        "end": segment.end + chunk_start_time,
                        "text": segment.text,
                    }
                    all_segments.append(seg_data)
                    all_text.append(segment.text)
                    chunk_segments.append(seg_data)

            except Exception as e:
                print(f"\nError transcribing chunk {chunk_idx}: {e}")
                continue
            finally:
                # Cleanup chunk file immediately to save disk space
                try:
                    chunk_wav.unlink()
                except Exception:
                    pass

            # Sample resources after each chunk
            cpu_samples.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            memory_samples.append(mem.used / (1024**3))

            # Track GPU memory using nvidia-smi (more accurate for CTranslate2)
            if device == "cuda":
                gpu_used, _ = get_gpu_memory_nvidia_smi()
                if gpu_used is not None:
                    peak_gpu_memory[0] = max(peak_gpu_memory[0], gpu_used)
                else:
                    # Fallback to PyTorch tracking
                    try:
                        current_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                        max_gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                        peak_gpu_memory[0] = max(peak_gpu_memory[0], current_gpu_mem, max_gpu_mem)
                    except Exception:
                        pass

            # Track chunk processing time
            chunk_process_time = time.time() - chunk_process_start
            chunk_times.append(chunk_process_time)

            # Calculate and show speed for this chunk
            chunk_speed = actual_chunk_duration / chunk_process_time if chunk_process_time > 0 else 0

            # Update progress with speed info
            print(f"\r[{progress_pct:5.1f}%] Chunk {chunk_idx + 1}/{num_chunks} | "
                  f"{audio_position} -> {audio_end} | "
                  f"Segments: {len(all_segments):,} | "
                  f"Speed: {chunk_speed:.1f}x | "
                  f"ETA: {eta_str}    ", end="", flush=True)

        # Final progress update
        print(f"\r[100.0%] Chunk {num_chunks}/{num_chunks} | "
              f"Complete | "
              f"Segments: {len(all_segments):,} | "
              f"Done!                              ")

    finally:
        # Cleanup temp directory
        try:
            temp_dir.rmdir()
        except Exception:
            pass

    transcribe_end = time.time()
    stats['transcription_time'] = transcribe_end - transcribe_start
    stats['chunks_processed'] = num_chunks
    stats['chunk_duration_min'] = chunk_duration // 60

    if cpu_samples:
        stats['cpu_percent'] = sum(cpu_samples) / len(cpu_samples)
    if memory_samples:
        stats['memory_used_gb'] = max(memory_samples)
    stats['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)

    if device == "cuda":
        stats['gpu_memory_used'] = peak_gpu_memory[0]
        # Also try to get the actual peak from PyTorch
        try:
            pytorch_peak = torch.cuda.max_memory_allocated() / (1024**3)
            stats['gpu_memory_used'] = max(stats['gpu_memory_used'], pytorch_peak)
        except Exception:
            pass

    if duration_seconds > 0:
        stats['realtime_factor'] = duration_seconds / stats['transcription_time']

    # Calculate average speed across chunks
    if chunk_times:
        stats['avg_chunk_speed'] = (chunk_duration * len(chunk_times)) / sum(chunk_times)

    # Format result
    result = {
        "text": "".join(all_text),
        "segments": all_segments,
        "language": detected_language or "en",
    }

    print(f"\nProcessed {len(all_segments):,} segments from {num_chunks} chunks")

    stats['processing_mode'] = 'chunked'
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

    # Print GPU detection info
    cuda_available, gpu_name = print_gpu_info()

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
    if cuda_available and gpu_name:
        device = "cuda"
        stats['device'] = 'GPU'
        stats['gpu_name'] = gpu_name
        stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Set GPU memory fraction limit if specified
        if gpu_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
            print(f"GPU memory capped at {gpu_memory_fraction*100:.0f}%")

        # Enable TF32 for better performance on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmark for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True

    else:
        device = "cpu"
        stats['device'] = 'CPU'
        print("Falling back to CPU (this will be very slow)")

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

            # Track GPU memory - prefer nvidia-smi, fallback to PyTorch
            if device == "cuda":
                gpu_used, _ = get_gpu_memory_nvidia_smi()
                if gpu_used is not None:
                    peak_gpu_memory[0] = max(peak_gpu_memory[0], gpu_used)
                else:
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
            pytorch_peak = torch.cuda.max_memory_allocated() / (1024**3)
            stats['gpu_memory_used'] = max(stats['gpu_memory_used'], pytorch_peak)
        except Exception:
            pass

    # Calculate realtime factor (how much faster/slower than realtime)
    if duration_seconds > 0:
        stats['realtime_factor'] = duration_seconds / stats['transcription_time']

    # Add backend and processing mode info
    stats['backend'] = 'openai-whisper'
    stats['processing_mode'] = 'direct'

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


def save_transcript(result: dict, audio_path: Path, duration_seconds: float, model_size: str, stats: dict = None, bg_color: str = "#ffffff") -> list:
    """Save transcript to multiple formats. Returns list of output file paths."""
    import json

    output_files = []
    segments = result["segments"]
    chapters = detect_chapters(segments)

    # Save statistics to JSON
    if stats:
        stats_path = audio_path.parent / f"{audio_path.stem}_stats.json"
        stats_data = {
            'source_file': audio_path.name,
            'audio_duration': duration_seconds,
            'audio_duration_formatted': format_timestamp(duration_seconds),
            'transcription_time': stats.get('transcription_time'),
            'transcription_time_formatted': format_timestamp(stats.get('transcription_time', 0)),
            'speed_realtime': stats.get('realtime_factor'),
            'model_size': stats.get('model_size'),
            'backend': stats.get('backend'),
            'processing_mode': stats.get('processing_mode'),
            'chunks_processed': stats.get('chunks_processed'),
            'chunk_duration_min': stats.get('chunk_duration_min'),
            'device': stats.get('device'),
            'gpu_name': stats.get('gpu_name'),
            'gpu_memory_used_gb': stats.get('gpu_memory_used'),
            'gpu_memory_total_gb': stats.get('gpu_memory_total'),
            'cpu_percent_avg': stats.get('cpu_percent'),
            'ram_used_gb': stats.get('memory_used_gb'),
            'ram_total_gb': stats.get('memory_total_gb'),
            'beam_size': stats.get('beam_size'),
            'batch_size': stats.get('batch_size'),
            'best_of': stats.get('best_of'),
            'model_load_time': stats.get('model_load_time'),
            'wav_conversion_time': stats.get('wav_conversion_time'),
            'word_count': len(result.get('text', '').split()),
            'segment_count': len(segments),
            'detected_language': result.get('language', 'en'),
            'generated_at': datetime.now().isoformat(),
        }
        # Remove None values
        stats_data = {k: v for k, v in stats_data.items() if v is not None}

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")
        output_files.append(stats_path)

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
    html_path = save_html(result, audio_path, duration_seconds, model_size, chapters, bg_color)
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


def save_html(result: dict, audio_path: Path, duration_seconds: float, model_size: str, chapters: list, bg_color: str = "#ffffff"):
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

    # Derive complementary colors from background
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def darken(hex_color, factor=0.9):
        r, g, b = hex_to_rgb(hex_color)
        return rgb_to_hex((int(r * factor), int(g * factor), int(b * factor)))

    def is_light(hex_color):
        r, g, b = hex_to_rgb(hex_color)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance > 0.5

    # Set colors based on background
    if is_light(bg_color):
        text_color = "#2c2c2c"
        chapter_color = "#1a1a1a"
        accent_color = "#6b4c35"
        border_color = darken(bg_color, 0.9)
        sidebar_bg = darken(bg_color, 0.96)
        link_color = "#6b4c35"
        highlight_bg = "#fff3cd"
    else:
        text_color = "#e0ddd8"
        chapter_color = "#f0eeeb"
        accent_color = "#c9a87c"
        border_color = "#333"
        sidebar_bg = darken(bg_color, 0.8)
        link_color = "#c9a87c"
        highlight_bg = "#3d3520"

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_lib.escape(title)}</title>
    <style>
        :root {{
            --bg-color: {bg_color};
            --text-color: {text_color};
            --chapter-color: {chapter_color};
            --accent-color: {accent_color};
            --border-color: {border_color};
            --sidebar-bg: {sidebar_bg};
            --link-color: {link_color};
            --highlight-bg: {highlight_bg};
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
        .settings {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid var(--border-color);
        }}
        .settings-label {{
            font-size: 0.8em;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 8px;
            display: block;
        }}
        .color-picker-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .color-picker {{
            width: 32px;
            height: 32px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            padding: 0;
        }}
        .color-presets {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }}
        .color-preset {{
            width: 24px;
            height: 24px;
            border: 1px solid var(--border-color);
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        .color-preset:hover {{
            transform: scale(1.1);
        }}
        .color-preset.active {{
            border: 2px solid var(--accent-color);
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
        .play-btn {{
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 50%;
            background: var(--accent-color);
            color: var(--bg-color);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.1s, opacity 0.2s;
            flex-shrink: 0;
        }}
        .play-btn:hover {{
            transform: scale(1.1);
        }}
        .play-btn.playing {{
            opacity: 0.9;
        }}
        .play-btn svg {{
            width: 18px;
            height: 18px;
        }}
        .play-icon {{
            margin-left: 2px;
        }}
        .speed-select {{
            padding: 4px 8px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.75em;
            cursor: pointer;
            flex-shrink: 0;
        }}
        .speed-select:focus {{
            outline: none;
            border-color: var(--accent-color);
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
            <div class="settings">
                <span class="settings-label">Background</span>
                <div class="color-picker-row">
                    <input type="color" id="bg-color-picker" class="color-picker" value="{bg_color}">
                </div>
                <div class="color-presets">
                    <div class="color-preset" data-color="#ffffff" style="background:#ffffff" title="White"></div>
                    <div class="color-preset" data-color="#f5f5dc" style="background:#f5f5dc" title="Beige"></div>
                    <div class="color-preset" data-color="#faf9f7" style="background:#faf9f7" title="Warm White"></div>
                    <div class="color-preset" data-color="#f0ead6" style="background:#f0ead6" title="Eggshell"></div>
                    <div class="color-preset" data-color="#e8e4d9" style="background:#e8e4d9" title="Parchment"></div>
                    <div class="color-preset" data-color="#1a1a1a" style="background:#1a1a1a" title="Dark"></div>
                    <div class="color-preset" data-color="#2d2d2d" style="background:#2d2d2d" title="Charcoal"></div>
                </div>
            </div>
        </aside>

        <main class="main">
            <div class="timeline-container">
                <div class="timeline-wrapper">
                    <button class="play-btn" id="play-btn" title="Play/Pause auto-scroll">
                        <svg class="play-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                        <svg class="pause-icon" viewBox="0 0 24 24" fill="currentColor" style="display:none"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                    </button>
                    <span class="timeline-time" id="current-time">00:00:00</span>
                    <input type="range" class="timeline-slider" id="timeline" min="0" max="{int(duration_seconds)}" value="0">
                    <span class="timeline-duration">{format_timestamp(duration_seconds)}</span>
                    <select class="speed-select" id="speed-select" title="Playback speed">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                        <option value="3">3x</option>
                    </select>
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

        // Play/Pause auto-scroll functionality
        const playBtn = document.getElementById('play-btn');
        const playIcon = playBtn.querySelector('.play-icon');
        const pauseIcon = playBtn.querySelector('.pause-icon');
        const speedSelect = document.getElementById('speed-select');
        const totalDuration = {int(duration_seconds)};

        let isPlaying = false;
        let playbackInterval = null;
        let currentPlayTime = 0;
        let playbackSpeed = 1;

        function updatePlaybackUI() {{
            if (isPlaying) {{
                playIcon.style.display = 'none';
                pauseIcon.style.display = 'block';
                playBtn.classList.add('playing');
            }} else {{
                playIcon.style.display = 'block';
                pauseIcon.style.display = 'none';
                playBtn.classList.remove('playing');
            }}
        }}

        function scrollToTime(time) {{
            const targetElement = findElementAtTime(time);
            if (targetElement) {{
                const offset = document.querySelector('.timeline-container').offsetHeight + 20;
                const elementTop = targetElement.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({{ top: elementTop, behavior: 'smooth' }});
            }}
        }}

        function startPlayback() {{
            if (playbackInterval) clearInterval(playbackInterval);

            // Get current position from slider
            currentPlayTime = parseInt(timeline.value);

            isScrolling = true; // Prevent scroll listener from interfering
            isPlaying = true;
            updatePlaybackUI();

            // Update every 100ms for smooth progress
            const updateInterval = 100;
            playbackInterval = setInterval(() => {{
                currentPlayTime += (updateInterval / 1000) * playbackSpeed;

                if (currentPlayTime >= totalDuration) {{
                    currentPlayTime = totalDuration;
                    stopPlayback();
                    return;
                }}

                timeline.value = currentPlayTime;
                currentTimeDisplay.textContent = formatTime(Math.floor(currentPlayTime));

                // Scroll to current position every second
                if (Math.floor(currentPlayTime * 10) % 10 === 0) {{
                    scrollToTime(currentPlayTime);
                }}
            }}, updateInterval);
        }}

        function stopPlayback() {{
            if (playbackInterval) {{
                clearInterval(playbackInterval);
                playbackInterval = null;
            }}
            isPlaying = false;
            isScrolling = false;
            updatePlaybackUI();
        }}

        function togglePlayback() {{
            if (isPlaying) {{
                stopPlayback();
            }} else {{
                startPlayback();
            }}
        }}

        playBtn.addEventListener('click', togglePlayback);

        // Keyboard shortcut: Space to play/pause
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space' && e.target.tagName !== 'INPUT') {{
                e.preventDefault();
                togglePlayback();
            }}
        }});

        // Speed selector
        speedSelect.addEventListener('change', (e) => {{
            playbackSpeed = parseFloat(e.target.value);
        }});

        // Stop playback when user interacts with timeline
        timeline.addEventListener('mousedown', stopPlayback);
        timeline.addEventListener('touchstart', stopPlayback);

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

        // Background color picker functionality
        const colorPicker = document.getElementById('bg-color-picker');
        const colorPresets = document.querySelectorAll('.color-preset');
        const storageKey = 'book-bg-color';

        function hexToRgb(hex) {{
            const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
            return result ? {{
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            }} : null;
        }}

        function rgbToHex(r, g, b) {{
            return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
        }}

        function darken(hex, factor) {{
            const rgb = hexToRgb(hex);
            return rgbToHex(
                Math.round(rgb.r * factor),
                Math.round(rgb.g * factor),
                Math.round(rgb.b * factor)
            );
        }}

        function isLight(hex) {{
            const rgb = hexToRgb(hex);
            const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
            return luminance > 0.5;
        }}

        function applyBackgroundColor(color) {{
            const root = document.documentElement;
            root.style.setProperty('--bg-color', color);

            if (isLight(color)) {{
                root.style.setProperty('--text-color', '#2c2c2c');
                root.style.setProperty('--chapter-color', '#1a1a1a');
                root.style.setProperty('--accent-color', '#6b4c35');
                root.style.setProperty('--border-color', darken(color, 0.9));
                root.style.setProperty('--sidebar-bg', darken(color, 0.96));
                root.style.setProperty('--link-color', '#6b4c35');
                root.style.setProperty('--highlight-bg', '#fff3cd');
            }} else {{
                root.style.setProperty('--text-color', '#e0ddd8');
                root.style.setProperty('--chapter-color', '#f0eeeb');
                root.style.setProperty('--accent-color', '#c9a87c');
                root.style.setProperty('--border-color', '#444');
                root.style.setProperty('--sidebar-bg', darken(color, 0.8));
                root.style.setProperty('--link-color', '#c9a87c');
                root.style.setProperty('--highlight-bg', '#3d3520');
            }}

            // Update color picker value
            colorPicker.value = color;

            // Update preset highlights
            colorPresets.forEach(preset => {{
                preset.classList.toggle('active', preset.dataset.color === color);
            }});

            // Save to localStorage
            localStorage.setItem(storageKey, color);
        }}

        // Load saved color on page load
        const savedColor = localStorage.getItem(storageKey);
        if (savedColor) {{
            applyBackgroundColor(savedColor);
        }}

        // Color picker change handler
        colorPicker.addEventListener('input', (e) => {{
            applyBackgroundColor(e.target.value);
        }});

        // Preset click handlers
        colorPresets.forEach(preset => {{
            preset.addEventListener('click', () => {{
                applyBackgroundColor(preset.dataset.color);
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
    print("\n+-- TIMING " + "-" * 58 + "+")
    print(f"|  Audio Duration:        {format_timestamp(duration_seconds):>12}                        |")
    print(f"|  Transcription Time:    {format_timestamp(stats['transcription_time']):>12}                        |")
    if stats.get('realtime_factor'):
        rtf = stats['realtime_factor']
        if rtf >= 1:
            print(f"|  Speed:                 {rtf:>10.1f}x realtime                      |")
        else:
            print(f"|  Speed:                 {rtf:>10.2f}x realtime (slower than audio)  |")
    if stats.get('model_load_time'):
        print(f"|  Model Load Time:       {stats['model_load_time']:>10.1f}s                          |")
    if stats.get('wav_conversion_time'):
        print(f"|  WAV Conversion Time:   {stats['wav_conversion_time']:>10.1f}s                          |")
    if stats.get('chunks_processed'):
        chunk_info = f"{stats['chunks_processed']} x {stats.get('chunk_duration_min', 30)}min"
        print(f"|  Chunks Processed:      {chunk_info:>12}                        |")
    print("+" + "-" * 68 + "+")

    # Resource Utilization Section
    print("\n+-- RESOURCE UTILIZATION " + "-" * 43 + "+")
    print(f"|  Device:                {stats['device']:>12}                        |")
    if stats.get('gpu_name'):
        gpu_name_short = stats['gpu_name'][:40] if len(stats['gpu_name']) > 40 else stats['gpu_name']
        print(f"|  GPU:                   {gpu_name_short:<40} |")
    if stats.get('gpu_memory_fraction'):
        print(f"|  GPU Memory Limit:      {stats['gpu_memory_fraction']*100:>10.0f}%                        |")
    if stats.get('gpu_memory_used') is not None and stats.get('gpu_memory_total'):
        gpu_mem_pct = (stats['gpu_memory_used'] / stats['gpu_memory_total']) * 100
        print(f"|  GPU Memory (Peak):     {stats['gpu_memory_used']:>6.2f} GB / {stats['gpu_memory_total']:.2f} GB ({gpu_mem_pct:.1f}%)       |")
    if stats.get('cpu_percent') is not None:
        print(f"|  CPU Usage (Avg):       {stats['cpu_percent']:>10.1f}%                        |")
    if stats.get('memory_used_gb') is not None and stats.get('memory_total_gb'):
        mem_pct = (stats['memory_used_gb'] / stats['memory_total_gb']) * 100
        print(f"|  RAM Usage (Peak):      {stats['memory_used_gb']:>6.2f} GB / {stats['memory_total_gb']:.1f} GB ({mem_pct:.1f}%)        |")
    print(f"|  Model Size:            {stats['model_size']:>12}                        |")
    if stats.get('backend'):
        print(f"|  Backend:               {stats['backend']:>12}                        |")
    if stats.get('processing_mode'):
        print(f"|  Processing Mode:       {stats['processing_mode']:>12}                        |")
    print(f"|  Beam Size:             {stats.get('beam_size', 5):>12}                        |")
    if stats.get('batch_size'):
        print(f"|  Batch Size:            {stats.get('batch_size'):>12}                        |")
    elif stats.get('best_of'):
        print(f"|  Best Of:               {stats.get('best_of', 5):>12}                        |")
    print("+" + "-" * 68 + "+")

    # Output Summary Section
    print("\n+-- OUTPUT SUMMARY " + "-" * 50 + "+")
    print(f"|  Total Words:           {word_count:>12,}                        |")
    print(f"|  Total Characters:      {char_count:>12,}                        |")
    print(f"|  Segments:              {segment_count:>12,}                        |")
    print(f"|  Avg Segment Duration:  {avg_segment_duration:>10.1f}s                          |")
    print(f"|  Words Per Minute:      {words_per_minute:>10.1f}                          |")
    print(f"|  Detected Language:     {result.get('language', 'en'):>12}                        |")
    print("+" + "-" * 68 + "+")

    # Files Generated Section
    print("\n+-- FILES GENERATED " + "-" * 48 + "+")
    total_size = 0
    for file_path in output_files:
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            size_str = format_file_size(size)
            name = file_path.name
            if len(name) > 45:
                name = "..." + name[-42:]
            print(f"|  {name:<45} {size_str:>10} |")
    print("|  " + "-" * 56 + "--------- |")
    print(f"|  {'Total':45} {format_file_size(total_size):>10} |")
    print("+" + "-" * 68 + "+")

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
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable chunk-based processing for faster-whisper (process entire file at once)"
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="#ffffff",
        help="Background color for HTML output (default: #ffffff white). Use hex colors like #f5f5dc for beige"
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
            batch_size=args.batch_size,
            disable_chunking=args.no_chunk
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
    output_files = save_transcript(result, audio_path, duration_seconds, args.model, stats, args.bg_color)

    # Display statistics
    if not args.no_stats:
        display_stats(result, audio_path, duration_seconds, stats, output_files)
    else:
        print("\n" + "=" * 60)
        print("Transcription complete!")


if __name__ == "__main__":
    main()
