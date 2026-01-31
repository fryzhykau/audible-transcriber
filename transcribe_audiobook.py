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
import html as html_lib
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
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


def transcribe_audio(audio_path: Path, model_size: str = "medium") -> tuple:
    """Transcribe audio using Whisper. Returns (result, duration_seconds)."""

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

    return result[0], duration_seconds


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


def save_transcript(result: dict, audio_path: Path, duration_seconds: float, model_size: str):
    """Save transcript to multiple formats."""

    segments = result["segments"]
    chapters = detect_chapters(segments)

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
        for segment in segments:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start} --> {end}]\n{text}\n\n")
    print(f"Timestamped transcript saved to: {ts_path}")

    # SRT subtitle format
    srt_path = audio_path.with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"SRT subtitles saved to: {srt_path}")

    # Markdown with chapters and TOC
    save_markdown(result, audio_path, duration_seconds, model_size, chapters)

    # HTML with chapters and TOC
    save_html(result, audio_path, duration_seconds, model_size, chapters)


def save_markdown(result: dict, audio_path: Path, duration_seconds: float, model_size: str, chapters: list):
    """Save transcript as Markdown with chapters and table of contents."""
    md_path = audio_path.with_suffix('.md')
    segments = result["segments"]

    with open(md_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {audio_path.stem}\n\n")

        # Metadata
        f.write("## Metadata\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        f.write(f"| Source | `{audio_path.name}` |\n")
        f.write(f"| Duration | {format_timestamp(duration_seconds)} |\n")
        f.write(f"| Model | {model_size} |\n")
        f.write(f"| Language | {result.get('language', 'en')} |\n")
        f.write(f"| Chapters | {len(chapters)} |\n")
        f.write(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")

        # Table of Contents
        f.write("## Table of Contents\n\n")
        for i, chapter in enumerate(chapters):
            anchor = f"chapter-{i + 1}"
            timestamp = format_timestamp(chapter['start_time'])
            f.write(f"{i + 1}. [{chapter['title']}](#{anchor}) `{timestamp}`\n")
        f.write("\n---\n\n")

        # Chapters with content
        for i, chapter in enumerate(chapters):
            anchor = f"chapter-{i + 1}"
            f.write(f"## <a id=\"{anchor}\"></a>{i + 1}. {chapter['title']}\n\n")
            f.write(f"*Starts at {format_timestamp(chapter['start_time'])}*\n\n")

            # Get segments for this chapter
            start_idx = chapter['start_segment_idx']
            end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

            for segment in segments[start_idx:end_idx]:
                start = format_timestamp(segment["start"])
                text = segment["text"].strip()
                f.write(f"**`{start}`** {text}\n\n")

            f.write("\n---\n\n")

    print(f"Markdown transcript saved to: {md_path}")
    return md_path


def save_html(result: dict, audio_path: Path, duration_seconds: float, model_size: str, chapters: list):
    """Save transcript as HTML with chapters, TOC, and navigation."""
    html_path = audio_path.with_suffix('.html')
    segments = result["segments"]

    # Build TOC HTML
    toc_html = ""
    for i, chapter in enumerate(chapters):
        timestamp = format_timestamp(chapter['start_time'])
        toc_html += f'''
        <li>
            <a href="#chapter-{i + 1}">{html_lib.escape(chapter['title'])}</a>
            <span class="toc-time">{timestamp}</span>
        </li>'''

    # Build chapters HTML
    chapters_html = ""
    for i, chapter in enumerate(chapters):
        start_idx = chapter['start_segment_idx']
        end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

        segments_html = ""
        for segment in segments[start_idx:end_idx]:
            start = format_timestamp(segment["start"])
            start_secs = segment["start"]
            text = html_lib.escape(segment["text"].strip())
            segments_html += f'''
            <div class="segment" data-start="{start_secs}">
                <span class="timestamp">{start}</span>
                <span class="text">{text}</span>
            </div>'''

        chapters_html += f'''
        <section class="chapter" id="chapter-{i + 1}">
            <h2>
                <span class="chapter-num">{i + 1}</span>
                {html_lib.escape(chapter['title'])}
            </h2>
            <p class="chapter-meta">Starts at {format_timestamp(chapter['start_time'])}</p>
            <div class="chapter-content">
                {segments_html}
            </div>
        </section>'''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_lib.escape(audio_path.stem)}</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --text-color: #c9d1d9;
            --accent-color: #238636;
            --highlight-color: #58a6ff;
            --border-color: #30363d;
            --timestamp-color: #7ee787;
            --chapter-color: #f78166;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.7;
            background: var(--bg-color);
            color: var(--text-color);
        }}
        .container {{
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 100vh;
        }}
        .sidebar {{
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            background: var(--card-bg);
            border-right: 1px solid var(--border-color);
            padding: 20px;
        }}
        .sidebar h1 {{
            font-size: 1.1em;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        .metadata {{
            font-size: 0.8em;
            color: #8b949e;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }}
        .metadata div {{ margin: 5px 0; }}
        .toc {{ list-style: none; }}
        .toc li {{
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .toc a {{
            color: var(--text-color);
            text-decoration: none;
            font-size: 0.9em;
            flex: 1;
            padding: 5px 8px;
            border-radius: 6px;
            transition: background 0.2s;
        }}
        .toc a:hover {{
            background: var(--border-color);
            color: var(--highlight-color);
        }}
        .toc-time {{
            font-family: monospace;
            font-size: 0.75em;
            color: var(--timestamp-color);
            margin-left: 10px;
        }}
        .main {{
            padding: 40px;
            max-width: 900px;
        }}
        .search-box {{
            position: sticky;
            top: 0;
            background: var(--bg-color);
            padding: 15px 0;
            margin-bottom: 20px;
            z-index: 100;
        }}
        #search {{
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: var(--card-bg);
            color: var(--text-color);
            font-size: 1em;
        }}
        #search:focus {{
            outline: none;
            border-color: var(--highlight-color);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
        }}
        .chapter {{
            margin-bottom: 50px;
            background: var(--card-bg);
            border-radius: 8px;
            padding: 25px;
            border: 1px solid var(--border-color);
        }}
        .chapter h2 {{
            color: var(--chapter-color);
            font-size: 1.4em;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .chapter-num {{
            background: var(--chapter-color);
            color: var(--bg-color);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .chapter-meta {{
            color: #8b949e;
            font-size: 0.85em;
            margin-bottom: 20px;
        }}
        .segment {{
            padding: 10px 0;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 15px;
        }}
        .segment:last-child {{ border-bottom: none; }}
        .segment .timestamp {{
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.8em;
            color: var(--timestamp-color);
            background: rgba(126, 231, 135, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            white-space: nowrap;
            height: fit-content;
        }}
        .segment .text {{
            flex: 1;
        }}
        .highlight {{
            background: rgba(88, 166, 255, 0.3);
            padding: 0 2px;
            border-radius: 2px;
        }}
        footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            font-size: 0.8em;
            color: #8b949e;
            border-top: 1px solid var(--border-color);
        }}
        @media (max-width: 768px) {{
            .container {{ grid-template-columns: 1fr; }}
            .sidebar {{
                position: relative;
                height: auto;
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }}
            .main {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <h1>{html_lib.escape(audio_path.stem)}</h1>
            <div class="metadata">
                <div><strong>Duration:</strong> {format_timestamp(duration_seconds)}</div>
                <div><strong>Model:</strong> {model_size}</div>
                <div><strong>Chapters:</strong> {len(chapters)}</div>
                <div><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d')}</div>
            </div>
            <h3 style="font-size: 0.9em; margin-bottom: 10px; color: #8b949e;">Table of Contents</h3>
            <ul class="toc">
                {toc_html}
            </ul>
        </aside>

        <main class="main">
            <div class="search-box">
                <input type="text" id="search" placeholder="Search transcript...">
            </div>

            {chapters_html}

            <footer>
                Generated by Audible Transcriber using OpenAI Whisper
            </footer>
        </main>
    </div>

    <script>
        document.getElementById('search').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            const segments = document.querySelectorAll('.segment');
            const chapters = document.querySelectorAll('.chapter');

            if (!query) {{
                segments.forEach(seg => {{
                    seg.style.display = 'flex';
                    const text = seg.querySelector('.text');
                    text.innerHTML = text.textContent;
                }});
                chapters.forEach(ch => ch.style.display = 'block');
                return;
            }}

            chapters.forEach(chapter => {{
                let hasMatch = false;
                const chapterSegments = chapter.querySelectorAll('.segment');

                chapterSegments.forEach(seg => {{
                    const text = seg.querySelector('.text');
                    const original = text.textContent;

                    if (original.toLowerCase().includes(query)) {{
                        seg.style.display = 'flex';
                        hasMatch = true;
                        const regex = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
                        text.innerHTML = original.replace(regex, '<span class="highlight">$1</span>');
                    }} else {{
                        seg.style.display = 'none';
                        text.innerHTML = original;
                    }}
                }});

                chapter.style.display = hasMatch ? 'block' : 'none';
            }});
        }});

        // Smooth scroll for TOC links
        document.querySelectorAll('.toc a').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }});
        }});
    </script>
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML transcript saved to: {html_path}")
    return html_path


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
    result, duration_seconds = transcribe_audio(audio_path, args.model)

    # Save transcripts
    save_transcript(result, audio_path, duration_seconds, args.model)

    print("\n" + "=" * 60)
    print("Transcription complete!")


if __name__ == "__main__":
    main()
