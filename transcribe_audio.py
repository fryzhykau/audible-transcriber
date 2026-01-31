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
import html
import re
import sys
import threading
from datetime import datetime
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

    # Save Markdown and HTML versions
    save_markdown(result, audio_file, duration_seconds, model_size)
    save_html(result, audio_file, duration_seconds, model_size)

    return result

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_chapters(segments: list) -> list:
    """
    Detect chapter/section breaks based on text patterns or create time-based sections.
    Returns list of chapter dicts with 'title', 'start_time', 'start_segment_idx'
    """
    chapters = []
    chapter_patterns = [
        r'^chapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^part\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^section\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'^act\s+(\d+|one|two|three|four|five)',
        r'^prologue', r'^epilogue', r'^introduction', r'^conclusion',
    ]

    for i, segment in enumerate(segments):
        text = segment["text"].strip().lower()
        for pattern in chapter_patterns:
            if re.match(pattern, text):
                title = segment["text"].strip()
                if len(title) > 60:
                    title = title[:57] + "..."
                chapters.append({
                    'title': title,
                    'start_time': segment["start"],
                    'start_segment_idx': i,
                })
                break

    # If no chapters detected, create time-based sections (every 10 min for shorter audio)
    if not chapters and segments:
        total_duration = segments[-1]["end"]
        section_length = 600 if total_duration < 3600 else 1800  # 10 or 30 min
        current_section = 0

        for i, segment in enumerate(segments):
            section_num = int(segment["start"] // section_length)
            if section_num > current_section or i == 0:
                current_section = section_num
                chapters.append({
                    'title': f"Section {len(chapters) + 1}",
                    'start_time': segment["start"],
                    'start_segment_idx': i,
                })

    return chapters


def save_markdown(result: dict, audio_file: Path, duration_seconds: float, model_size: str):
    """Save transcript as Markdown with chapters and table of contents."""
    md_path = audio_file.with_suffix('.md')
    segments = result["segments"]
    chapters = detect_chapters(segments)

    with open(md_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {audio_file.stem}\n\n")

        # Metadata
        f.write("## Metadata\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        f.write(f"| Source | `{audio_file.name}` |\n")
        f.write(f"| Duration | {format_timestamp(duration_seconds)} |\n")
        f.write(f"| Model | {model_size} |\n")
        f.write(f"| Language | {result.get('language', 'auto-detected')} |\n")
        f.write(f"| Sections | {len(chapters)} |\n")
        f.write(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")

        # Table of Contents
        f.write("## Table of Contents\n\n")
        for i, chapter in enumerate(chapters):
            anchor = f"section-{i + 1}"
            timestamp = format_timestamp(chapter['start_time'])
            f.write(f"{i + 1}. [{chapter['title']}](#{anchor}) `{timestamp}`\n")
        f.write("\n---\n\n")

        # Sections with content
        for i, chapter in enumerate(chapters):
            anchor = f"section-{i + 1}"
            f.write(f"## <a id=\"{anchor}\"></a>{i + 1}. {chapter['title']}\n\n")
            f.write(f"*Starts at {format_timestamp(chapter['start_time'])}*\n\n")

            start_idx = chapter['start_segment_idx']
            end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

            for segment in segments[start_idx:end_idx]:
                start = format_timestamp(segment["start"])
                text = segment["text"].strip()
                f.write(f"**`{start}`** {text}\n\n")

            f.write("\n---\n\n")

    print(f"Markdown transcript saved to: {md_path}")
    return md_path


def save_html(result: dict, audio_file: Path, duration_seconds: float, model_size: str):
    """Save transcript as HTML with chapters, TOC, and search."""
    html_path = audio_file.with_suffix('.html')
    segments = result["segments"]
    chapters = detect_chapters(segments)

    # Build TOC HTML
    toc_html = ""
    for i, chapter in enumerate(chapters):
        timestamp = format_timestamp(chapter['start_time'])
        toc_html += f'''
        <li>
            <a href="#section-{i + 1}">{html.escape(chapter['title'])}</a>
            <span class="toc-time">{timestamp}</span>
        </li>'''

    # Build sections HTML
    sections_html = ""
    for i, chapter in enumerate(chapters):
        start_idx = chapter['start_segment_idx']
        end_idx = chapters[i + 1]['start_segment_idx'] if i + 1 < len(chapters) else len(segments)

        segments_inner = ""
        for segment in segments[start_idx:end_idx]:
            start = format_timestamp(segment["start"])
            start_secs = segment["start"]
            text = html.escape(segment["text"].strip())
            segments_inner += f'''
            <div class="segment" data-start="{start_secs}">
                <span class="timestamp">{start}</span>
                <span class="text">{text}</span>
            </div>'''

        sections_html += f'''
        <section class="chapter" id="section-{i + 1}">
            <h2><span class="chapter-num">{i + 1}</span>{html.escape(chapter['title'])}</h2>
            <p class="chapter-meta">Starts at {format_timestamp(chapter['start_time'])}</p>
            <div class="chapter-content">{segments_inner}</div>
        </section>'''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(audio_file.stem)}</title>
    <style>
        :root {{
            --bg: #0d1117; --card: #161b22; --text: #c9d1d9;
            --border: #30363d; --accent: #58a6ff; --highlight: #f78166; --time: #7ee787;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); line-height: 1.7; }}
        .container {{ display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }}
        .sidebar {{ position: sticky; top: 0; height: 100vh; overflow-y: auto; background: var(--card); border-right: 1px solid var(--border); padding: 20px; }}
        .sidebar h1 {{ font-size: 1em; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }}
        .meta {{ font-size: 0.8em; color: #8b949e; margin-bottom: 20px; }}
        .meta div {{ margin: 4px 0; }}
        .toc {{ list-style: none; }}
        .toc li {{ margin: 6px 0; display: flex; justify-content: space-between; }}
        .toc a {{ color: var(--text); text-decoration: none; font-size: 0.85em; padding: 4px 8px; border-radius: 4px; flex: 1; }}
        .toc a:hover {{ background: var(--border); color: var(--accent); }}
        .toc-time {{ font-family: monospace; font-size: 0.7em; color: var(--time); }}
        .main {{ padding: 30px; max-width: 850px; }}
        .search-box {{ position: sticky; top: 0; background: var(--bg); padding: 12px 0; z-index: 100; }}
        #search {{ width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--text); }}
        #search:focus {{ outline: none; border-color: var(--accent); }}
        .chapter {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 25px; }}
        .chapter h2 {{ color: var(--highlight); font-size: 1.2em; display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }}
        .chapter-num {{ background: var(--highlight); color: var(--bg); width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75em; }}
        .chapter-meta {{ color: #8b949e; font-size: 0.8em; margin-bottom: 15px; }}
        .segment {{ padding: 8px 0; border-bottom: 1px solid var(--border); display: flex; gap: 12px; }}
        .segment:last-child {{ border-bottom: none; }}
        .timestamp {{ font-family: monospace; font-size: 0.75em; color: var(--time); background: rgba(126,231,135,0.1); padding: 2px 6px; border-radius: 3px; white-space: nowrap; }}
        .segment .text {{ flex: 1; }}
        .highlight {{ background: rgba(88,166,255,0.3); padding: 0 2px; border-radius: 2px; }}
        footer {{ margin-top: 30px; text-align: center; font-size: 0.75em; color: #8b949e; }}
        @media (max-width: 768px) {{ .container {{ grid-template-columns: 1fr; }} .sidebar {{ position: relative; height: auto; }} }}
    </style>
</head>
<body>
<div class="container">
    <aside class="sidebar">
        <h1>{html.escape(audio_file.stem)}</h1>
        <div class="meta">
            <div><strong>Duration:</strong> {format_timestamp(duration_seconds)}</div>
            <div><strong>Model:</strong> {model_size}</div>
            <div><strong>Sections:</strong> {len(chapters)}</div>
        </div>
        <ul class="toc">{toc_html}</ul>
    </aside>
    <main class="main">
        <div class="search-box"><input type="text" id="search" placeholder="Search transcript..."></div>
        {sections_html}
        <footer>Generated by Audible Transcriber using OpenAI Whisper</footer>
    </main>
</div>
<script>
document.getElementById('search').addEventListener('input', function(e) {{
    const q = e.target.value.toLowerCase();
    document.querySelectorAll('.chapter').forEach(ch => {{
        let hasMatch = false;
        ch.querySelectorAll('.segment').forEach(seg => {{
            const text = seg.querySelector('.text');
            const orig = text.textContent;
            if (q && orig.toLowerCase().includes(q)) {{
                seg.style.display = 'flex'; hasMatch = true;
                text.innerHTML = orig.replace(new RegExp('(' + q.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi'), '<span class="highlight">$1</span>');
            }} else if (q) {{ seg.style.display = 'none'; text.innerHTML = orig; }}
            else {{ seg.style.display = 'flex'; text.innerHTML = orig; }}
        }});
        ch.style.display = q && !hasMatch ? 'none' : 'block';
    }});
}});
document.querySelectorAll('.toc a').forEach(a => a.addEventListener('click', e => {{
    e.preventDefault(); document.querySelector(a.getAttribute('href')).scrollIntoView({{ behavior: 'smooth' }});
}}));
</script>
</body>
</html>'''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML transcript saved to: {html_path}")
    return html_path

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
