#!/usr/bin/env python3
"""
Birdler: command-line voice cloning tool using ChatterboxTTS.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from shutil import which


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio with ChatterboxTTS and a reference sample"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-i", "--audio-sample",
        type=Path,
        default=Path("audio-samples/bigbird/bigbird_youtube_clean.wav"),
        help="Path to the clean reference audio sample"
    )
    group.add_argument(
        "--youtube-url",
        type=str,
        help="YouTube URL to extract audio from (requires yt-dlp or youtube-dl)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("generated-audio"),
        help="Directory to save generated audio or extracted audio"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on (auto-detected if not set)"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.3,
        help="Guidance scale for TTS (higher is more faithful)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.8,
        help="Exaggeration factor for expressiveness"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for TTS generation"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty to discourage repetition in generation"
    )
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--text-file",
        type=Path,
        help="Path to a text file to synthesize (mutually exclusive with --text)"
    )
    group2.add_argument(
        "--text",
        type=str,
        help="Text string to synthesize directly (mutually exclusive with --text-file)"
    )
    return parser.parse_args()


def select_device(preferred=None):
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_text_chunks(args):
    """
    Determine text chunks to synthesize based on args.
    Supports default hardcoded chunks, or user-supplied --text-file or --text.
    Splits long scripts into the same number of chunks as the default.
    """
    # Safe-for-work default: opening stanzas of Bukowski's poem "Style"
    DEFAULT_TEXT_CHUNKS = [
        "style is the answer to everything --",
        "a fresh way to approach a dull or a dangerous thing.",
        "to do a dull thing with style is preferable to doing a dangerous thing without it."
    ]
    if args.text_file or args.text:
        if args.text_file:
            script = args.text_file.read_text()
        else:
            script = args.text
        chunk_count = len(DEFAULT_TEXT_CHUNKS)
        max_def = max(len(c) for c in DEFAULT_TEXT_CHUNKS)
        if len(script) > max_def:
            print(f"Script length {len(script)} > {max_def}, splitting into {chunk_count} chunks")
            size = len(script) // chunk_count
            chunks = [script[i * size:(i + 1) * size] for i in range(chunk_count - 1)]
            chunks.append(script[(chunk_count - 1) * size:])
            return chunks
        return [script]
    return DEFAULT_TEXT_CHUNKS


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.youtube_url:
        ytdlp_cmd = which("yt-dlp") or which("youtube-dl")
        if not ytdlp_cmd:
            print(
                "Error: yt-dlp or youtube-dl is required to extract audio; please install one of them"
            )
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_template = args.output_dir / "%(id)s.%(ext)s"
        print(f"Extracting audio from {args.youtube_url} to {args.output_dir}")
        subprocess.run(
            [
                ytdlp_cmd,
                "--extract-audio",
                "--audio-format",
                "wav",
                "-o",
                str(out_template),
                args.youtube_url,
            ],
            check=True,
        )
        print("Done.")
        return 0

    if not args.audio_sample.exists():
        print(f"Error: audio sample not found: {args.audio_sample}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import heavy dependencies only when performing TTS
    import torch
    import torchaudio
    from chatterbox import ChatterboxTTS

    # Determine text chunks to synthesize
    text_chunks = get_text_chunks(args)
    print(f"Generating {len(text_chunks)} chunks ({sum(len(c) for c in text_chunks)} chars)")
    tts = ChatterboxTTS.from_pretrained(device=device)
    audio_chunks = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"Chunk {i}/{len(text_chunks)}: '{chunk[:30]}...'")
        wav = tts.generate(
            chunk,
            audio_prompt_path=str(args.audio_sample),
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        audio_chunks.append(wav)

    final_audio = torch.cat(audio_chunks, dim=1)
    out_path = args.output_dir / "bigbird_exhausting_week.wav"
    torchaudio.save(str(out_path), final_audio, tts.sr)
    print(f"Saved output to: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
