"""
Birdler: command-line voice cloning tool using ChatterboxTTS.
- Splits long input text into chunks.
- Trims leading/trailing silence of each generated chunk.
- Crossfades between chunks to avoid gaps.
- Writes an output filename based on sample/text plus a Unix timestamp.
"""
import argparse
import subprocess
import sys
import time
import re
from pathlib import Path
from shutil import which


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio with ChatterboxTTS and a reference sample"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-i",
        "--audio-sample",
        type=Path,
        default=Path("audio-samples/bigbird/bigbird_youtube_clean.wav"),
        help="Path to the clean reference audio sample",
    )
    group.add_argument(
        "--youtube-url",
        type=str,
        help="YouTube URL to extract audio from (requires yt-dlp or youtube-dl)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("generated-audio"),
        help="Directory to save generated audio or extracted audio",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on (auto-detected if not set)",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.3,
        help="Guidance scale for TTS (higher is more faithful)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.8,
        help="Exaggeration factor for expressiveness",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for TTS generation",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty to discourage repetition in generation",
    )
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--text-file",
        type=Path,
        help="Path to a text file to synthesize (mutually exclusive with --text)",
    )
    group2.add_argument(
        "--text",
        type=str,
        help="Text string to synthesize directly (mutually exclusive with --text-file)",
    )
    return parser.parse_args()


def select_device(preferred=None):
    if preferred:
        return preferred
    try:
        import torch  # noqa: F401
    except ImportError:
        return "cpu"
    import torch

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
    # Safe-for-work default: opening stanzas of Bukowski's "Style" (short excerpt)
    DEFAULT_TEXT_CHUNKS = [
        "style is the answer to everything --",
        "a fresh way to approach a dull or a dangerous thing.",
        "to do a dull thing with style is preferable to doing a dangerous thing without it.",
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
            chunks = [script[i * size: (i + 1) * size] for i in range(chunk_count - 1)]
            chunks.append(script[(chunk_count - 1) * size:])
            return chunks
        return [script]
    return DEFAULT_TEXT_CHUNKS


def _slugify_text(s: str, max_len: int = 40) -> str:
    s = re.sub(r"\s+", " ", s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- _]", "", s).replace(" ", "-")
    return s[:max_len] or "default"


def _derive_outname(args, text_chunks) -> str:
    # sample slug
    sample_slug = (args.audio_sample.stem if args.audio_sample else "sample")
    sample_slug = _slugify_text(sample_slug, max_len=40)

    # text slug preference: direct --text, then --text-file name, then first chunk
    if args.text:
        text_slug = _slugify_text(" ".join(args.text.split()[:6]), max_len=40)
    elif args.text_file:
        text_slug = _slugify_text(args.text_file.stem, max_len=40)
    else:
        first = text_chunks[0] if text_chunks else "default"
        text_slug = _slugify_text(" ".join(first.split()[:6]), max_len=40)

    ts = int(time.time())
    return f"{sample_slug}_{text_slug}_{ts}.wav"


def main():
    args = parse_args()
    # Prompt if output directory does not exist
    if not args.output_dir.exists():
        create = input(f"Output directory {args.output_dir!r} does not exist. Create it? [y/N]: ")
        if create.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1
        args.output_dir.mkdir(parents=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    # YouTube extraction path
    if args.youtube_url:
        ytdlp_cmd = which("yt-dlp") or which("youtube-dl")
        if not ytdlp_cmd:
            print("Error: yt-dlp or youtube-dl is required to extract audio; please install one of them")
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_template = args.output_dir / "%(id)s.%(ext)s"
        print(f"Extracting audio from {args.youtube_url} to {args.output_dir}")
        subprocess.run(
            [ytdlp_cmd, "--extract-audio", "--audio-format", "wav", "-o", str(out_template), args.youtube_url],
            check=True,
        )
        print("Done.")
        return 0

    # TTS generation path
    if not args.audio_sample or not args.audio_sample.exists():
        print(f"Error: audio sample not found: {args.audio_sample}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import heavy deps here
    import torch
    import torchaudio
    from chatterbox import ChatterboxTTS

    def trim_silence(wav: torch.Tensor, thresh: float = 1e-3) -> torch.Tensor:
        """
        Trim leading and trailing regions where mean(|x|) <= thresh.
        wav shape: [C, T] or [T].
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = wav.abs().mean(dim=0)  # [T]
        idx = (x > thresh).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            return wav
        start, end = int(idx[0].item()), int(idx[-1].item()) + 1
        return wav[:, start:end]

    def crossfade_concat(chunks, fade_samples: int = 2048) -> torch.Tensor:
        """
        Crossfade-adjacent concatenate to avoid gaps. Assumes same sample rate.
        """
        if not chunks:
            return torch.zeros(1, 0)
        out = chunks[0]
        for nxt in chunks[1:]:
            # channel align
            if nxt.size(0) != out.size(0):
                if nxt.size(0) == 1:
                    nxt = nxt.repeat(out.size(0), 1)
                elif out.size(0) == 1:
                    out = out.repeat(nxt.size(0), 1)
                else:
                    c = min(out.size(0), nxt.size(0))
                    out, nxt = out[:c], nxt[:c]
            f = min(fade_samples, out.size(1), nxt.size(1))
            if f <= 0:
                out = torch.cat([out, nxt], dim=1)
                continue
            fade_out = torch.linspace(1.0, 0.0, f, device=out.device).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, f, device=out.device).unsqueeze(0)
            x_tail = out[:, -f:] * fade_out
            y_head = nxt[:, :f] * fade_in
            blended = x_tail + y_head
            out = torch.cat([out[:, :-f], blended, nxt[:, f:]], dim=1)
        return out

    # Determine text chunks to synthesize
    text_chunks = get_text_chunks(args)
    print(f"Generating {len(text_chunks)} chunks ({sum(len(c) for c in text_chunks)} chars)")

    tts = ChatterboxTTS.from_pretrained(device=device)

    audio_chunks = []
    for i, chunk in enumerate(text_chunks, 1):
        preview = chunk[:30].replace("\n", " ")
        print(f"Chunk {i}/{len(text_chunks)}: '{preview}...'")
        wav = tts.generate(
            chunk,
            audio_prompt_path=str(args.audio_sample),
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        # to float [-1, 1], [C, T]
        if not torch.is_floating_point(wav):
            wav = wav.float() / 32768.0
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = trim_silence(wav, thresh=1e-3)
        audio_chunks.append(wav)

    # Concatenate with crossfade to remove audible gaps
    final_audio = crossfade_concat(audio_chunks, fade_samples=2048).contiguous()

    # Ensure dtype and sane amplitude
    final_audio = final_audio.clamp(-1.0, 1.0).to(dtype=torch.float32)

    # Build dynamic filename
    out_filename = _derive_outname(args, text_chunks)
    out_path = args.output_dir / out_filename

    # Save
    sr = getattr(tts, "sr", 22050)
    torchaudio.save(str(out_path), final_audio, sr)
    print(f"Saved output to: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
