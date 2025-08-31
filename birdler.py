#!/usr/bin/env python3
"""
Birdler: command-line voice cloning tool using ChatterboxTTS.
- Dynamic, sentence-aware chunking:
  * Soft target (--max-chars, default 280)
  * Hard cap per chunk (--hard-max-chars, default 360) with safe fallback split
- Trims leading/trailing silence per chunk
- Crossfades between chunks
- Timestamped, descriptive output filenames
"""
import argparse
import subprocess
import sys
import time
import re
from pathlib import Path
from shutil import which
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio with ChatterboxTTS and a reference sample"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-i",
        "--audio-sample",
        type=Path,
        default=None,
        help="Reference WAV used to bootstrap a managed voice (requires --voice)",
    )
    # Voice-based workflow (can be combined with --audio-sample on first run)
    parser.add_argument(
        "--voice",
        type=str,
        help="Name of the voice to use (creates/uses voices/<name> directory)",
    )
    parser.add_argument(
        "--voice-dir",
        type=Path,
        default=Path("voices"),
        help="Root directory for managed voice data (default: voices)",
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
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Only set up the managed voice (download/copy + cache embedding), then exit",
    )
    parser.add_argument(
        "--force-voice-ref",
        action="store_true",
        help="Overwrite an existing voices/<voice>/samples/reference.wav during bootstrap",
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
    parser.add_argument(
        "--max-chars",
        type=int,
        default=280,
        help="Soft target characters per chunk (sentence-aware)",
    )
    parser.add_argument(
        "--hard-max-chars",
        type=int,
        default=360,
        help="Hard cap per chunk; overflow is forcibly split on whitespace",
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


DEFAULT_TEXT = (
    "style is the answer to everything -- "
    "a fresh way to approach a dull or a dangerous thing. "
    "to do a dull thing with style is preferable to doing a dangerous thing without it."
)


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _split_into_sentences(text: str) -> list[str]:
    # Simple sentence splitter on ., !, ? while keeping the delimiter.
    # Falls back gracefully for short or delimiter-less inputs.
    pieces = re.split(r"([.!?])", text)
    if len(pieces) == 1:
        return [text.strip()] if text.strip() else []
    out = []
    for i in range(0, len(pieces) - 1, 2):
        sent = (pieces[i] + pieces[i + 1]).strip()
        if sent:
            out.append(sent)
    # Append any trailing fragment
    if len(pieces) % 2 == 1 and pieces[-1].strip():
        out.append(pieces[-1].strip())
    return out


def _greedy_sentence_pack(sentences: list[str], max_chars: int, hard_max: int) -> list[str]:
    """
    Pack sentences into chunks aiming for max_chars. Enforce hard_max by whitespace split.
    """
    chunks = []
    buf = ""
    for s in sentences:
        s_norm = _normalize_whitespace(s)
        candidate = (buf + " " + s_norm).strip() if buf else s_norm
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        # buf has content; flush it
        if buf:
            chunks.append(buf)
            buf = s_norm
        else:
            # Single sentence too long; enforce hard split
            buf = s_norm

        # If current buffer exceeds hard_max, spill by whitespace
        while len(buf) > hard_max:
            cut = buf.rfind(" ", 0, hard_max)
            cut = cut if cut != -1 else hard_max
            chunks.append(buf[:cut].strip())
            buf = buf[cut:].strip()
    if buf:
        chunks.append(buf)
    return chunks


def get_text_chunks(args):
    """
    Determine text chunks to synthesize based on args.
    Sentence-aware packing with soft target (--max-chars) and hard cap (--hard-max-chars).
    """
    if args.text_file or args.text:
        script = args.text_file.read_text() if args.text_file else args.text
        script = _normalize_whitespace(script)
    else:
        script = DEFAULT_TEXT

    if not script:
        return []

    sentences = _split_into_sentences(script)
    if not sentences:
        # No punctuation; split by words
        words = script.split()
        sentences = [" ".join(words)] if len(words) <= args.max_chars else [" ".join(words)]

    chunks = _greedy_sentence_pack(sentences, args.max_chars, args.hard_max_chars)
    print(f"Dynamic chunking: {len(chunks)} chunks (len={len(script)} chars, target={args.max_chars}, hard={args.hard_max_chars})")
    return chunks


def _slugify_text(s: str, max_len: int = 40) -> str:
    s = re.sub(r"\s+", " ", s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- _]", "", s).replace(" ", "-")
    return s[:max_len] or "default"


def trim_silence(wav, thresh: float = 1e-3):
    """
    Trim leading/trailing regions where mean(|x|) <= thresh.
    wav shape: [C, T] or [T].
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return wav
    import torch

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    x = wav.abs().mean(dim=0)  # [T]
    idx = (x > thresh).nonzero(as_tuple=False).squeeze()
    if getattr(idx, "numel", lambda: 0)() == 0:
        return wav
    start, end = int(idx[0].item()), int(idx[-1].item()) + 1
    return wav[:, start:end]


def crossfade_concat(chunks, fade_samples: int = 2048):
    """
    Crossfade-adjacent concatenate to avoid gaps. Assumes same sample rate.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return chunks[0] if chunks else None
    import torch

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


def _derive_outname(args, text_chunks) -> str:
    if getattr(args, "voice", None):
        sample_slug = _slugify_text(args.voice, max_len=40)
    else:
        sample_slug = (args.audio_sample.stem if args.audio_sample else "sample")
    sample_slug = _slugify_text(sample_slug, max_len=40)

    if args.text:
        text_slug = _slugify_text(" ".join(args.text.split()[:6]), max_len=40)
    elif args.text_file:
        text_slug = _slugify_text(args.text_file.stem, max_len=40)
    else:
        first = text_chunks[0] if text_chunks else "default"
        text_slug = _slugify_text(" ".join(first.split()[:6]), max_len=40)

    ts = int(time.time())
    return f"{sample_slug}_{text_slug}_{ts}.wav"


def _voice_paths(voice: str, voices_root: Path) -> dict:
    root = voices_root / voice
    samples = root / "samples"
    embedding = root / "embedding"
    ref_wav = samples / "reference.wav"
    emb_path = embedding / "cond.pt"
    return {
        "root": root,
        "samples": samples,
        "embedding": embedding,
        "ref_wav": ref_wav,
        "emb_path": emb_path,
    }


def prepare_voice(args) -> dict | None:
    """
    Ensure voice directories exist. If a reference wav is provided and missing, copy it.
    Returns a dict of paths when args.voice is set; otherwise None.
    """
    if not getattr(args, "voice", None):
        return None

    vp = _voice_paths(args.voice, args.voice_dir)
    vp["samples"].mkdir(parents=True, exist_ok=True)
    vp["embedding"].mkdir(parents=True, exist_ok=True)

    # Bootstrap reference.wav on first run if provided
    if not vp["ref_wav"].exists() and getattr(args, "audio_sample", None):
        src = args.audio_sample
        if src and src.exists():
            from shutil import copy2

            copy2(src, vp["ref_wav"])
            print(f"[voice] Bootstrapped reference: {vp['ref_wav']}")
    return vp


def get_or_build_embedding(tts, voice_paths: dict):
    """
    Load cached embedding if present; otherwise build via tts.get_audio_conditioning and cache.
    Returns None if API is unavailable.
    """
    emb_path: Path = voice_paths["emb_path"]
    ref_wav: Path = voice_paths["ref_wav"]
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    try:
        import torch
        if emb_path.exists():
            return torch.load(emb_path)
        if hasattr(tts, "get_audio_conditioning") and ref_wav.exists():
            cond = tts.get_audio_conditioning(str(ref_wav))
            try:
                torch.save(cond, emb_path)
                print(f"[voice] Cached embedding: {emb_path}")
            except Exception:
                print("[warn] Failed to cache embedding; continuing without cache")
            return cond
    except Exception as e:
        print(f"[warn] Embedding build/load failed: {e}")
    return None


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    # YouTube extraction
    if args.youtube_url and not args.voice:
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
    elif args.youtube_url and args.voice:
        # Voice-aware bootstrap using YouTube audio directly into reference.wav
        vp = _voice_paths(args.voice, args.voice_dir)
        vp["samples"].mkdir(parents=True, exist_ok=True)
        ref = vp["ref_wav"]
        if ref.exists() and not args.force_voice_ref:
            print(f"[voice] reference.wav exists; keeping: {ref} (use --force-voice-ref to overwrite)")
        else:
            if ref.exists() and args.force_voice_ref:
                try:
                    ref.unlink()
                except OSError:
                    pass
            ytdlp_cmd = which("yt-dlp") or which("youtube-dl")
            if not ytdlp_cmd:
                print("Error: yt-dlp or youtube-dl is required to extract audio; please install one of them")
                return 1
            out_template = vp["samples"] / "reference.%(ext)s"
            print(f"[voice] Extracting reference audio for '{args.voice}' → {ref}")
            subprocess.run(
                [ytdlp_cmd, "--extract-audio", "--audio-format", "wav", "-o", str(out_template), args.youtube_url],
                check=True,
            )

    # TTS generation
    if not args.voice:
        print("Error: --voice is required for TTS generation (YouTube extraction is the only exception).")
        return 1

    voice_paths = prepare_voice(args)
    ref_path = voice_paths["ref_wav"] if voice_paths and voice_paths["ref_wav"].exists() else None
    if not ref_path and not args.audio_sample:
        print("Error: first run for a new --voice requires --audio-sample to bootstrap reference.wav")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy deps
    import torch
    import torchaudio
    from chatterbox import ChatterboxTTS

    # Chunking
    text_chunks = get_text_chunks(args)
    total_chars = sum(len(c) for c in text_chunks)
    print(f"Generating {len(text_chunks)} chunks ({total_chars} chars)")
    # Show preview of the first chunk before processing
    if text_chunks:
        preview = text_chunks[0][:30].replace("\n", " ")
        print(f"Chunk 1/{len(text_chunks)}: '{preview}...'")

    tts = ChatterboxTTS.from_pretrained(device=device)

    # Prepare cached embedding (required)
    audio_prompt_cond = get_or_build_embedding(tts, voice_paths)
    if audio_prompt_cond is None:
        print("Error: TTS backend must support get_audio_conditioning and audio_prompt_cond")
        return 1

    if args.bootstrap_only:
        print(f"[voice] Bootstrap complete for '{args.voice}' (embedding cached).")
        return 0

    def _generate_with_prompt(text: str):
        return tts.generate(
            text,
            audio_prompt_cond=audio_prompt_cond,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

    audio_chunks = []
    for i, chunk in enumerate(text_chunks, 1):
        # Preview remaining chunks starting from the second
        if i > 1:
            preview = chunk[:30].replace("\n", " ")
            print(f"Chunk {i}/{len(text_chunks)}: '{preview}...'")
        wav = _generate_with_prompt(chunk)
        # float [-1, 1], [C, T]
        if not hasattr(wav, "dim"):
            # Some APIs might return numpy; convert if needed
            import numpy as np
            if isinstance(wav, np.ndarray):
                import torch as _torch
                wav = _torch.from_numpy(wav)
        import torch as _torch
        if not _torch.is_floating_point(wav):
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
