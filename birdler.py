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
from importlib.metadata import version, PackageNotFoundError
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
        default=0.5,
        help="Guidance scale for TTS (higher is more faithful)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration factor for expressiveness",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for TTS generation",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty to discourage repetition in generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base random seed for deterministic generation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms where supported (may reduce speed)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers for chunk generation"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=1, help="Number of candidates per chunk"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=1, help="Max retries per chunk when validating"
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Only set up the managed voice (download/copy + cache embedding), then exit",
    )
    parser.add_argument(
        "--build-embedding",
        action="store_true",
        help="After setting up the managed voice, compute and cache the embedding now",
    )
    parser.add_argument(
        "--force-voice-ref",
        action="store_true",
        help="Overwrite an existing voices/<voice>/samples/reference.wav during bootstrap",
    )
    parser.add_argument(
        "--compat-legacy-prompt",
        action="store_true",
        help="Use path-based prompting when embeddings are unsupported by backend (no caching)",
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
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum characters per chunk (merge short adjacent chunks)",
    )
    parser.add_argument("--warmup", action="store_true", help="Run a short warmup generate to reduce first-latency")
    parser.add_argument("--no-trim", action="store_true", help="Disable silence trimming on chunks")
    parser.add_argument("--no-crossfade", action="store_true", help="Disable crossfade concat between chunks")
    parser.add_argument("--denoise", action="store_true", help="Apply RNNoise denoise after render (requires CLI denoise and ffmpeg)")
    parser.add_argument("--auto-editor", action="store_true", help="Run auto-editor to tighten long silences (requires auto-editor)")
    parser.add_argument("--auto-editor-threshold", type=float, default=0.06, help="Silence threshold for auto-editor")
    parser.add_argument("--auto-editor-margin", type=float, default=0.2, help="Silence margin (s) for auto-editor")
    parser.add_argument("--keep-original", action="store_true", help="Keep original WAV when using auto-editor")
    parser.add_argument("--normalize", choices=["ebu", "peak"], help="Normalize output loudness with ffmpeg")
    parser.add_argument("--validate", action="store_true", help="Enable validation/selection pipeline")
    parser.add_argument("--validate-threshold", type=float, default=0.85, help="Validation score threshold")
    parser.add_argument("--prefer-longest-on-fail", action="store_true", help="On failure, prefer longest transcript")
    parser.add_argument("--whisper-backend", type=str, default="faster", help="Whisper backend (openai|faster)")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model name")
    return parser.parse_args()


def select_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu"


def set_determinism(device: str):
    try:
        import torch
    except Exception:
        return
    try:
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    import torch, sys as _sys
    try:
        is_macos = (_sys.platform == "darwin")
    except Exception:
        is_macos = False
    if is_macos and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_determinism(device: str):
    try:
        import torch
    except Exception:
        return
    try:
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


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

    try:
        from chunking import get_chunks as _ext_get_chunks
        chunks = _ext_get_chunks(script, soft_max=args.max_chars, hard_max=args.hard_max_chars, min_len=args.min_chars)
    except Exception:
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
    gen_part = f"_gen{getattr(args, 'run_index', 0)}"
    seed_val = getattr(args, 'seed', None)
    seed_part = f"_seed{seed_val}" if seed_val is not None else ""
    return f"{sample_slug}_{text_slug}{gen_part}{seed_part}_{ts}.wav"



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

    # Bootstrap or overwrite reference.wav from provided sample
    src = getattr(args, "audio_sample", None)
    ref = vp["ref_wav"]
    if ref.exists():
        if getattr(args, "force_voice_ref", False) and src and src.exists():
            from shutil import copy2
            copy2(src, ref)
            print(f"[voice] Overwrote existing reference: {ref}")
        else:
            print(f"[voice] reference.wav exists; keeping: {ref}")
    else:
        if src and src.exists():
            from shutil import copy2
            copy2(src, ref)
            print(f"[voice] Bootstrapped reference: {ref}")
    return vp


def get_or_build_embedding(tts, voice_paths: dict, device: str | None = None, exaggeration: float | None = None):
    """
    Preferred path (chatterbox>=0.1.x):
    - Load/save Conditionals via chatterbox.tts.Conditionals
    Fallback (older behavior):
    - Use torch.load/save on a raw object if present.
    Returns an object representing the loaded/created conditioning, or None.
    """
    emb_path: Path = voice_paths["emb_path"]
    ref_wav: Path = voice_paths["ref_wav"]
    try:
        import torch  # noqa: F401
    except Exception:
        print("Error: torch not available; cannot build or load embedding")
        return None
    try:
        import torch
        # Try Conditionals API
        try:
            from chatterbox.tts import Conditionals  # type: ignore
        except Exception:
            Conditionals = None  # type: ignore

        if emb_path.exists():
            if Conditionals is not None:
                conds = Conditionals.load(emb_path, map_location=(device or "cpu"))
                if hasattr(conds, "to"):
                    conds = conds.to(device or "cpu")
                # attach to tts if possible
                try:
                    tts.conds = conds
                except Exception:
                    pass
                return conds
            # Raw torch object fallback
            return torch.load(emb_path)

        if not ref_wav.exists():
            print(f"Error: reference WAV missing at: {ref_wav}")
            return None

        # Build using preferred API: prepare_conditionals
        if hasattr(tts, "prepare_conditionals"):
            try:
                tts.prepare_conditionals(str(ref_wav), exaggeration=exaggeration or 0.5)
                conds = getattr(tts, "conds", None)
                if conds is None:
                    print("Error: prepare_conditionals did not set tts.conds")
                    return None
                # Save via Conditionals.save if available
                if Conditionals is not None and hasattr(conds, "save"):
                    try:
                        conds.save(emb_path)
                        print(f"[voice] Cached embedding: {emb_path}")
                    except Exception as e:
                        print(f"[warn] Failed to cache embedding to {emb_path}: {e}")
                else:
                    # Raw torch save as a best-effort
                    try:
                        torch.save(conds, emb_path)
                        print(f"[voice] Cached embedding (raw): {emb_path}")
                    except Exception as e:
                        print(f"[warn] Failed to cache raw embedding to {emb_path}: {e}")
                return conds
            except Exception as e:
                print(f"[warn] prepare_conditionals failed: {e}")

        # Legacy embedding API (if any)
        if hasattr(tts, "get_audio_conditioning"):
            try:
                cond = tts.get_audio_conditioning(str(ref_wav))
                try:
                    torch.save(cond, emb_path)
                    print(f"[voice] Cached embedding: {emb_path}")
                except Exception as e:
                    print(f"[warn] Failed to cache embedding to {emb_path}: {e}")
                return cond
            except Exception as e:
                print(f"[warn] get_audio_conditioning failed: {e}")

    except Exception as e:
        print(f"[warn] Embedding build/load failed: {e}")
    return None


def _print_embedding_diagnostics(tts, voice_paths: dict):
    ref_wav: Path = voice_paths["ref_wav"]
    emb_path: Path = voice_paths["emb_path"]
    print("Embedding diagnostics:")
    print(f"- voice ref: {ref_wav} (exists={ref_wav.exists()})")
    print(f"- target emb: {emb_path}")
    print(f"- tts.has_get_audio_conditioning={hasattr(tts, 'get_audio_conditioning')}")


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")
    if args.deterministic:
        set_determinism(device)

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
        if not args.build_embedding:
            print("[voice] Reference downloaded. Skipping embedding build for YouTube sources by default.")
            print("       Review/clean the reference.wav, then run with --build-embedding.")
            return 0
        # If building embedding now, continue into TTS path but exit early after embedding
        args.bootstrap_only = True

    # TTS generation
    if not args.voice:
        print("Error: --voice is required for TTS generation (YouTube extraction is the only exception).")
        return 1

    # If a WAV was provided for bootstrap, it must exist (outside YouTube flow)
    if args.audio_sample and not args.youtube_url and not args.audio_sample.exists():
        print(f"Error: provided --audio-sample not found: {args.audio_sample}")
        return 1

    voice_paths = prepare_voice(args)
    ref_path = voice_paths["ref_wav"] if voice_paths and voice_paths["ref_wav"].exists() else None
    if not ref_path and not args.audio_sample:
        print("Error: first run for a new --voice requires --audio-sample to bootstrap reference.wav")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy deps
    # Dependency checks and informative versions
    try:
        import chatterbox  # noqa: F401
        try:
            _cbx_ver = version("chatterbox-tts")
        except PackageNotFoundError:
            print("[warn] chatterbox-tts package metadata not found; proceeding (ensure chatterbox-tts>=0.1.2 is installed)")
            _cbx_ver = "unknown"
    except Exception:
        print("Error: chatterbox-tts is not installed. Try: pip install 'chatterbox-tts>=0.1.2'")
        return 1

    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

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
    audio_prompt_cond = get_or_build_embedding(tts, voice_paths, device=device, exaggeration=args.exaggeration)
    if audio_prompt_cond is None and not args.compat_legacy_prompt:
        _print_embedding_diagnostics(tts, voice_paths)
        print("Error: embedding not available; see diagnostics above.")
        return 1

    # Optional warmup
    if args.warmup:
        try:
            _ = tts.generate(
                "OK.",
                cfg_weight=args.cfg_weight,
                exaggeration=args.exaggeration,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception:
            pass

    if args.bootstrap_only:
        if args.build_embedding:
            if audio_prompt_cond is None:
                if args.compat_legacy_prompt:
                    _print_embedding_diagnostics(tts, voice_paths)
                    print("[voice] Bootstrap complete. Note: embedding API unavailable; compat mode will use path-based prompting at runtime.")
                    return 0
                _print_embedding_diagnostics(tts, voice_paths)
                print("Error: embedding build requested but unavailable; re-run with --compat-legacy-prompt or upgrade chatterbox-tts.")
                return 1
            print(f"[voice] Bootstrap complete for '{args.voice}' (embedding cached).")
            return 0
        # No embedding requested → success if reference exists
        print(f"[voice] Bootstrap complete for '{args.voice}'.")
        return 0

    def _process_wav(wav):
        if not hasattr(wav, "dim"):
            import numpy as np
            if isinstance(wav, np.ndarray):
                import torch as _torch
                wav = _torch.from_numpy(wav)
        import torch as _torch
        if not _torch.is_floating_point(wav):
            wav = wav.float() / 32768.0
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if not args.no_trim:
            wav = trim_silence(wav, thresh=1e-3)
        return wav

    def _generate_with_prompt(text: str):
        # If tts has prepared conditionals (cached embedding loaded or built), call without path
        if getattr(tts, 'conds', None) is not None:
            return tts.generate(
                text,
                cfg_weight=args.cfg_weight,
                exaggeration=args.exaggeration,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
        # compat path-based prompting
        return tts.generate(
            text,
            audio_prompt_path=str(ref_path),
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

    audio_chunks = []
    use_parallel = args.workers > 1 or args.n_candidates > 1 or args.max_attempts > 1
    if use_parallel:
        try:
            import gen_parallel
        except Exception:
            use_parallel = False
    if use_parallel:
        base_seed = args.seed or int(time.time())
        gen_kwargs = {
            "device": device,
            "cfg_weight": args.cfg_weight,
            "exaggeration": args.exaggeration,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        }
        result_map = gen_parallel.generate_chunks_parallel(
            tts,
            text_chunks,
            base_seed,
            workers=max(1, args.workers),
            n_cands=max(1, args.n_candidates),
            attempts=max(1, args.max_attempts),
            gen_kwargs=gen_kwargs,
        )
        for i in range(len(text_chunks)):
            items = result_map.get(i, [])
            pick = None
            for it in items:
                if it.get("cand_idx") == 0 and it.get("attempt") == 0:
                    pick = it
                    break
            if pick is None and items:
                pick = items[0]
            if pick is None:
                continue
            wav = pick["wav"]
            wav = _process_wav(wav)
            audio_chunks.append(wav)
    else:
        for i, chunk in enumerate(text_chunks, 1):
            try:
                from progress import log_progress as _lp
                _lp(i, len(text_chunks))
            except Exception:
                pass
            if i > 1:
                preview = chunk[:30].replace("\n", " ")
                print(f"Chunk {i}/{len(text_chunks)}: '{preview}...'")
            wav = _generate_with_prompt(chunk)
            wav = _process_wav(wav)
            audio_chunks.append(wav)
    # Concatenate with crossfade to remove audible gaps
    if args.no_crossfade:
        import torch as _torch
        final_audio = _torch.cat(audio_chunks, dim=1)
    else:
        adaptive_fade = max(256, min(2048, int(0.02 * getattr(tts, 'sr', 22050))))
        final_audio = crossfade_concat(audio_chunks, fade_samples=adaptive_fade).contiguous()

    # Ensure dtype and sane amplitude
    final_audio = final_audio.clamp(-1.0, 1.0).to(dtype=torch.float32)

    # Build dynamic filename
    out_filename = _derive_outname(args, text_chunks)
    out_path = args.output_dir / out_filename

    # Save
    sr = getattr(tts, "sr", 22050)
    torchaudio.save(str(out_path), final_audio, sr)
    # Optional post-processing
    try:
        import postprocess as _pp
        if args.denoise:
            _ = _pp.denoise_in_place(out_path)
        if args.auto_editor:
            _pp.run_auto_editor_in_place(out_path, threshold=args.auto_editor_threshold, margin=args.auto_editor_margin, keep_original=args.keep_original)
        if args.normalize:
            _pp.normalize_ffmpeg_in_place(out_path, method=args.normalize)
    except Exception:
        pass
    try:
        from settings_io import save_settings as _save
        params = {
            'voice': getattr(args, 'voice', None),
            'voice_dir': str(args.voice_dir),
            'seed': getattr(args, 'seed', None),
            'run_index': getattr(args, 'run_index', 0),
            'device': device,
            'cfg_weight': args.cfg_weight,
            'exaggeration': args.exaggeration,
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'validate': getattr(args, 'validate', False),
            'denoise': getattr(args, 'denoise', False),
            'auto_editor': getattr(args, 'auto_editor', False),
            'normalize': getattr(args, 'normalize', None),
            'text_input': ''
        }
        _save(out_path, [out_path], params)
    except Exception:
        pass
    print(f"Saved output to: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
