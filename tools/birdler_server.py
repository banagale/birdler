#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
import warnings
import logging
import os
from pathlib import Path
from typing import Optional, Dict
import contextlib


def parse_args():
    p = argparse.ArgumentParser(description="Persistent Birdler TTS server (stdin/stdout NDJSON)")
    p.add_argument("--voice-dir", type=Path, default=Path("voices"))
    p.add_argument("--default-voice", type=str, help="Default voice to use if request omits it")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--no-crossfade", action="store_true", help="Disable crossfade when concatenating chunks")
    p.add_argument("--verbose", action="store_true", help="Print debug info to stderr")
    p.add_argument("--strict-device", action="store_true", help="On MPS, disable CPU fallback (may raise if ops unsupported)")
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def setup_quiet_logging(verbose: bool):
    """Reduce noisy third-party warnings; keep our own debug when verbose.
    Also disable progress bars that fight with status tickers."""
    # Always suppress warnings to avoid ticker interference
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    # Disable progress bars from tqdm/HF ecosystems
    os.environ.setdefault("DISABLE_TQDM", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    # Only clamp library logging in non-verbose mode
    if not verbose:
        for name in ("diffusers", "transformers", "accelerate", "torch", "pkg_resources"):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass
    # Set library logging verbosity, if available
    try:
        from transformers.utils import logging as _tlog
        _tlog.set_verbosity_error()
    except Exception:
        pass
    try:
        from diffusers.utils import logging as _dlog
        _dlog.set_verbosity_error()
    except Exception:
        pass


def select_device(preferred: Optional[str]) -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu"
    import torch as _t, sys as _s
    try:
        is_macos = (_s.platform == "darwin")
    except Exception:
        is_macos = False
    if is_macos and getattr(_t.backends, "mps", None) and _t.backends.mps.is_available():
        return "mps"
    if _t.cuda.is_available():
        return "cuda"
    if getattr(_t.backends, "mps", None) and _t.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_determinism(device: str):
    try:
        import torch
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


@dataclass
class VoiceState:
    name: str
    ref_wav: Path
    emb_path: Path


def voice_paths(voice: str, root: Path) -> VoiceState:
    base = root / voice
    samples = base / "samples"
    embedding = base / "embedding"
    return VoiceState(
        name=voice,
        ref_wav=samples / "reference.wav",
        emb_path=embedding / "cond.pt",
    )


def ensure_voice_dirs(vs: VoiceState):
    vs.ref_wav.parent.mkdir(parents=True, exist_ok=True)
    vs.emb_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    setup_quiet_logging(verbose=args.verbose)
    # Enforce strict device behavior for MPS if requested
    if args.strict_device and (args.device in (None, "auto", "mps")):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    device = select_device(args.device)

    # Verbose banner and environment report
    if args.verbose:
        try:
            import torch as _t
            torch_ver = getattr(_t, "__version__", "unknown")
            mps_avail = getattr(_t.backends, "mps", None) and _t.backends.mps.is_available()
            mps_built = getattr(_t.backends, "mps", None) and getattr(_t.backends.mps, "is_built", lambda: False)()
            cuda_avail = _t.cuda.is_available()
        except Exception:
            torch_ver = "unavailable"
            mps_avail = False
            mps_built = False
            cuda_avail = False
        try:
            from importlib.metadata import version as _ver
            cbx_ver = _ver("chatterbox-tts")
        except Exception:
            cbx_ver = "unknown"
        banner = (
            "\n+--------------------------- BIRDLER SERVER ---------------------------\n"
            f" device={device} | torch={torch_ver} | cbx={cbx_ver}\n"
            f" mps_avail={bool(mps_avail)} | mps_built={bool(mps_built)} | cuda_avail={bool(cuda_avail)}\n"
            f" PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','')}\n"
            f" voice_dir={args.voice_dir}\n"
            "----------------------------------------------------------------\n"
        )
        try:
            sys.stderr.write(banner)
            sys.stderr.flush()
        except Exception:
            pass
    if args.deterministic:
        set_determinism(device)

    # Lazy imports to avoid heavy cost until needed
    import importlib
    birdler = importlib.import_module("birdler")
    # Model
    from chatterbox.tts import ChatterboxTTS
    tts = ChatterboxTTS.from_pretrained(device=device)
    # Best-effort: ensure model is on the requested device
    try:
        if hasattr(tts, "to"):
            tts.to(device)
    except Exception:
        pass

    # Voice cache: voice -> prepared (embedding loaded into tts) snapshot
    prepared: Dict[str, bool] = {}

    def prepare_voice(voice: str) -> Optional[VoiceState]:
        vs = voice_paths(voice, args.voice_dir)
        ensure_voice_dirs(vs)
        if not vs.ref_wav.exists() and not vs.emb_path.exists():
            sys.stdout.write(json.dumps({"kind": "error", "error": f"voice '{voice}' missing reference.wav and embedding"}) + "\n")
            sys.stdout.flush()
            return None
        # Load or build conds once per voice
        if not prepared.get(voice):
            vp = {
                "root": args.voice_dir / voice,
                "samples": vs.ref_wav.parent,
                "embedding": vs.emb_path.parent,
                "ref_wav": vs.ref_wav,
                "emb_path": vs.emb_path,
            }
            conds = birdler.get_or_build_embedding(tts, vp, device=device, exaggeration=args.exaggeration)
            if conds is None:
                sys.stdout.write(json.dumps({"kind": "error", "error": f"failed to load/build embedding for '{voice}'"}) + "\n")
                sys.stdout.flush()
                return None
            prepared[voice] = True
            if args.verbose:
                try:
                    sys.stderr.write(f"\n[server] prepared voice='{voice}' ref={vs.ref_wav.exists()} emb={vs.emb_path.exists()}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        return vs

    # I/O loop: read NDJSON requests and write NDJSON responses
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stdout.write(json.dumps({"kind": "error", "error": f"bad json: {e}"}) + "\n")
            sys.stdout.flush()
            continue
        kind = req.get("kind")
        if kind == "shutdown":
            sys.stdout.write(json.dumps({"kind": "result", "ok": True}) + "\n")
            sys.stdout.flush()
            break
        if kind == "warmup":
            voice = req.get("voice") or args.default_voice
            if not voice:
                sys.stdout.write(json.dumps({"kind": "result", "ok": False, "error": "missing voice"}) + "\n")
                sys.stdout.flush()
                continue
            vs = prepare_voice(voice)
            if vs is None:
                continue
            try:
                cfg = dict(cfg_weight=args.cfg_weight, exaggeration=args.exaggeration,
                           temperature=args.temperature, repetition_penalty=args.repetition_penalty)
                # tiny throwaway
                txt = "hi"
                if getattr(tts, 'conds', None) is not None:
                    _ = tts.generate(txt, **cfg)
                else:
                    _ = tts.generate(txt, audio_prompt_path=str(vs.ref_wav), **cfg)
                sys.stdout.write(json.dumps({"kind": "result", "ok": True}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(json.dumps({"kind": "result", "ok": False, "error": f"warmup failed: {e}"}) + "\n")
                sys.stdout.flush()
            continue
        if kind != "speak":
            sys.stdout.write(json.dumps({"kind": "error", "error": "unsupported kind"}) + "\n")
            sys.stdout.flush()
            continue
        text = (req.get("text") or "").strip()
        voice = req.get("voice") or args.default_voice
        out_dir = Path(req.get("out_dir") or "generated-audio")
        no_cross = bool(req.get("no_crossfade", args.no_crossfade))
        if not text:
            sys.stdout.write(json.dumps({"kind": "result", "ok": False, "error": "empty text"}) + "\n")
            sys.stdout.flush()
            continue
        if not voice:
            sys.stdout.write(json.dumps({"kind": "result", "ok": False, "error": "missing voice"}) + "\n")
            sys.stdout.flush()
            continue
        vs = prepare_voice(voice)
        if vs is None:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate
        cfg = dict(cfg_weight=args.cfg_weight, exaggeration=args.exaggeration,
                   temperature=args.temperature, repetition_penalty=args.repetition_penalty)
        # Some libraries (HF/transformers) may write progress to stdout. Our
        # stdout is reserved for NDJSON protocol, so we must suppress stdout
        # at the FD level during generation.
        @contextlib.contextmanager
        def _suppress_stdout_fd():
            try:
                fd = sys.stdout.fileno()
            except Exception:
                # Fallback: Python-level redirection
                import io
                old = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    yield
                finally:
                    sys.stdout = old
                return
            import os as _os
            devnull = _os.open(_os.devnull, _os.O_WRONLY)
            saved = _os.dup(fd)
            try:
                _os.dup2(devnull, fd)
                yield
            finally:
                try:
                    _os.dup2(saved, fd)
                except Exception:
                    pass
                try:
                    _os.close(saved)
                except Exception:
                    pass
                try:
                    _os.close(devnull)
                except Exception:
                    pass

        try:
            with _suppress_stdout_fd():
                if getattr(tts, 'conds', None) is not None:
                    wav = tts.generate(text, **cfg)
                else:
                    wav = tts.generate(text, audio_prompt_path=str(vs.ref_wav), **cfg)
        except Exception as e:
            sys.stdout.write(json.dumps({"kind": "result", "ok": False, "error": f"generate failed: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        # Post
        import torch
        import torchaudio
        if not hasattr(wav, "dim"):
            import numpy as np
            wav = torch.from_numpy(wav.astype("float32")) if isinstance(wav, np.ndarray) else wav
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.clamp(-1.0, 1.0).to(dtype=torch.float32)

        # File name
        import secrets
        stamp = int(time.time())
        slug = "-".join(text.lower().split()[:4]) or "utt"
        suffix = secrets.token_hex(2)
        out_path = out_dir / f"{voice}_{slug}_{stamp}_{suffix}.wav"

        sr = getattr(tts, "sr", 22050)
        torchaudio.save(str(out_path), wav, sr)

        sys.stdout.write(json.dumps({"kind": "result", "ok": True, "path": str(out_path)}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
