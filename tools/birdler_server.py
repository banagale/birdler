#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Persistent Birdler TTS server (stdin/stdout NDJSON)")
    p.add_argument("--voice-dir", type=Path, default=Path("voices"))
    p.add_argument("--default-voice", type=str, help="Default voice to use if request omits it")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--no-crossfade", action="store_true", help="Disable crossfade when concatenating chunks")
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


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
    device = select_device(args.device)
    if args.deterministic:
        set_determinism(device)

    # Lazy imports to avoid heavy cost until needed
    import importlib
    birdler = importlib.import_module("birdler")
    # Model
    from chatterbox.tts import ChatterboxTTS
    tts = ChatterboxTTS.from_pretrained(device=device)

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
        if req.get("kind") != "speak":
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
        try:
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
        stamp = int(time.time())
        slug = "-".join(text.lower().split()[:4]) or "utt"
        out_path = out_dir / f"{voice}_{slug}_{stamp}.wav"

        sr = getattr(tts, "sr", 22050)
        torchaudio.save(str(out_path), wav, sr)

        sys.stdout.write(json.dumps({"kind": "result", "ok": True, "path": str(out_path)}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

