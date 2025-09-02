#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
from pathlib import Path


def gen_tone(path: Path, freq: float = 440.0, dur_s: float = 0.6, sr: int = 22050, amp: float = 0.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * dur_s)
    # 16-bit PCM mono
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            t = i / sr
            x = math.sin(2 * math.pi * freq * t) * amp
            s = max(-1.0, min(1.0, x))
            wf.writeframes(struct.pack("<h", int(s * 32767)))
    return path


def play(path: Path) -> tuple[bool, str]:
    # Try common CLI players in order
    candidates = [
        ("afplay", ["afplay", str(path)]),
        ("ffplay", ["ffplay", "-v", "quiet", "-nodisp", "-autoexit", str(path)]),
        ("aplay", ["aplay", str(path)]),
    ]
    for name, cmd in candidates:
        if shutil.which(name):
            try:
                r = subprocess.run(cmd, check=False)
                return (r.returncode == 0, name)
            except Exception:
                return (False, name)
    return (False, "none-found")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a short test tone and (optionally) play it")
    p.add_argument("--out", type=Path, default=Path("generated-audio/codex-audio-smoke.wav"))
    p.add_argument("--freq", type=float, default=440.0)
    p.add_argument("--dur", type=float, default=0.6)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--amp", type=float, default=0.2)
    p.add_argument("--play", action="store_true")
    args = p.parse_args()

    wav = gen_tone(args.out, args.freq, args.dur, args.sr, args.amp)
    print(f"[audio-smoke] Wrote {wav}")
    if args.play:
        ok, used = play(wav)
        if used == "none-found":
            print("[audio-smoke] No CLI audio player found (tried afplay, ffplay, aplay)")
            return 2
        print(f"[audio-smoke] Player: {used} -> {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

