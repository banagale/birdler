import os
import subprocess
from pathlib import Path

def has_cli(name: str) -> bool:
    from shutil import which
    return which(name) is not None

def normalize_ffmpeg_in_place(wav_path: Path, method: str = "ebu", I: float = -24, TP: float = -2, LRA: float = 7) -> None:
    wav_path = Path(wav_path)
    tmp = wav_path.with_name(wav_path.stem + "_norm.wav")
    if method == "ebu":
        af = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
    elif method == "peak":
        af = "alimiter=limit=-2dB"
    else:
        raise ValueError("bad normalize method")
    subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-af", af, str(tmp)], check=True)
    os.replace(tmp, wav_path)

def run_auto_editor_in_place(wav_path: Path, threshold: float = 0.06, margin: float = 0.2, keep_original: bool = False) -> None:
    wav_path = Path(wav_path)
    inp = wav_path
    if keep_original:
        backup = wav_path.with_name(wav_path.stem + "_original.wav")
        os.rename(wav_path, backup)
        inp = backup
    out = wav_path.with_name(wav_path.stem + "_cleaned.wav")
    subprocess.run([
            "auto-editor",
            "--edit", f"audio:threshold={threshold}",
            "--margin", f"{margin}s",
            "--export", "audio",
            str(inp), "-o", str(out),
        ], check=True)
    os.replace(out, wav_path)

def denoise_in_place(wav_path: Path) -> bool:
    """Attempt RNNoise denoise via `denoise` CLI; return True on success.
    Minimal flow: convert to 48k mono s16, run `denoise`, then replace."""
    wav_path = Path(wav_path)
    tmp_48k = wav_path.with_name(wav_path.stem + "_48kmono.wav")
    tmp_dn = wav_path.with_name(wav_path.stem + "_dn.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(wav_path), "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(tmp_48k)
        ], check=True)
        if has_cli("denoise"):
            r = subprocess.run(["denoise", str(tmp_48k), str(tmp_dn)], check=False)
            ok = (r.returncode == 0) and tmp_dn.exists() and tmp_dn.stat().st_size > 0
        else:
            ok = False
        if ok:
            os.replace(tmp_dn, wav_path)
        return ok
    finally:
        for p in (tmp_48k, tmp_dn):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
