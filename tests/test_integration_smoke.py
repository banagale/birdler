import sys
import types
from pathlib import Path

import birdler


def make_torch_stub():
    torch = types.SimpleNamespace()

    class Backends:
        class _MPS:
            @staticmethod
            def is_available():
                return False

    torch.backends = types.SimpleNamespace(mps=Backends._MPS())
    class _CUDA:
        @staticmethod
        def is_available():
            return False
    torch.cuda = _CUDA()
    torch.Tensor = object
    torch.float32 = object()
    return torch


def test_end_to_end_smoke(tmp_path, monkeypatch):
    out_dir = tmp_path / "generated-audio"
    voices_dir = tmp_path / "voices"
    sample = tmp_path / "ref.wav"
    sample.write_bytes(b"RIFF....WAVE")

    # Stub heavy deps
    monkeypatch.setitem(sys.modules, "torch", make_torch_stub())

    toro = types.ModuleType("torchaudio")

    def save(path, data, sr):
        Path(path).write_bytes(b"WAV")

    toro.save = save
    monkeypatch.setitem(sys.modules, "torchaudio", toro)

    # Stub chatterbox TTS
    chat = types.ModuleType("chatterbox")

    class FakeTTS:
        sr = 22050

        def get_audio_conditioning(self, path):
            # Return a simple serializable object
            return {"ok": True, "path": Path(path).name}

        def generate(self, text, **kwargs):
            # Return a lightweight stand-in object; downstream we bypass heavy ops
            return object()

        @classmethod
        def from_pretrained(cls, device=None):
            return cls()

    chat.ChatterboxTTS = FakeTTS
    monkeypatch.setitem(sys.modules, "chatterbox", chat)

    # Bypass heavy audio ops in pipeline
    monkeypatch.setattr(birdler, "trim_silence", lambda w, thresh=1e-3: w, raising=False)

    class FakeFinal:
        def contiguous(self):
            return self

        def clamp(self, a, b):
            return self

        def to(self, dtype=None):
            return self

    monkeypatch.setattr(birdler, "crossfade_concat", lambda chunks, fade_samples=2048: FakeFinal(), raising=False)

    # First run: bootstrap and generate
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "birdler.py",
            "--voice",
            "smokey",
            "--voice-dir",
            str(voices_dir),
            "--audio-sample",
            str(sample),
            "--text",
            "hello world",
            "--output-dir",
            str(out_dir),
        ],
    )
    rc = birdler.main()
    assert rc == 0
    wavs = list(out_dir.glob("*.wav"))
    assert wavs, "expected an output wav file"

    # Second run: reuse cache without audio-sample
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "birdler.py",
            "--voice",
            "smokey",
            "--voice-dir",
            str(voices_dir),
            "--text",
            "another line",
            "--output-dir",
            str(out_dir),
        ],
    )
    rc2 = birdler.main()
    assert rc2 == 0
