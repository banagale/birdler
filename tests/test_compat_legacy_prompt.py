import sys
from pathlib import Path
import types

import birdler


def make_stubs(monkeypatch):
    # torch stub (minimal)
    torch = types.SimpleNamespace()
    class Back:
        class _MPS:
            @staticmethod
            def is_available():
                return False
    torch.backends = types.SimpleNamespace(mps=Back._MPS())
    class _CUDA:
        @staticmethod
        def is_available():
            return False
    torch.cuda = _CUDA()
    torch.Tensor = object
    torch.is_floating_point = lambda x: True
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, 'torch', torch)

    # torchaudio stub
    ta = types.ModuleType('torchaudio')
    ta.save = lambda path, data, sr: Path(path).write_bytes(b'WAV')
    monkeypatch.setitem(sys.modules, 'torchaudio', ta)

    # chatterbox stub without get_audio_conditioning
    chat = types.ModuleType('chatterbox')
    class FakeTTS:
        sr = 22050
        class _Wave:
            def dim(self): return 1
            def unsqueeze(self, ax): return self
            def float(self): return self
            def abs(self): return self
            def mean(self, dim=0): return self
            def __getitem__(self, idx): return self
            def __gt__(self, other): return self
            def nonzero(self, as_tuple=False): return self
            def squeeze(self): return self
            def item(self): return 0
        def generate(self, text, **kw):
            return self._Wave()
        @classmethod
        def from_pretrained(cls, device=None):
            return cls()
    # Patch crossfade_concat to return an object with expected chain methods
    class FakeFinal:
        def contiguous(self): return self
        def clamp(self, a, b): return self
        def to(self, dtype=None): return self
    monkeypatch.setattr(birdler, 'crossfade_concat', lambda chunks, fade_samples=2048: FakeFinal(), raising=False)
    chat.ChatterboxTTS = FakeTTS
    monkeypatch.setitem(sys.modules, 'chatterbox', chat)


def test_generation_with_compat_legacy_prompt(tmp_path, monkeypatch):
    make_stubs(monkeypatch)
    voices_dir = tmp_path / 'voices'
    out_dir = tmp_path / 'out'
    ref = voices_dir / 'tuba' / 'samples' / 'reference.wav'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b'RIFF..WAVE')

    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--voice', 'tuba', '--voice-dir', str(voices_dir),
        '--compat-legacy-prompt', '--text', 'hello', '--output-dir', str(out_dir)
    ])
    rc = birdler.main()
    assert rc == 0
    wavs = list(out_dir.glob('*.wav'))
    assert wavs


def test_bootstrap_build_embedding_fails_in_compat(tmp_path, monkeypatch):
    make_stubs(monkeypatch)
    voices_dir = tmp_path / 'voices'
    ref = voices_dir / 'tuba' / 'samples' / 'reference.wav'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b'RIFF..WAVE')

    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--voice', 'tuba', '--voice-dir', str(voices_dir),
        '--bootstrap-only', '--build-embedding', '--compat-legacy-prompt'
    ])
    rc = birdler.main()
    assert rc == 1
