import sys
from pathlib import Path
import types

import birdler
import pickle


def make_stubs(monkeypatch):
    # torch stub
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
    def save(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    torch.save = save
    torch.load = load
    monkeypatch.setitem(sys.modules, 'torch', torch)

    # torchaudio stub
    ta = types.ModuleType('torchaudio')
    ta.save = lambda path, data, sr: Path(path).write_bytes(b'WAV')
    monkeypatch.setitem(sys.modules, 'torchaudio', ta)

    # chatterbox stub
    chat = types.ModuleType('chatterbox')
    class FakeTTS:
        sr = 22050
        def get_audio_conditioning(self, path):
            return {'ok': True}
        class _Wave:
            def dim(self): return 1
            def unsqueeze(self, ax): return self
            def float(self): return self
        def generate(self, text, **kw):
            return self._Wave()
        @classmethod
        def from_pretrained(cls, device=None):
            return cls()
    chat.ChatterboxTTS = FakeTTS
    monkeypatch.setitem(sys.modules, 'chatterbox', chat)


def test_youtube_bootstrap_voice(tmp_path, monkeypatch):
    make_stubs(monkeypatch)

    voices_dir = tmp_path / 'voices'
    out_dir = tmp_path / 'generated-audio'
    ref_path = voices_dir / 'neo' / 'samples' / 'reference.wav'

    # Fake yt-dlp presence and run to write reference.wav
    monkeypatch.setattr(birdler, 'which', lambda name: 'yt-dlp')
    def fake_run(cmd, check=True):
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_bytes(b'RIFF..WAVE')
    monkeypatch.setattr(birdler.subprocess, 'run', fake_run)

    # Bootstrap only
    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--voice', 'neo', '--voice-dir', str(voices_dir),
        '--youtube-url', 'http://example.com', '--bootstrap-only',
        '--output-dir', str(out_dir),
    ])
    rc = birdler.main()
    assert rc == 0
    assert ref_path.exists()
    # Embedding should be cached
    emb = voices_dir / 'neo' / 'embedding' / 'cond.pt'
    assert emb.exists()
