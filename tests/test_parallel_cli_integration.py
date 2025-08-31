import sys
import types
from pathlib import Path

import birdler

def make_stubs(monkeypatch):
    torch = types.SimpleNamespace()
    class _CUDA:
        @staticmethod
        def is_available():
            return False
    class _MPS:
        @staticmethod
        def is_available():
            return False
    torch.backends = types.SimpleNamespace(mps=_MPS())
    torch.cuda = _CUDA()
    torch.Tensor = object
    torch.is_floating_point = lambda x: True
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, 'torch', torch)

    ta = types.ModuleType('torchaudio')
    ta.save = lambda path, data, sr: Path(path).write_bytes(b'WAV')
    monkeypatch.setitem(sys.modules, 'torchaudio', ta)

    chat = types.ModuleType('chatterbox')
    chat_tts = types.ModuleType('chatterbox.tts')
    class FakeTTS:
        sr = 22050
        def prepare_conditionals(self, path, exaggeration=0.5):
            self.conds = object()
        class _Wave:
            def dim(self): return 1
            def unsqueeze(self, ax): return self
            def float(self): return self
            def abs(self): return self
            def mean(self, dim=0): return self
            def __gt__(self, other): return self
            def nonzero(self, as_tuple=False): return self
            def squeeze(self): return self
            def item(self): return 0
        def generate(self, text, **kw):
            return self._Wave()
        @classmethod
        def from_pretrained(cls, device=None):
            return cls()
    chat_tts.ChatterboxTTS = FakeTTS
    monkeypatch.setitem(sys.modules, 'chatterbox', chat)
    monkeypatch.setitem(sys.modules, 'chatterbox.tts', chat_tts)


def test_parallel_cli_builds_output(monkeypatch, tmp_path):
    make_stubs(monkeypatch)
    import gen_parallel
    def fake_parallel(tts, chunks, base_seed, workers, n_cands, attempts, gen_kwargs=None):
        class W:
            def dim(self): return 1
            def unsqueeze(self, ax): return self
            def float(self): return self
            def abs(self): return self
            def mean(self, dim=0): return self
            def __gt__(self, other): return self
            def nonzero(self, as_tuple=False): return self
            def squeeze(self): return self
            def item(self): return 0
        return {i: [{"seed": 1, "cand_idx": 0, "attempt": 0, "wav": W()}] for i in range(len(chunks))}
    monkeypatch.setattr(gen_parallel, 'generate_chunks_parallel', fake_parallel)
    class FakeFinal:
        def contiguous(self): return self
        def clamp(self, a, b): return self
        def to(self, dtype=None): return self
    import birdler as _b
    monkeypatch.setattr(_b, 'crossfade_concat', lambda chunks, fade_samples=0: FakeFinal(), raising=False)

    out_dir = tmp_path / 'out'
    voices_dir = tmp_path / 'voices'
    ref = voices_dir / 'v' / 'samples' / 'reference.wav'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b'RIFF..WAVE')
    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--voice', 'v', '--voice-dir', str(voices_dir),
        '--text', 'hello world', '--output-dir', str(out_dir),
        '--workers', '2', '--n-candidates', '2', '--max-attempts', '2'
    ])
    rc = birdler.main()
    assert rc == 0
    assert any(p.suffix == '.wav' for p in out_dir.iterdir())
