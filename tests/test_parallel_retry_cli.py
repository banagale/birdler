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


def test_cli_validation_with_retry(monkeypatch, tmp_path):
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
        return {0: [{"seed": 1, "cand_idx": 0, "attempt": 0, "wav": W()}]}
    monkeypatch.setattr(gen_parallel, 'generate_chunks_parallel', fake_parallel)

    import validate as _val
    calls = {"n": 0}
    def fake_validate_map(candidate_map, text_chunks, whisper_model=None, threshold=0.85):
        calls["n"] += 1
        if calls["n"] == 1:
            return {0: ([], [(0.0, None, '')])}
        items = candidate_map.get(0, [])
        return {0: ([(1.0, items[0], text_chunks[0])], [])}
    monkeypatch.setattr(_val, 'validate_candidates_map', fake_validate_map)

    import retry as _retry
    def fake_retry(tts, chunks, base_seed, prev_map, max_attempts, n_cands, gen_kwargs, validate_fn):
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
        return {0: [{"seed": 2, "cand_idx": 0, "attempt": 1, "wav": W()}]}
    monkeypatch.setattr(_retry, 'retry_failed_chunks', fake_retry)

    class FakeFinal:
        def contiguous(self): return self
        def clamp(self, a, b): return self
        def to(self, dtype=None): return self
    monkeypatch.setattr(birdler, 'crossfade_concat', lambda chunks, fade_samples=0: FakeFinal(), raising=False)

    out_dir = tmp_path / 'out'
    voices_dir = tmp_path / 'voices'
    ref = voices_dir / 'v' / 'samples' / 'reference.wav'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b'RIFF..WAVE')
    monkeypatch.setattr(sys, 'argv', [
        'birdler.py', '--voice', 'v', '--voice-dir', str(voices_dir),
        '--text', 'hello', '--output-dir', str(out_dir),
        '--workers', '2', '--n-candidates', '2', '--max-attempts', '2',
        '--validate', '--validate-threshold', '0.9', '--prefer-longest-on-fail'
    ])
    rc = birdler.main()
    assert rc == 0
    assert any(p.suffix == '.wav' for p in out_dir.iterdir())
