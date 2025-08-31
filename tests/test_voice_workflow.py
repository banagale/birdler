import sys
import types
import pickle
from pathlib import Path

import birdler


class Args:
    def __init__(self, voice=None, voice_dir=None, audio_sample=None):
        self.voice = voice
        self.voice_dir = voice_dir
        self.audio_sample = audio_sample


def make_torch_stub():
    mod = types.ModuleType("torch")

    def save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    mod.save = save
    mod.load = load
    return mod


def test_prepare_voice_bootstrap(tmp_path):
    voice_dir = tmp_path / "voices"
    sample = tmp_path / "ref.wav"
    sample.write_bytes(b"RIFF....WAVE")

    args = Args(voice="alice", voice_dir=voice_dir, audio_sample=sample)
    vp = birdler.prepare_voice(args)

    assert vp is not None
    assert vp["root"].exists()
    assert vp["samples"].exists()
    assert vp["embedding"].exists()
    assert vp["ref_wav"].exists()


def test_get_or_build_embedding_cache(tmp_path, monkeypatch):
    # Inject stubs before calling function
    torch_stub = make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    # Provide a minimal chatterbox.tts.Conditionals API
    tts_mod = types.ModuleType('chatterbox.tts')
    class DummyCond:
        def __init__(self, data=None): self.data = data or {'ok': True}
        def to(self, device): return self
        def save(self, fpath):
            import pickle
            with open(fpath, 'wb') as f: pickle.dump(self.data, f)
        @classmethod
        def load(cls, fpath, map_location='cpu'):
            import pickle
            with open(fpath, 'rb') as f: data = pickle.load(f)
            return cls(data)
    tts_mod.Conditionals = DummyCond
    chat_mod = types.ModuleType('chatterbox')
    chat_mod.tts = tts_mod
    monkeypatch.setitem(sys.modules, 'chatterbox', chat_mod)
    monkeypatch.setitem(sys.modules, 'chatterbox.tts', tts_mod)

    voice_dir = tmp_path / "voices"
    vp = birdler._voice_paths("bob", voice_dir)
    vp["samples"].mkdir(parents=True)
    vp["embedding"].mkdir(parents=True)
    # reference wav stub
    vp["ref_wav"].write_bytes(b"RIFF....WAVE")

    class FakeTTS:
        def __init__(self):
            self.called = 0
            self.conds = None
        def prepare_conditionals(self, path, exaggeration=0.5):
            self.called += 1
            # set a dummy conds instance
            self.conds = tts_mod.Conditionals({'cond': 'ok', 'path': Path(path).name})

    tts = FakeTTS()
    # Build and cache
    cond1 = birdler.get_or_build_embedding(tts, vp)
    assert cond1 and getattr(cond1, 'data', {}).get("cond") == "ok"
    assert vp["emb_path"].exists()
    assert tts.called == 1

    # Second call should load from cache without calling TTS
    tts2 = FakeTTS()
    cond2 = birdler.get_or_build_embedding(tts2, vp)
    assert getattr(cond2, 'data', None) == getattr(cond1, 'data', None)
    assert tts2.called == 0
