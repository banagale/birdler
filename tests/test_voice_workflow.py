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
    # Inject torch stub before calling function
    torch_stub = make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    voice_dir = tmp_path / "voices"
    vp = birdler._voice_paths("bob", voice_dir)
    vp["samples"].mkdir(parents=True)
    vp["embedding"].mkdir(parents=True)
    # reference wav stub
    vp["ref_wav"].write_bytes(b"RIFF....WAVE")

    class FakeTTS:
        def __init__(self):
            self.called = 0

        def get_audio_conditioning(self, path):
            self.called += 1
            return {"cond": "ok", "path": Path(path).name}

    tts = FakeTTS()
    # Build and cache
    cond1 = birdler.get_or_build_embedding(tts, vp)
    assert cond1 and cond1.get("cond") == "ok"
    assert vp["emb_path"].exists()
    assert tts.called == 1

    # Second call should load from cache without calling TTS
    tts2 = FakeTTS()
    cond2 = birdler.get_or_build_embedding(tts2, vp)
    assert cond2 == cond1
    assert tts2.called == 0

