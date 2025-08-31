import time
import types

import gen_parallel
from seeding import derive_seed

def test_parallel_ordering_and_counts(monkeypatch):
    # Monkeypatch generate_with_seed to simulate delays by chunk index
    def fake_gws(tts, text, seed, **kw):
        delay = kw.get("_delay", 0)
        if delay:
            time.sleep(delay)
        return (text, seed)

    monkeypatch.setattr(gen_parallel, "generate_with_seed", fake_gws)

    class TTS:
        def generate(self, text, **kw):
            return (text, kw.get("seed"))

    chunks = ["one", "two", "three"]
    base_seed = 42
    results = gen_parallel.generate_chunks_parallel(
        TTS(), chunks, base_seed, workers=2, n_cands=1, attempts=1, gen_kwargs={"_delay": 0.01}
    )
    assert sorted(results.keys()) == [0, 1, 2]
    assert all(len(v) == 1 for v in results.values())
    for i in range(len(chunks)):
        assert results[i][0]["seed"] == derive_seed(base_seed, i, 0, 0)

def test_parallel_seed_grid(monkeypatch):
    monkeypatch.setattr(gen_parallel, "generate_with_seed", lambda tts, text, seed, **kw: seed)
    class TTS:
        def generate(self, text, **kw):
            return 0
    chunks = ["a", "b"]
    base_seed = 123
    results = gen_parallel.generate_chunks_parallel(TTS(), chunks, base_seed, workers=2, n_cands=2, attempts=2)
    for i in range(len(chunks)):
        assert len(results[i]) == 4
    for i in range(len(chunks)):
        seeds = {item["seed"] for item in results[i]}
        assert len(seeds) == 4
