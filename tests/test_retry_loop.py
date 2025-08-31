from retry import retry_failed_chunks

def test_retry_converges_on_second_attempt(monkeypatch):
    # Fake TTS; wave returned is irrelevant
    class TTS:
        def generate(self, text, **kw):
            return object()
    chunks = ['alpha']
    base_seed = 123
    prev_map = {0: []}
    # validate_fn fails first time (attempt=1), passes second (attempt=2)
    calls = {'n': 0}
    def validate_fn(candidate_map, chunks_):
        calls['n'] += 1
        out = {}
        for i, items in candidate_map.items():
            if calls['n'] == 1:
                out[i] = ([], [(0.0, None, '')])
            else:
                out[i] = ([(1.0, items[0], chunks_[i])], [])
        return out
    out_map = retry_failed_chunks(TTS(), chunks, base_seed, prev_map, max_attempts=3, n_cands=1, gen_kwargs={}, validate_fn=validate_fn)
    assert 0 in out_map and len(out_map[0]) >= 1
