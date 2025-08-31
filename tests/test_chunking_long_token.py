from birdler import _greedy_sentence_pack


def test_long_token_forced_split():
    s = ["A" * 1000 + "."]
    chunks = _greedy_sentence_pack(s, max_chars=280, hard_max=360)
    assert chunks
    assert all(len(c) <= 360 for c in chunks)

