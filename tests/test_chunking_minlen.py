from chunking import get_chunks

def test_min_length_merge_simple():
    text = "A. B. C. This is a longer sentence that should remain."
    chunks = get_chunks(text, soft_max=60, hard_max=80, min_len=20)
    assert chunks
    for c in chunks[:-1]:
        assert len(c) >= 20
