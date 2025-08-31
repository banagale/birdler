from chunking import split_long_sentence

def test_split_long_sentence_priority():
    s = 'This_is_a_super_long-sentence:with;many,separators and words that exceed the limit significantly.'
    parts = split_long_sentence(s, max_len=30)
    assert parts
    assert all(len(p) <= 30 for p in parts)
