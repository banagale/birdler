from selection import pick_candidate

def test_pick_highest_score():
    passed = [(0.7, {'p':1}, 'a'), (0.9, {'p':2}, 'b')]
    failed = []
    best = pick_candidate(passed, failed)
    assert best[0] == 0.9

def test_pick_longest_on_fail():
    passed = []
    failed = [(0.6, {'p':1}, 'short'), (0.5, {'p':2}, 'this is longer text')]
    best = pick_candidate(passed, failed, bypass=False, prefer_longest_on_fail=True)
    assert best[2] == 'this is longer text'
