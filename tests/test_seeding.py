from seeding import derive_seed

def test_derive_seed_grid_uniqueness():
    base = 12345
    seen = set()
    for i in range(3):
        for c in range(2):
            for a in range(2):
                s = derive_seed(base, i, c, a)
                assert s != 0
                seen.add(s)
    assert len(seen) == 3 * 2 * 2
