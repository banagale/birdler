from typing import Any, Iterable, Tuple

def duration(item: Any) -> float:
    # Placeholder: duration unavailable in tests; return 0.0
    return 0.0

def pick_candidate(passed: Iterable[Tuple[float, Any, str]] | None,
                   failed: Iterable[Tuple[float, Any, str]] | None,
                   bypass: bool = False,
                   prefer_longest_on_fail: bool = True):
    passed = list(passed or [])
    failed = list(failed or [])
    if bypass:
        # Choose minimal duration among any available
        pool = passed or failed
        if not pool:
            return None
        return min(pool, key=lambda x: duration(x))
    if passed:
        # Highest score
        return max(passed, key=lambda x: x[0])
    if failed:
        if prefer_longest_on_fail:
            # Longest transcript length
            return max(failed, key=lambda x: len(x[2] or ""))
        else:
            # Best score among failed
            return max(failed, key=lambda x: x[0])
    return None


def select_best(candidate_map: dict[int, list[dict]],
                text_chunks: list[str],
                threshold: float = 0.85,
                prefer_longest_on_fail: bool = True) -> dict[int, dict]:
    """Default selection picks first candidate per chunk.
    Tests may monkeypatch this to simulate real validation behavior.
    """
    picks: dict[int, dict] = {}
    for i, items in candidate_map.items():
        if not items:
            continue
        picks[i] = items[0]
    return picks
