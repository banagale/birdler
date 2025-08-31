from typing import Callable, Any
from seeding import derive_seed, generate_with_seed

def retry_failed_chunks(tts: Any,
                        chunks: list[str],
                        base_seed: int,
                        prev_map: dict[int, list[dict]],
                        max_attempts: int,
                        n_cands: int,
                        gen_kwargs: dict | None,
                        validate_fn: Callable[[dict[int, list[dict]], list[str]], dict[int, tuple[list, list]]]):
    gen_kwargs = gen_kwargs or {}
    attempts = {i: 1 for i in range(len(chunks))}
    # Determine chunks needing retry: those with empty lists in prev_map
    need = [i for i in range(len(chunks)) if not prev_map.get(i)]
    while need:
        round_map: dict[int, list[dict]] = {i: [] for i in need}
        for i in need:
            for c in range(n_cands):
                seed = derive_seed(base_seed, i, c, attempts[i])
                wav = generate_with_seed(tts, chunks[i], seed, **gen_kwargs)
                round_map[i].append({"seed": seed, "cand_idx": c, "attempt": attempts[i], "wav": wav})
        # Validate new candidates
        result = validate_fn(round_map, chunks)
        # Append passing candidates to prev_map
        for i in need:
            passed, failed = result.get(i, ([], []))
            # Items in passed are tuples (score, item, transcript) in our stub
            for score, it, txt in passed:
                prev_map.setdefault(i, []).append(it)
        # Update attempts and recompute need
        attempts = {i: attempts[i] + 1 for i in need if attempts[i] < max_attempts}
        need = [i for i in need if attempts.get(i, max_attempts) < max_attempts and not prev_map.get(i)]
    return prev_map
