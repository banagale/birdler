from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from seeding import derive_seed, generate_with_seed

def gen_chunk_job(
    tts: Any,
    chunk_text: str,
    chunk_idx: int,
    base_seed: int,
    cand_idx: int,
    attempt_idx: int,
    gen_kwargs: dict | None,
):
    gen_kwargs = gen_kwargs or {}
    seed = derive_seed(base_seed, chunk_idx, cand_idx, attempt_idx)
    wav = generate_with_seed(tts, chunk_text, seed, **gen_kwargs)
    return chunk_idx, {"seed": seed, "cand_idx": cand_idx, "attempt": attempt_idx, "wav": wav}

def generate_chunks_parallel(
    tts: Any,
    chunks: list[str],
    base_seed: int,
    workers: int = 4,
    n_cands: int = 1,
    attempts: int = 1,
    gen_kwargs: dict | None = None,
) -> dict[int, list[dict]]:
    gen_kwargs = gen_kwargs or {}
    results: dict[int, list[dict]] = {i: [] for i in range(len(chunks))}
    if not chunks:
        return results

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = []
        for i, text in enumerate(chunks):
            for c in range(max(1, n_cands)):
                for a in range(max(1, attempts)):
                    futs.append(
                        ex.submit(gen_chunk_job, tts, text, i, base_seed, c, a, gen_kwargs)
                    )
        for fut in as_completed(futs):
            idx, item = fut.result()
            results[idx].append(item)
    return results
