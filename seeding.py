import inspect

def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int, attempt_idx: int) -> int:
    import numpy as np
    mix = (
        np.uint64(base_seed) * np.uint64(1000003)
        + np.uint64(chunk_idx) * np.uint64(10007)
        + np.uint64(cand_idx) * np.uint64(10009)
        + np.uint64(attempt_idx) * np.uint64(101)
    )
    out = int(mix & np.uint64(0xFFFFFFFF))
    return out or 1

def generate_with_seed(tts, text: str, seed: int, **kw):
    """Call tts.generate with deterministic seeding where supported.

    - If generate() accepts a `generator` kwarg (common in diffusion APIs), build a per-call
      torch.Generator and seed it (skip on MPS which may not support torch.Generator correctly).
    - Otherwise, fork RNG locally and seed torch (and cuda if applicable) to avoid polluting globals.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        # No torch available: just call through
        return tts.generate(text, **kw)

    import torch
    params = inspect.signature(tts.generate).parameters
    device = kw.get("device")
    if "generator" in params and device != "mps":
        gen_device = "cuda" if device == "cuda" else "cpu"
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(seed & 0xFFFFFFFFFFFFFFFF)
        return tts.generate(text, generator=gen, **kw)

    devices = [torch.cuda.current_device()] if device == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return tts.generate(text, **kw)
