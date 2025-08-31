from typing import Any, Tuple

def load_whisper(backend: str, model_name: str, device: str):
    # Placeholder loader; tests monkeypatch as needed
    return object()

def transcribe_and_score(path: str, target_text: str, whisper_model) -> Tuple[float, str]:
    # Placeholder; score 1.0 and echo text
    return 1.0, target_text

def validate_candidates_map(candidate_map: dict[int, list[dict]],
                            text_chunks: list[str],
                            whisper_model: Any = None,
                            threshold: float = 0.85) -> dict[int, Tuple[list, list]]:
    """Return per-chunk (passed, failed) lists.
    Default marks all as passed with score 1.0 and transcript=chunk text.
    """
    out: dict[int, tuple[list, list]] = {}
    for i, items in candidate_map.items():
        txt = text_chunks[i] if i < len(text_chunks) else ""
        passed = [(1.0, it, txt) for it in items]
        out[i] = (passed, [])
    return out
