import re

def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter that keeps delimiters (.,!?) and
    gracefully handles delimiter-less text.
    """
    text = (text or "").strip()
    if not text:
        return []
    pieces = re.split(r"([.!?])", text)
    if len(pieces) == 1:
        return [text]
    out = []
    for i in range(0, len(pieces) - 1, 2):
        sent = (pieces[i] + pieces[i + 1]).strip()
        if sent:
            out.append(sent)
    if len(pieces) % 2 == 1 and pieces[-1].strip():
        out.append(pieces[-1].strip())
    return out

def greedy_pack(sentences: list[str], soft_max: int = 300, hard_max: int = 360) -> list[str]:
    """Greedy pack sentences up to soft_max; enforce hard_max by whitespace split."""
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s_norm = re.sub(r"\s+", " ", s.strip())
        candidate = (buf + " " + s_norm).strip() if buf else s_norm
        if len(candidate) <= soft_max:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = s_norm
        else:
            buf = s_norm
        while len(buf) > hard_max:
            cut = buf.rfind(" ", 0, hard_max)
            cut = cut if cut != -1 else hard_max
            chunks.append(buf[:cut].strip())
            buf = buf[cut:].strip()
    if buf:
        chunks.append(buf)
    return chunks

def split_long_sentence(s: str, max_len: int = 300, seps: list[str] | None = None) -> list[str]:
    seps = seps or [";", ":", "-", ",", " "]
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= max_len:
        return [s]
    if not seps:
        return [s[i : i + max_len].strip() for i in range(0, len(s), max_len)]
    sep = seps[0]
    parts = s.split(sep)
    if len(parts) == 1:
        return split_long_sentence(s, max_len, seps=seps[1:])
    out: list[str] = []
    cur = parts[0].strip()
    for part in parts[1:]:
        cand = (cur + sep + part).strip()
        if len(cand) > max_len:
            out.extend(split_long_sentence(cur, max_len, seps=seps[1:]))
            cur = part.strip()
        else:
            cur = cand
    out.extend(split_long_sentence(cur, max_len, seps=seps[1:]) if len(cur) > max_len else [cur])
    return [x for x in out if x]

def enforce_min_chunk_length(chunks: list[str], min_len: int = 20, max_len: int = 300) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(chunks)
    while i < n:
        cur = (chunks[i] or "").strip()
        if len(cur) >= min_len or i == n - 1:
            out.append(cur)
            i += 1
        else:
            if i + 1 < n:
                merged = f"{cur} {chunks[i+1]}".strip()
                if len(merged) <= max_len:
                    out.append(merged)
                    i += 2
                else:
                    out.append(cur)
                    i += 1
            else:
                out.append(cur)
                i += 1
    return [c for c in out if c]

def get_chunks(text: str, soft_max: int = 300, hard_max: int = 360, min_len: int = 20) -> list[str]:
    sents = split_sentences(text)
    refined: list[str] = []
    for s in sents:
        if len(s) > hard_max:
            refined.extend(split_long_sentence(s, max_len=hard_max))
        else:
            refined.append(s)
    packed = greedy_pack(refined, soft_max=soft_max, hard_max=hard_max)
    return enforce_min_chunk_length(packed, min_len=min_len, max_len=soft_max)
