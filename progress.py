def log_progress(done: int, total: int):
    total = max(1, int(total))
    pct = int(100 * done / total)
    print(f"[PROGRESS] {done}/{total} ({pct}%)")
