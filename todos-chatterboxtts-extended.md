````markdown
# Birdler ← Chatterbox-TTS-Extended: Migration Shortlist (Implementation Guidelines)

> Each section states the intent/why, then a **Syntax lead** showing the shape of the change (not drop-in code).

---

## 1) Deterministic, Per-Chunk/Per-Candidate Seeding

**Intent & Why**  
Make generations reproducible across chunks, candidates, and retries without polluting global RNG. Extended uses a derived seed grid and (when available) a per-call `torch.Generator`. This stabilizes results, enables debugging, and prevents thread races from altering outputs.

**Syntax lead**
```python
# seeding.py
def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int, attempt_idx: int) -> int:
    mix = (np.uint64(base_seed) * np.uint64(1000003)
           + np.uint64(chunk_idx) * np.uint64(10007)
           + np.uint64(cand_idx) * np.uint64(10009)
           + np.uint64(attempt_idx) * np.uint64(101))
    out = int(mix & np.uint64(0xFFFFFFFF))
    return out or 1

def generate_with_seed(tts, text, seed: int, **kw):
    # Prefer per-call generator if supported (skip on mps)
    if "generator" in inspect.signature(tts.generate).parameters and kw.get("device") != "mps":
        gen = torch.Generator(device="cuda" if kw.get("device") == "cuda" else "cpu")
        gen.manual_seed(seed & 0xFFFFFFFFFFFFFFFF)
        return tts.generate(text, generator=gen, **kw)
    # Fallback: fork RNG locally so globals aren’t polluted
    devices = [torch.cuda.current_device()] if kw.get("device") == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed); 
        if kw.get("device") == "cuda": torch.cuda.manual_seed_all(seed)
        return tts.generate(text, **kw)
````

---

## 2) Robust Sentence Batching + Minimum-Chunk Length

**Intent & Why**
Extended’s batching reduces staccato prosody by: (a) sentence tokenization; (b) greedy packing to a soft limit; (c) post-pass enforcing a minimum chunk length. This yields smoother rhythm and fewer breath/gap artifacts.

**Syntax lead**

```python
# chunking.py
def split_sentences(text: string) -> list[str]: 
    # keep current simple splitter; consider NLTK/Punkt as optional extra
    ...

def greedy_pack(sentences: list[str], soft_max=300, hard_max=360) -> list[str]:
    # keep Birdler's greedy + hard whitespace splits

def enforce_min_chunk_length(chunks: list[str], min_len=20, max_len=300) -> list[str]:
    out, i = [], 0
    while i < len(chunks):
        cur = chunks[i].strip()
        if len(cur) >= min_len or i == len(chunks) - 1:
            out.append(cur); i += 1
        else:
            if i + 1 < len(chunks):
                merged = f"{cur} {chunks[i+1]}".strip()
                out.append(merged if len(merged) <= max_len else cur)
                i += 2 if len(merged) <= max_len else 1
            else:
                out.append(cur); i += 1
    return out

def get_chunks(text, soft_max=300, hard_max=360, min_len=20):
    sents = split_sentences(text)
    chunks = greedy_pack(sents, soft_max, hard_max)
    return enforce_min_chunk_length(chunks, min_len=min_len, max_len=soft_max)
```

---

## 3) Recursive Long-Sentence Splitting (Graceful Over-max Handling)

**Intent & Why**
When a single sentence exceeds the hard limit, recursively split by priority separators (`; : - ,` then spaces) before forced character splits. This reduces mid-word cuts and keeps phrasing natural.

**Syntax lead**

```python
def split_long_sentence(s, max_len=300, seps=None):
    seps = seps or [';', ':', '-', ',', ' ']
    if len(s) <= max_len: return [s.strip()]
    if not seps:  # hard fallback
        return [s[i:i+max_len].strip() for i in range(0, len(s), max_len)]
    sep = seps[0]; parts = s.split(sep)
    if len(parts) == 1: return split_long_sentence(s, max_len, seps=seps[1:])
    out, cur = [], parts[0].strip()
    for part in parts[1:]:
        cand = (cur + sep + part).strip()
        if len(cand) > max_len:
            out.extend(split_long_sentence(cur, max_len, seps=seps[1:]))
            cur = part.strip()
        else:
            cur = cand
    out.extend(split_long_sentence(cur, max_len, seps=seps[1:]) if len(cur) > max_len else [cur])
    return out
```

---

## 4) Parallel Chunk Generation (Thread Pool) with Deterministic RNG

**Intent & Why**
Speed up long scripts by generating chunks concurrently, but keep determinism by per-task derived seeds. Use a bounded worker pool and collect results by chunk index to preserve order.

**Syntax lead**

```python
# gen_parallel.py
from concurrent.futures import ThreadPoolExecutor, as_completed

def gen_chunk_job(tts, chunk_text, chunk_idx, base_seed, cand_idx, attempt, gen_kwargs):
    seed = derive_seed(base_seed, chunk_idx, cand_idx, attempt)
    wav = generate_with_seed(tts, chunk_text, seed, **gen_kwargs)
    return chunk_idx, {"seed": seed, "cand_idx": cand_idx, "attempt": attempt, "wav": wav}

def generate_chunks_parallel(tts, chunks, base_seed, workers=4, n_cands=2, attempts=1, gen_kwargs=None):
    gen_kwargs = gen_kwargs or {}
    results = {i: [] for i in range(len(chunks))}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i, text in enumerate(chunks):
            for c in range(n_cands):
                for a in range(attempts):
                    futs.append(ex.submit(gen_chunk_job, tts, text, i, base_seed, c, a, gen_kwargs))
        for fut in as_completed(futs):
            idx, item = fut.result()
            results[idx].append(item)
    return results  # {chunk_idx: [ {seed, cand_idx, attempt, wav}, ... ]}
```

---

## 5) Whisper / Faster-Whisper Validation Pipeline

**Intent & Why**
Validate each candidate against its intended text and pick the best. This dramatically reduces misreads, stutters, and hallucinations. Include a bypass toggle and a fallback (e.g., best fuzzy score or longest transcript).

**Syntax lead**

```python
# validate.py
def load_whisper(backend: Literal["openai", "faster"], model_name: str, device: str):
    # try compute types: (cuda: float16→int8_float16→int8, cpu/mps: int8→float32)
    ...

def transcribe_and_score(path: str, target_text: str, whisper_model) -> tuple[float, str]:
    # return (fuzzy_score, transcript)
    ...

def validate_candidates(candidate_paths: list[str], target_text: str, whisper_model, threshold=0.85):
    passed, failed = [], []
    for p in candidate_paths:
        score, txt = transcribe_and_score(p, target_text, whisper_model)
        (passed if score >= threshold else failed).append((score, p, txt))
    return passed, failed
```

---

## 6) Retry Loop on Failed Chunks (Candidate Regeneration)

**Intent & Why**
If no candidate passes validation for a chunk, automatically regenerate candidates with new derived seeds up to `max_attempts`. This increases the success rate without manual intervention.

**Syntax lead**

```python
# retry.py
def retry_failed_chunks(tts, chunks, base_seed, prev_map, max_attempts, n_cands, gen_kwargs, validate_fn):
    attempts = {i:1 for i in range(len(chunks))}
    need = [i for i, items in prev_map.items() if not items]  # or: items exist but none passed
    while need:
        round_map = {}
        for i in need:
            round_map[i] = []
            for c in range(n_cands):
                seed = derive_seed(base_seed, i, c, attempts[i])
                wav = generate_with_seed(tts, chunks[i], seed, **gen_kwargs)
                round_map[i].append((seed, wav))
        # run validation on round_map, update prev_map with passing picks
        ...
        attempts = {i: attempts[i]+1 for i in need if attempts[i] < max_attempts}
        need = [i for i in need if attempts[i] < max_attempts and not has_pass(i, prev_map)]
    return prev_map
```

---

## 7) Candidate Selection (Policy) & Assembly

**Intent & Why**
Define a clear, deterministic policy to pick the winner per chunk:

1. If Whisper enabled: choose highest score; if still none, fallback to longest transcript or best score.
2. If bypassed: prefer primary seeded candidate `(cand_idx=0, attempt=0)`; else shortest duration.

**Syntax lead**

```python
# select.py
def pick_candidate(passed, failed, bypass=False, prefer_longest_on_fail=True):
    if bypass:
        return min(passed or failed, key=lambda x: duration(x), default=None)
    if passed:
        return max(passed, key=lambda x: x[0])  # score
    if failed:
        return (max if prefer_longest_on_fail else max)(failed, key=lambda x: len(x[2] or ""))  # transcript length or score
    return None

def assemble_waveforms(selected_by_chunk: dict[int, str], sr: int):
    # load with torchaudio; then Birdler’s trim + crossfade_concat
    ...
```

---

## 8) RNNoise (pyrnnoise) Denoising (Optional, Pre Cleanup)

**Intent & Why**
Apply RNNoise to the concatenated WAV (or per-chunk) before silence trimming/normalization; this removes hiss/click artifacts. Use CLI `denoise` if present, else Python API. Convert to 48k mono s16 for best results, then resample back.

**Syntax lead**

```python
# denoise.py
def denoise_in_place(wav_path: Path):
    orig_sr = probe_sr(wav_path)  # librosa or soundfile
    tmp_48k = str(wav_path).replace(".wav", "_48kmono.wav")
    tmp_dn  = str(wav_path).replace(".wav", "_dn.wav")
    # ffmpeg → 48k mono s16
    run(["ffmpeg","-y","-i",wav_path,"-ac","1","-ar","48000","-sample_fmt","s16",tmp_48k])
    if has_cli("denoise"):
        ok = run(["denoise", tmp_48k, tmp_dn]).returncode == 0 and Path(tmp_dn).stat().st_size > 1024
    else:
        ok = pyrnnoise_api(tmp_48k, tmp_dn)
    if not ok: return False
    # resample back to original sr if known
    if orig_sr:
        run(["ffmpeg","-y","-i",tmp_dn,"-ac","1","-ar",str(orig_sr), str(wav_path).replace(".wav","_resamp.wav")])
        os.replace(str(wav_path).replace(".wav","_resamp.wav"), wav_path)
    else:
        os.replace(tmp_dn, wav_path)
    cleanup(tmp_48k, tmp_dn)
    return True
```

---

## 9) Auto-Editor Silence Trimming (Optional)

**Intent & Why**
Use `auto-editor` to remove long silences and tighten pacing, then replace the main WAV. Offer knobs for threshold and margin; allow keeping the original WAV.

**Syntax lead**

```python
# post_silence.py
def run_auto_editor_in_place(wav_path: Path, threshold=0.06, margin=0.2, keep_original=False):
    inp = wav_path
    if keep_original:
        backup = str(wav_path).replace(".wav", "_original.wav")
        os.rename(wav_path, backup)
        inp = Path(backup)
    out = str(wav_path).replace(".wav", "_cleaned.wav")
    run(["auto-editor","--edit",f"audio:threshold={threshold}","--margin",f"{margin}s","--export","audio",str(inp),"-o",out], check=True)
    os.replace(out, wav_path)
```

---

## 10) FFmpeg Normalization (EBU R128 / Peak) (Optional)

**Intent & Why**
Normalize output loudness to a target standard (EBU R128) or peak limit to avoid clipping. Place after denoise & auto-editor for stable levels across takes.

**Syntax lead**

```python
# normalize.py
def normalize_ffmpeg_in_place(wav_path: Path, method="ebu", I=-24, TP=-2, LRA=7):
    tmp = str(wav_path).replace(".wav", "_norm.wav")
    if method == "ebu":
        af = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
    elif method == "peak":
        af = "alimiter=limit=-2dB"
    else:
        raise ValueError("bad normalize method")
    run(["ffmpeg","-y","-i",str(wav_path),"-af",af,tmp], check=True)
    os.replace(tmp, wav_path)
```

---

## 11) Per-Run Settings Artifacts (JSON + CSV) Without PII Text

**Intent & Why**
Write a `.settings.json` and `.settings.csv` beside outputs capturing knobs (not raw text). This simplifies reproducing a take and comparing runs while protecting content.

**Syntax lead**

```python
# settings_io.py
def save_settings(base_output_path: Path, output_files: list[str], params: dict):
    # zero out raw text fields; include seed(s), device, model hash, tts params, validation mode, postproc flags
    safe = {**params, "text_input": "", "output_audio_files": output_files, "generation_time": datetime.now().isoformat()}
    json.dump(safe, open(base_output_path.with_suffix(".settings.json"), "w"), indent=2)
    write_csv(base_output_path.with_suffix(".settings.csv"), safe)
```

---

## 12) Device Selection & Determinism Guardrails

**Intent & Why**
Prefer CUDA if available, else **MPS on macOS**, else CPU. Disable TF32 on CUDA for full determinism; set deterministic algorithms when possible. This mirrors Extended’s caution around repeatability.

**Syntax lead**

```python
# device.py
def select_device(user_pref=None):
    if user_pref: return user_pref
    import torch
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def set_determinism(device):
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = True
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
```

---

## 13) Progress, Logging, and Ordering Guarantees

**Intent & Why**
Provide user feedback (chunk N/M, %) and ensure final assembly respects original chunk order regardless of parallel completion. Switch from ad-hoc `print` to structured logging hooks where possible.

**Syntax lead**

```python
# progress.py
def log_progress(done, total):
    pct = int(100 * done / max(total, 1))
    print(f"[PROGRESS] {done}/{total} ({pct}%)")

# After futures complete:
ordered = [pick_candidate_for_chunk(i, result_map[i]) for i in range(total_chunks)]
full = crossfade_concat(ordered, fade_samples=2048)
```

---

## 14) Output Naming with Stable Metadata

**Intent & Why**
Encode voice name, first words of text/script stem, timestamp, generation index, and seed into filenames. Aids traceability when comparing multiple takes.

**Syntax lead**

```python
# naming.py
def build_outname(voice_slug, text_slug, gen_idx, seed, ts=None):
    ts = ts or int(time.time())
    return f"{voice_slug}_{text_slug}_gen{gen_idx}_seed{seed}_{ts}.wav"
```

---

## 15) Optional: VC (Voice Conversion) Hook (Future)

**Intent & Why**
Birdler is TTS-first, but wiring a VC subcommand enables converting recorded speech to a target voice using the same device/IO scaffolding. Keep this modular (separate file/flag) to avoid UI dependency bloat.

**Syntax lead**

```python
# vc_cmd.py
def run_vc(input_wav: Path, target_ref: Path, pitch_shift=0, watermark=False):
    vc = ChatterboxVC.from_pretrained(select_device())
    out = vc.generate(str(input_wav), target_voice_path=str(target_ref),
                      apply_watermark=watermark, pitch_shift=pitch_shift)
    torchaudio.save(out_path, out, vc.sr)
```

---

## Integration Order (Suggested)

1. **Chunking upgrades** (#2, #3)
2. **Deterministic seeding & device guardrails** (#1, #12)
3. **Parallel generation with deterministic RNG** (#4)
4. **Validation + selection + retries** (#5–#7)
5. **Post-processing chain (RNNoise → Auto-Editor → Normalization)** (#8–#10)
6. **Settings artifacts & naming** (#11, #14)
7. **Progress/logging polish** (#13)
8. **Optional VC hook** (#15)

This sequence minimizes regressions: improve chunk quality, make outputs reproducible, then add speed and quality gates, and finally polish ergonomics.

---

```
```
