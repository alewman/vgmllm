"""Diagnose training data for issues that could cause gradient explosions.

Checks:
1. Out-of-range tokens (>= VOCAB_SIZE)
2. Sequences that are all-PAD or all a single token
3. Token frequency distribution (detect extreme outliers)
4. Sequences with suspiciously high/low entropy
5. Reports worst offending chunk indices
"""
import numpy as np
from pathlib import Path
from collections import Counter
import sys

VOCAB_SIZE = 449
PAD = 0
CHUNK_SIZE = 16384

def diagnose(data_path: Path, max_chunks: int = None):
    print(f"\nLoading {data_path} ...")
    tokens = np.load(data_path, mmap_mode="r")
    total_tokens = len(tokens)
    n_chunks = total_tokens // CHUNK_SIZE
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Chunks ({CHUNK_SIZE} tokens each): {n_chunks:,}")
    print()

    # --- 1. Out-of-range tokens ---
    print("=== Checking for out-of-range tokens ===")
    # Sample in blocks to avoid loading everything at once
    oob_count = 0
    oob_chunks = []
    for i in range(0, n_chunks, max(1, n_chunks // 1000)):
        start = i * CHUNK_SIZE
        chunk = tokens[start:start + CHUNK_SIZE].astype(np.int64)
        oob = np.where((chunk < 0) | (chunk >= VOCAB_SIZE))[0]
        if len(oob) > 0:
            oob_count += len(oob)
            oob_chunks.append((i, chunk[oob].tolist()[:5]))
    if oob_count:
        print(f"  WARNING: {oob_count} out-of-range tokens found!")
        for chunk_idx, vals in oob_chunks[:10]:
            print(f"    Chunk {chunk_idx}: values {vals}")
    else:
        print("  OK: No out-of-range tokens found (sampled)")

    # --- 2. Global token frequency ---
    print("\n=== Token frequency distribution ===")
    # Sample 10% of chunks
    freq = Counter()
    sample_step = max(1, n_chunks // 100)
    for i in range(0, n_chunks, sample_step):
        start = i * CHUNK_SIZE
        chunk = tokens[start:start + CHUNK_SIZE]
        for tok, cnt in zip(*np.unique(chunk, return_counts=True)):
            freq[int(tok)] += int(cnt)

    total_sampled = sum(freq.values())
    top10 = freq.most_common(10)
    print(f"  Sampled {total_sampled:,} tokens from {n_chunks // sample_step} chunks")
    print(f"  Top 10 most frequent tokens:")
    for tok, cnt in top10:
        pct = 100 * cnt / total_sampled
        print(f"    Token {tok:4d}: {cnt:8,}  ({pct:.1f}%)")

    pad_pct = 100 * freq.get(PAD, 0) / max(total_sampled, 1)
    print(f"\n  PAD token (0) frequency: {pad_pct:.1f}%")
    if pad_pct > 50:
        print("  WARNING: Over 50% PAD tokens — sequences may be mostly padding!")

    # --- 3. Degenerate chunks (all same token, or >90% PAD) ---
    print("\n=== Checking for degenerate chunks ===")
    degenerate = []
    high_pad = []
    sample_step2 = max(1, n_chunks // 500)
    for i in range(0, n_chunks, sample_step2):
        start = i * CHUNK_SIZE
        chunk = tokens[start:start + CHUNK_SIZE].astype(np.int64)
        unique_toks = len(np.unique(chunk))
        pad_frac = np.mean(chunk == PAD)
        if unique_toks <= 2:
            degenerate.append((i, unique_toks, np.unique(chunk).tolist()))
        if pad_frac > 0.9:
            high_pad.append((i, pad_frac))

    if degenerate:
        print(f"  WARNING: {len(degenerate)} near-degenerate chunks (<=2 unique tokens):")
        for idx, n_unique, vals in degenerate[:10]:
            print(f"    Chunk {idx}: {n_unique} unique tokens, values={vals}")
    else:
        print("  OK: No degenerate chunks found (sampled)")

    if high_pad:
        print(f"  WARNING: {len(high_pad)} chunks with >90% PAD tokens:")
        for idx, frac in high_pad[:5]:
            print(f"    Chunk {idx}: {frac:.1%} PAD")
    else:
        print("  OK: No high-PAD chunks found (sampled)")

    # --- 4. Entropy check ---
    print("\n=== Per-chunk entropy (sample of 100 chunks) ===")
    entropies = []
    sample_step3 = max(1, n_chunks // 100)
    for i in range(0, n_chunks, sample_step3):
        start = i * CHUNK_SIZE
        chunk = tokens[start:start + CHUNK_SIZE].astype(np.int64)
        # Remove PAD for entropy calc
        non_pad = chunk[chunk != PAD]
        if len(non_pad) < 10:
            entropies.append(0.0)
            continue
        _, counts = np.unique(non_pad, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(entropy)

    entropies = np.array(entropies)
    print(f"  Mean entropy: {entropies.mean():.2f} bits")
    print(f"  Min entropy:  {entropies.min():.2f} bits (chunk {entropies.argmin() * sample_step3})")
    print(f"  Max entropy:  {entropies.max():.2f} bits")
    low_entropy = np.where(entropies < 1.0)[0]
    if len(low_entropy) > 0:
        print(f"  WARNING: {len(low_entropy)} chunks with entropy < 1.0 bit (very repetitive)")
        for idx in low_entropy[:5]:
            print(f"    Chunk ~{idx * sample_step3}")
    else:
        print("  OK: All sampled chunks have healthy entropy")

    print("\n=== Summary ===")
    print(f"  VOCAB_SIZE expected: {VOCAB_SIZE}")
    print(f"  OOB tokens found: {oob_count}")
    print(f"  Degenerate chunks: {len(degenerate)}")
    print(f"  High-PAD chunks: {len(high_pad)}")
    print(f"  Low-entropy chunks: {len(low_entropy)}")
    issues = oob_count + len(degenerate) + len(high_pad) + len(low_entropy)
    if issues == 0:
        print("\n  Data looks clean. Gradient explosions likely due to LR/model instability.")
    else:
        print(f"\n  {issues} potential data quality issues found — these could cause loss spikes.")

if __name__ == "__main__":
    data_dir = Path("data/prepared_v4")
    print("Diagnosing TRAIN data...")
    diagnose(data_dir / "train.npy", max_chunks=50000)
    print("\n" + "="*60)
    print("Diagnosing VAL data...")
    diagnose(data_dir / "val.npy")
