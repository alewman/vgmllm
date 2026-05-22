"""
Analyze loop structure across the VGM corpus.

Metrics:
- What fraction of tracks have a loop_offset (LOOP_PRESENT)?
- For looping tracks: what is the intro length vs loop length vs total length?
- How many times does the loop repeat within the full VGM playback duration?
- How much token budget is wasted on redundant loop repetitions in training data?
- Distribution of loop lengths (in seconds and estimated tokens).
"""

import sys
from pathlib import Path
import struct
import gzip
import statistics

VGM_DIR = Path(__file__).parents[2] / "data" / "vgm"
MAX_FILES = 2000  # sample — full corpus is large

# Rough token rate estimates from prior analysis
# Dense game (TFIV-style): ~2000 tok/s; Sparse (Sonic-style): ~300 tok/s
# Use a middle estimate: ~600 tok/s as corpus average
TOKENS_PER_SEC = 600
SEQ_LEN = 16384


def read_vgm_header(path: Path) -> dict | None:
    """Parse VGM/VGZ header to extract loop_offset, total_samples, loop_samples."""
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == ".vgz":
            raw = gzip.decompress(raw)

        if len(raw) < 64 or raw[:4] != b"Vgm ":
            return None

        eof_offset    = struct.unpack_from("<I", raw, 0x04)[0]
        version       = struct.unpack_from("<I", raw, 0x08)[0]
        total_samples = struct.unpack_from("<I", raw, 0x18)[0]  # samples at 44100 Hz
        loop_offset   = struct.unpack_from("<I", raw, 0x1C)[0]
        loop_samples  = struct.unpack_from("<I", raw, 0x20)[0]

        total_sec = total_samples / 44100.0
        loop_sec  = loop_samples  / 44100.0
        intro_sec = total_sec - loop_sec if loop_offset != 0 else 0.0

        return {
            "has_loop":   loop_offset != 0,
            "total_sec":  total_sec,
            "loop_sec":   loop_sec,
            "intro_sec":  intro_sec,
        }
    except Exception:
        return None


def main():
    files = list(VGM_DIR.rglob("*.vgz"))[:MAX_FILES]
    if not files:
        files = list(VGM_DIR.rglob("*.vgm"))[:MAX_FILES]
    print(f"Analyzing {len(files)} VGM files...\n")

    no_loop, has_loop = [], []
    loop_repeat_counts = []
    wasted_token_fractions = []

    for f in files:
        h = read_vgm_header(f)
        if h is None:
            continue
        if h["has_loop"] and h["loop_sec"] > 0.5 and h["total_sec"] > 1.0:
            has_loop.append(h)
            # How many times does the loop body repeat in the full file?
            # VGM standard plays intro once then loops twice (total_samples counts 2 loop passes)
            # But total_samples can vary — compute repeat count from samples
            if h["loop_sec"] > 0:
                repeats = (h["total_sec"] - h["intro_sec"]) / h["loop_sec"]
                loop_repeat_counts.append(repeats)
                # Token budget used by redundant repetitions
                redundant_sec = max(0.0, h["total_sec"] - h["intro_sec"] - h["loop_sec"])
                wasted_frac = redundant_sec / h["total_sec"] if h["total_sec"] > 0 else 0.0
                wasted_token_fractions.append(wasted_frac)
        elif not h["has_loop"] and h["total_sec"] > 1.0:
            no_loop.append(h)

    total = len(has_loop) + len(no_loop)
    print(f"=== Loop Presence ===")
    print(f"  LOOP_PRESENT : {len(has_loop):5d}  ({100*len(has_loop)/total:.1f}%)")
    print(f"  LOOP_ABSENT  : {len(no_loop):5d}  ({100*len(no_loop)/total:.1f}%)")
    print(f"  Total parsed : {total}")

    if has_loop:
        total_secs   = [h["total_sec"]  for h in has_loop]
        loop_secs    = [h["loop_sec"]   for h in has_loop]
        intro_secs   = [h["intro_sec"]  for h in has_loop]

        print(f"\n=== Looping Track Durations ===")
        print(f"  Total duration  — median: {statistics.median(total_secs):.1f}s  mean: {statistics.mean(total_secs):.1f}s  max: {max(total_secs):.1f}s")
        print(f"  Loop body       — median: {statistics.median(loop_secs):.1f}s   mean: {statistics.mean(loop_secs):.1f}s   max: {max(loop_secs):.1f}s")
        print(f"  Intro (pre-loop)— median: {statistics.median(intro_secs):.1f}s  mean: {statistics.mean(intro_secs):.1f}s")

        print(f"\n=== Loop Repetitions per Track ===")
        print(f"  Median repeats : {statistics.median(loop_repeat_counts):.1f}x")
        print(f"  Mean repeats   : {statistics.mean(loop_repeat_counts):.1f}x")
        print(f"  Max repeats    : {max(loop_repeat_counts):.1f}x")

        print(f"\n=== Token Budget Waste (redundant loop passes) ===")
        mean_waste = statistics.mean(wasted_token_fractions)
        median_waste = statistics.median(wasted_token_fractions)
        print(f"  Median wasted fraction : {100*median_waste:.1f}%")
        print(f"  Mean wasted fraction   : {100*mean_waste:.1f}%")

        # Estimate: if we truncated to intro+1 loop pass, how long would tracks be?
        one_pass_secs = [h["intro_sec"] + h["loop_sec"] for h in has_loop]
        print(f"\n=== If Truncated to Intro + 1 Loop Pass ===")
        print(f"  Median one-pass duration : {statistics.median(one_pass_secs):.1f}s")
        print(f"  Mean one-pass duration   : {statistics.mean(one_pass_secs):.1f}s")
        est_tokens_median = statistics.median(one_pass_secs) * TOKENS_PER_SEC
        est_tokens_mean   = statistics.mean(one_pass_secs)   * TOKENS_PER_SEC
        print(f"  Est. tokens (median)     : {est_tokens_median:,.0f}  (seq_len={SEQ_LEN})")
        print(f"  Est. tokens (mean)       : {est_tokens_mean:,.0f}")

        short = sum(1 for s in one_pass_secs if s * TOKENS_PER_SEC < SEQ_LEN)
        print(f"  Tracks fitting in seq_len: {short}/{len(has_loop)} ({100*short/len(has_loop):.1f}%)")

        # Distribution of loop lengths
        buckets = {"<10s": 0, "10-30s": 0, "30-60s": 0, "60-120s": 0, ">120s": 0}
        for s in loop_secs:
            if s < 10:       buckets["<10s"] += 1
            elif s < 30:     buckets["10-30s"] += 1
            elif s < 60:     buckets["30-60s"] += 1
            elif s < 120:    buckets["60-120s"] += 1
            else:            buckets[">120s"] += 1
        print(f"\n=== Loop Body Length Distribution ===")
        for k, v in buckets.items():
            bar = "#" * (v // 5)
            print(f"  {k:>8s}: {v:4d}  {bar}")


if __name__ == "__main__":
    main()
