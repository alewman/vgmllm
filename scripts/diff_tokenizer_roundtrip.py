"""Tokenizer round-trip diff tool.

Compares the NoteEvents produced directly from a VGM (the "shortcut" path that
sounds great) against the NoteEvents that come out of encode→decode (the full
token round-trip used for training).

Any timing / ordering / pitch problem in the full round-trip will show up as a
structural diff here, without needing any audio comparison.

Pipeline
--------
    VGM ─► decode_vgm() ─► NoteEvents_A   (shortcut — known good)
    VGM ─► tokenizer.encode() ─► tokens ─► tokenizer.decode() ─► NoteEvents_B

Usage
-----
    python scripts/diff_tokenizer_roundtrip.py <vgm_or_vgz>
    python scripts/diff_tokenizer_roundtrip.py <vgm_or_vgz> --tol 441 --csv
    python scripts/diff_tokenizer_roundtrip.py <vgm_or_vgz> --csv --out-dir output/diff

Outputs
-------
- Console summary: worst channels, counts, drift stats, reorder count
- Per-channel CSV (with --csv): one row per aligned note pair

Notes
-----
- Sample rate is always 44100 Hz (Genesis VGM standard).
- "Reorder events" = pairs of consecutive notes (in A order) where B has
  sample_on[i] > sample_on[i+1].  These are the "slightly backwards" notes.
- Drift is always signed (positive = B fires LATER than A).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ── repo layout ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from genesis_music.vgm_parser import load_vgm
from genesis_music.tokenizer_v6 import TokenizerV6, ComposerMap, VOCAB_SIZE
from genesis_music.ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    NoteEvent, decode_vgm,
)

SAMPLE_RATE = 44_100

# Channel names for display
CH_NAMES: dict[int, str] = {
    0: "CH_FM_1",  1: "CH_FM_2",  2: "CH_FM_3",
    3: "CH_FM_4",  4: "CH_FM_5",  5: "CH_FM_6",
    6: "CH_DAC",
    7: "CH_PSG_1", 8: "CH_PSG_2", 9: "CH_PSG_3",
    10: "CH_PSG_NOISE",
}

# ── per-note diff record ──────────────────────────────────────────────────────

class NotePair(NamedTuple):
    """Aligned pair from NoteEvents_A and NoteEvents_B."""
    note_index_a: int          # index in channel's A list
    note_index_b: int          # index in channel's B list (-1 = dropped)
    pitch_a: int
    pitch_b: int               # -1 if dropped
    sample_on_a: int
    sample_on_b: int           # -1 if dropped
    sample_off_a: int          # -1 if open in A
    sample_off_b: int          # -1 if dropped or open in B
    d_on: int                  # sample_on_b - sample_on_a  (0 if dropped)
    d_duration: int            # (dur_b - dur_a)            (0 if open/dropped)
    d_pitch: int               # pitch_b - pitch_a          (0 if dropped)
    d_velocity: int
    reorder: bool              # True when B[i].sample_on > B[i+1].sample_on
                               # for what was monotone in A


@dataclass
class ChannelDiff:
    ch: int
    name: str
    count_a: int
    count_b: int
    pairs: list[NotePair]
    dropped_in_b: int          # notes in A with no B match
    inserted_in_b: int         # notes in B with no A match
    reorder_count: int         # consecutive pairs where B order flips

    # Drift statistics (over matched pairs, samples)
    mean_d_on: float
    median_d_on: float
    p95_d_on: float
    max_abs_d_on: int

    mean_d_dur: float
    p95_d_dur: float

    pitch_mismatches: int
    velocity_mismatches: int


# ── alignment ────────────────────────────────────────────────────────────────

def _group_by_channel(events: list[NoteEvent]) -> dict[int, list[NoteEvent]]:
    groups: dict[int, list[NoteEvent]] = defaultdict(list)
    for e in events:
        groups[e.channel].append(e)
    return groups


def _align_channel(
    a_notes: list[NoteEvent],
    b_notes: list[NoteEvent],
    tol_samples: int,
) -> list[NotePair]:
    """Greedy nearest-time alignment, pitch-aware."""
    pairs: list[NotePair] = []

    if not a_notes:
        return pairs

    # Simple case: same count — pair by index directly
    if len(a_notes) == len(b_notes):
        for i, (a, b) in enumerate(zip(a_notes, b_notes)):
            pairs.append(_make_pair(i, i, a, b))
        return pairs

    # Different counts: greedy match by closest sample_on + same pitch within tol.
    # We work through A in order and claim the earliest matching unmatched B note.
    used_b: set[int] = set()
    for ia, a in enumerate(a_notes):
        best_ib: int = -1
        best_dist: int = tol_samples + 1

        for ib, b in enumerate(b_notes):
            if ib in used_b:
                continue
            if b.pitch != a.pitch:
                continue
            dist = abs(b.sample_on - a.sample_on)
            if dist < best_dist:
                best_dist = dist
                best_ib = ib

        if best_ib >= 0:
            used_b.add(best_ib)
            pairs.append(_make_pair(ia, best_ib, a_notes[ia], b_notes[best_ib]))
        else:
            # Dropped — no B match
            a = a_notes[ia]
            dur_a = a.sample_off - a.sample_on if a.sample_off >= 0 else -1
            pairs.append(NotePair(
                note_index_a=ia, note_index_b=-1,
                pitch_a=a.pitch, pitch_b=-1,
                sample_on_a=a.sample_on, sample_on_b=-1,
                sample_off_a=a.sample_off, sample_off_b=-1,
                d_on=0, d_duration=0, d_pitch=0, d_velocity=0,
                reorder=False,
            ))

    return pairs


def _make_pair(ia: int, ib: int, a: NoteEvent, b: NoteEvent) -> NotePair:
    dur_a = (a.sample_off - a.sample_on) if a.sample_off >= 0 else -1
    dur_b = (b.sample_off - b.sample_on) if b.sample_off >= 0 else -1
    d_dur = (dur_b - dur_a) if (dur_a >= 0 and dur_b >= 0) else 0
    return NotePair(
        note_index_a=ia,
        note_index_b=ib,
        pitch_a=a.pitch,
        pitch_b=b.pitch,
        sample_on_a=a.sample_on,
        sample_on_b=b.sample_on,
        sample_off_a=a.sample_off,
        sample_off_b=b.sample_off,
        d_on=b.sample_on - a.sample_on,
        d_duration=d_dur,
        d_pitch=b.pitch - a.pitch,
        d_velocity=b.velocity - a.velocity,
        reorder=False,  # filled in below
    )


def _compute_reorders(pairs: list[NotePair]) -> list[NotePair]:
    """Mark consecutive matched pairs where B order flips."""
    matched = [(i, p) for i, p in enumerate(pairs) if p.note_index_b >= 0]
    result = list(pairs)
    for k in range(len(matched) - 1):
        idx_curr, p_curr = matched[k]
        idx_next, p_next = matched[k + 1]
        if p_curr.sample_on_b > p_next.sample_on_b:
            # Replace immutable NamedTuple by reconstructing
            result[idx_curr] = p_curr._replace(reorder=True)
    return result


def _stats(values: list[int]) -> tuple[float, float, float, int]:
    """Return (mean, median, p95, max_abs)."""
    if not values:
        return 0.0, 0.0, 0.0, 0
    n = len(values)
    s = sorted(values)
    mean   = sum(values) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    p95    = s[min(int(n * 0.95), n - 1)]
    max_abs = max(abs(v) for v in values)
    return mean, median, p95, max_abs


def diff_channel(
    ch: int,
    a_notes: list[NoteEvent],
    b_notes: list[NoteEvent],
    tol_samples: int,
) -> ChannelDiff:
    # Sort both by sample_on (should already be, but be safe)
    a_notes = sorted(a_notes, key=lambda e: e.sample_on)
    b_notes = sorted(b_notes, key=lambda e: e.sample_on)

    pairs = _align_channel(a_notes, b_notes, tol_samples)
    pairs = _compute_reorders(pairs)

    matched = [p for p in pairs if p.note_index_b >= 0]
    dropped  = sum(1 for p in pairs if p.note_index_b < 0)
    inserted = len(b_notes) - len(matched)

    d_ons  = [p.d_on for p in matched]
    d_durs = [p.d_duration for p in matched if p.d_duration != 0 or
              (p.sample_off_a >= 0 and p.sample_off_b >= 0)]
    reorder_count = sum(1 for p in pairs if p.reorder)

    mean_on, median_on, p95_on, max_abs_on = _stats(d_ons)
    mean_dur, _, p95_dur, _ = _stats(d_durs)

    pitch_mismatches   = sum(1 for p in matched if p.d_pitch != 0)
    velocity_mismatches = sum(1 for p in matched if p.d_velocity != 0)

    return ChannelDiff(
        ch=ch,
        name=CH_NAMES.get(ch, f"CH_{ch}"),
        count_a=len(a_notes),
        count_b=len(b_notes),
        pairs=pairs,
        dropped_in_b=dropped,
        inserted_in_b=max(0, inserted),
        reorder_count=reorder_count,
        mean_d_on=mean_on,
        median_d_on=median_on,
        p95_d_on=p95_on,
        max_abs_d_on=max_abs_on,
        mean_d_dur=mean_dur,
        p95_d_dur=p95_dur,
        pitch_mismatches=pitch_mismatches,
        velocity_mismatches=velocity_mismatches,
    )


# ── reporting ─────────────────────────────────────────────────────────────────

def _ms(samples: float) -> str:
    return f"{samples / SAMPLE_RATE * 1000:.1f} ms"


def _sign(v: float) -> str:
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"


def print_report(diffs: list[ChannelDiff]) -> None:
    BAR = "─" * 72

    # ── per-channel detail ──
    print(f"\n{BAR}")
    print("  PER-CHANNEL DETAIL")
    print(BAR)
    for d in diffs:
        total_a = d.count_a
        total_b = d.count_b
        count_ok = "✓" if total_a == total_b else f"  ← count mismatch ({total_b - total_a:+d})"
        print(f"\n  {d.name}:  {total_a} notes A  /  {total_b} notes B  {count_ok}")

        if total_a == 0 and total_b == 0:
            print("    (channel unused in both paths)")
            continue

        matched_count = len([p for p in d.pairs if p.note_index_b >= 0])
        if matched_count == 0:
            print("    No matched pairs — all notes dropped or unaligned.")
            continue

        print(f"    Dropped in B : {d.dropped_in_b}   |   Inserted in B : {d.inserted_in_b}")
        print(f"    on-time drift  :  mean={_sign(d.mean_d_on)} samp  "
              f"({_ms(d.mean_d_on)})  |  median={_sign(d.median_d_on)}  "
              f"|  p95={_sign(d.p95_d_on)} ({_ms(d.p95_d_on)})  "
              f"|  max_abs={d.max_abs_d_on} ({_ms(d.max_abs_d_on)})")
        print(f"    duration drift :  mean={_sign(d.mean_d_dur)} samp  "
              f"({_ms(d.mean_d_dur)})  |  p95={_sign(d.p95_d_dur)} ({_ms(d.p95_d_dur)})")
        print(f"    pitch mismatches : {d.pitch_mismatches}   |   "
              f"velocity mismatches : {d.velocity_mismatches}")
        print(f"    reorder events   : {d.reorder_count}", end="")
        if d.reorder_count > 0:
            # Show first reorder location
            first = next(p for p in d.pairs if p.reorder)
            t_sec = first.sample_on_a / SAMPLE_RATE
            print(f"   ← first reorder at t={t_sec:.2f}s "
                  f"(A sample {first.sample_on_a}, pitch {first.pitch_a})", end="")
        print()

    # ── summary ──
    print(f"\n{BAR}")
    print("  SUMMARY")
    print(BAR)

    dirty = [d for d in diffs if d.count_a > 0 or d.count_b > 0]
    clean = [d for d in dirty if
             d.reorder_count == 0 and d.pitch_mismatches == 0 and
             d.dropped_in_b == 0 and abs(d.mean_d_on) < 1.0]
    problem = [d for d in dirty if d not in clean]

    if problem:
        by_reorder = sorted(problem, key=lambda d: d.reorder_count, reverse=True)
        top_reorder = [(d.name, d.reorder_count) for d in by_reorder if d.reorder_count > 0]
        if top_reorder:
            items = ", ".join(f"{n} ({c})" for n, c in top_reorder)
            print(f"  Reorder counts   : {items}")

        by_drift = sorted(problem, key=lambda d: abs(d.p95_d_on), reverse=True)
        top_drift = by_drift[:5]
        items = ", ".join(
            f"{d.name} ({_sign(d.p95_d_on)} samp = {_ms(d.p95_d_on)})"
            for d in top_drift if abs(d.p95_d_on) > 0
        )
        if items:
            print(f"  Worst p95 drift  : {items}")

        pitch_bad = [d for d in dirty if d.pitch_mismatches > 0]
        if pitch_bad:
            items = ", ".join(f"{d.name} ({d.pitch_mismatches})" for d in pitch_bad)
            print(f"  Pitch mismatches : {items}")

        count_bad = [d for d in dirty if d.count_a != d.count_b]
        if count_bad:
            items = ", ".join(f"{d.name} ({d.count_a}A/{d.count_b}B)" for d in count_bad)
            print(f"  Count mismatches : {items}")
    else:
        print("  All active channels: no reorders, no pitch errors, drift < 1 sample.")

    if clean:
        print(f"  Channels clean   : {', '.join(d.name for d in clean)}")

    print(BAR)


def write_csv(d: ChannelDiff, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "note_index_a", "note_index_b",
            "pitch_a", "pitch_b", "d_pitch",
            "sample_on_a", "sample_on_b", "d_on_samp", "d_on_ms",
            "sample_off_a", "sample_off_b",
            "d_duration_samp", "d_duration_ms",
            "d_velocity",
            "reorder",
        ])
        for p in d.pairs:
            writer.writerow([
                p.note_index_a, p.note_index_b,
                p.pitch_a, p.pitch_b, p.d_pitch,
                p.sample_on_a, p.sample_on_b,
                p.d_on, f"{p.d_on / SAMPLE_RATE * 1000:.2f}",
                p.sample_off_a, p.sample_off_b,
                p.d_duration, f"{p.d_duration / SAMPLE_RATE * 1000:.2f}",
                p.d_velocity,
                1 if p.reorder else 0,
            ])


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff NoteEvents_A (direct parse) vs NoteEvents_B (token round-trip)"
    )
    parser.add_argument("vgm", type=Path, help="Input VGM or VGZ file")
    parser.add_argument(
        "--tol", type=int, default=441,
        help="Alignment tolerance in samples when counts differ (default 441 = 10 ms)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Write per-channel CSV files alongside the report",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for CSV output (default: output/diff/<stem>/)",
    )
    parser.add_argument(
        "--channels", type=str, default=None,
        help="Comma-separated channel indices to analyse (default: all). E.g. 0,1,5",
    )
    args = parser.parse_args()

    # ── load VGM ──
    log.info("Loading: %s", args.vgm)
    vgm = load_vgm(args.vgm)

    # ── NoteEvents_A: direct parse (shortcut path) ──
    events_a, _last_patches = decode_vgm(vgm)
    log.info("NoteEvents_A (direct parse): %d events across %d channels",
             len(events_a), len(set(e.channel for e in events_a)))

    # ── optional: load support maps (same as hear_training_data_v6.py) ──
    data_dir   = REPO_ROOT / "data"
    prepared   = data_dir / "prepared_v4"

    composer_map = None
    cmap_path = prepared / "composer_map_v4.json"
    if cmap_path.exists():
        composer_map = ComposerMap.load(cmap_path)

    dac_slot_map: dict[int, int] = {}
    dac_path = prepared / "dac_slot_map_v4.json"
    if dac_path.exists():
        raw = json.loads(dac_path.read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}

    tokenizer = TokenizerV6(composer_map=composer_map, dac_slot_map=dac_slot_map)
    log.info("TokenizerV6 ready (vocab size %d)", VOCAB_SIZE)

    # ── encode ──
    tokens = tokenizer.encode(vgm, skip_filter=True)
    if tokens is None:
        log.error("encode() returned None — file filtered out.")
        sys.exit(1)
    log.info("Encoded: %d tokens", len(tokens))

    # ── NoteEvents_B: token round-trip ──
    events_b, header = tokenizer.decode(tokens)
    log.info("NoteEvents_B (token round-trip): %d events across %d channels",
             len(events_b), len(set(e.channel for e in events_b)))

    # ── diff per channel ──
    a_by_ch = _group_by_channel(events_a)
    b_by_ch = _group_by_channel(events_b)

    all_channels = sorted(set(a_by_ch.keys()) | set(b_by_ch.keys()))

    if args.channels:
        filter_set = {int(c.strip()) for c in args.channels.split(",")}
        all_channels = [c for c in all_channels if c in filter_set]

    diffs: list[ChannelDiff] = []
    for ch in all_channels:
        a_notes = a_by_ch.get(ch, [])
        b_notes = b_by_ch.get(ch, [])
        d = diff_channel(ch, a_notes, b_notes, args.tol)
        diffs.append(d)
        log.info("  %s: A=%d  B=%d  reorders=%d  p95_drift=%+d samp  pitch_err=%d",
                 d.name, d.count_a, d.count_b, d.reorder_count,
                 int(d.p95_d_on), d.pitch_mismatches)

    # ── console report ──
    print(f"\n{'═' * 72}")
    print(f"  TOKENIZER ROUND-TRIP DIFF:  {args.vgm.name}")
    print(f"  Detected tempo: {header.get('tempo_bpm', '?'):.1f} BPM  |  "
          f"meter: {header.get('meter', '?')}  |  "
          f"tol: {args.tol} samp ({args.tol / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"{'═' * 72}")
    print_report(diffs)

    # ── CSV output ──
    if args.csv:
        if args.out_dir is None:
            out_dir = REPO_ROOT / "output" / "diff" / args.vgm.stem
        else:
            out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in diffs:
            if d.count_a == 0 and d.count_b == 0:
                continue
            csv_path = out_dir / f"{d.name}.csv"
            write_csv(d, csv_path)
            log.info("Wrote CSV: %s", csv_path)

        # Also write a summary JSON for easy programmatic parsing
        summary = {
            "source": str(args.vgm),
            "tempo_bpm": header.get("tempo_bpm"),
            "meter": header.get("meter"),
            "tol_samples": args.tol,
            "channels": [
                {
                    "name": d.name,
                    "ch_index": d.ch,
                    "count_a": d.count_a,
                    "count_b": d.count_b,
                    "dropped_in_b": d.dropped_in_b,
                    "inserted_in_b": d.inserted_in_b,
                    "reorder_count": d.reorder_count,
                    "mean_d_on_samp": round(d.mean_d_on, 2),
                    "median_d_on_samp": round(d.median_d_on, 2),
                    "p95_d_on_samp": round(d.p95_d_on, 2),
                    "max_abs_d_on_samp": d.max_abs_d_on,
                    "mean_d_dur_samp": round(d.mean_d_dur, 2),
                    "p95_d_dur_samp": round(d.p95_d_dur, 2),
                    "pitch_mismatches": d.pitch_mismatches,
                    "velocity_mismatches": d.velocity_mismatches,
                }
                for d in diffs
            ],
        }
        summary_path = out_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Wrote summary: %s", summary_path)

        print(f"\n  CSV output → {out_dir}")


if __name__ == "__main__":
    main()
