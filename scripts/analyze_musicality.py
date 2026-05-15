"""Analyze musicality metrics of generated vs real VGM files.

Measures:
1. Note distribution (pitch histogram, interval distribution)
2. Rhythmic regularity (note timing quantization)
3. Repetition/structure (n-gram repetition of note sequences)
4. Channel usage (polyphony, voice leading)
"""

from __future__ import annotations
import sys, json, math
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genesis_music.vgm_parser import load_vgm, EventType
from genesis_music.tokenizer_v2 import encode_events_v2


def analyze_file(path: Path) -> dict | None:
    """Extract musicality metrics from a single VGM file."""
    try:
        vgm = load_vgm(path)
    except Exception as e:
        return None

    tokens = encode_events_v2(vgm.events, include_dac=False)

    # --- Extract note events per channel ---
    notes_by_ch: dict[int, list[tuple[str, str]]] = defaultdict(list)  # ch -> [(note, type)]
    all_notes: list[str] = []
    all_intervals: list[int] = []
    note_on_times: list[int] = []  # token positions of note-ons
    wait_positions: list[int] = []  # accumulated wait tokens

    ch_active: dict[int, str | None] = {}  # ch -> current note name
    token_pos = 0
    total_waits = 0
    consecutive_waits = 0
    max_consecutive_waits = 0
    note_count = 0
    wait_count = 0
    raw_count = 0
    total_tokens = len(tokens)

    for t in tokens:
        if t.startswith("<WAIT:"):
            wait_count += 1
            consecutive_waits += 1
            max_consecutive_waits = max(max_consecutive_waits, consecutive_waits)
        elif ":ON:" in t:
            consecutive_waits = 0
            parts = t.split(":")
            ch = int(parts[0][2:])
            note = parts[2]
            if note != "X":
                all_notes.append(note)
                note_on_times.append(token_pos)
                notes_by_ch[ch].append(note)

                # Interval from previous note on same channel
                if ch in ch_active and ch_active[ch] is not None:
                    prev = ch_active[ch]
                    interval = _note_to_midi(note) - _note_to_midi(prev)
                    all_intervals.append(interval)
                ch_active[ch] = note
            note_count += 1
        elif ":OFF" in t and ":PITCH:" not in t:
            consecutive_waits = 0
            # Don't reset ch_active — keep last pitch for interval tracking
            note_count += 1
        elif ":PITCH:" in t:
            consecutive_waits = 0
            parts = t.split(":")
            ch = int(parts[0][2:])
            note = parts[2]
            if note != "X":
                all_notes.append(note)
                notes_by_ch[ch].append(note)
                if ch in ch_active and ch_active[ch] is not None:
                    prev = ch_active[ch]
                    interval = _note_to_midi(note) - _note_to_midi(prev)
                    all_intervals.append(interval)
                ch_active[ch] = note
            note_count += 1
        else:
            consecutive_waits = 0
            raw_count += 1
        token_pos += 1

    if not all_notes:
        return None

    # --- Pitch distribution ---
    pitch_counts = Counter(all_notes)
    # Pitch class distribution (C, C#, D, ...)
    pitch_classes = Counter()
    for n in all_notes:
        pc = n.rstrip("0123456789")
        pitch_classes[pc] += 1

    # Pitch class entropy (higher = more varied)
    total_pc = sum(pitch_classes.values())
    pc_entropy = 0.0
    for count in pitch_classes.values():
        p = count / total_pc
        if p > 0:
            pc_entropy -= p * math.log2(p)

    # --- Interval distribution ---
    interval_counts = Counter(all_intervals)
    # What fraction of intervals are "melodic" (within an octave)?
    melodic_intervals = sum(1 for i in all_intervals if abs(i) <= 12)
    melodic_frac = melodic_intervals / max(len(all_intervals), 1)

    # Stepwise motion (intervals of 1-2 semitones) — hallmark of melody
    stepwise = sum(1 for i in all_intervals if abs(i) in (1, 2))
    stepwise_frac = stepwise / max(len(all_intervals), 1)

    # --- Repetition analysis (note bigrams and trigrams) ---
    note_bigrams = Counter()
    note_trigrams = Counter()
    for ch_notes in notes_by_ch.values():
        for i in range(len(ch_notes) - 1):
            note_bigrams[(ch_notes[i], ch_notes[i+1])] += 1
        for i in range(len(ch_notes) - 2):
            note_trigrams[(ch_notes[i], ch_notes[i+1], ch_notes[i+2])] += 1

    # Repetition score: fraction of bigrams that appear more than once
    repeated_bigrams = sum(1 for c in note_bigrams.values() if c > 1)
    bigram_repeat_frac = repeated_bigrams / max(len(note_bigrams), 1)

    repeated_trigrams = sum(1 for c in note_trigrams.values() if c > 1)
    trigram_repeat_frac = repeated_trigrams / max(len(note_trigrams), 1)

    # --- Channel usage ---
    active_channels = len(notes_by_ch)
    notes_per_channel = {ch: len(notes) for ch, notes in notes_by_ch.items()}

    # --- Token composition ---
    note_frac = note_count / max(total_tokens, 1)
    wait_frac = wait_count / max(total_tokens, 1)
    raw_frac = raw_count / max(total_tokens, 1)

    return {
        "file": path.name,
        "total_tokens": total_tokens,
        "note_count": note_count,
        "unique_pitches": len(pitch_counts),
        "pitch_class_entropy": round(pc_entropy, 3),
        "total_intervals": len(all_intervals),
        "melodic_interval_frac": round(melodic_frac, 3),
        "stepwise_motion_frac": round(stepwise_frac, 3),
        "bigram_repeat_frac": round(bigram_repeat_frac, 3),
        "trigram_repeat_frac": round(trigram_repeat_frac, 3),
        "active_channels": active_channels,
        "max_consecutive_waits": max_consecutive_waits,
        "note_frac": round(note_frac, 3),
        "wait_frac": round(wait_frac, 3),
        "raw_frac": round(raw_frac, 3),
    }


_NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

def _note_to_midi(name: str) -> int:
    """Convert note name like 'C4' or 'Eb5' to MIDI number."""
    for i in range(len(name)-1, 0, -1):
        if name[i].isdigit() or (name[i] == '-' and i == len(name)-1):
            pc = name[:i]
            octave = int(name[i:])
            return _NOTE_MAP.get(pc, 0) + (octave + 1) * 12
    return 60


def summarize(results: list[dict], label: str) -> dict:
    """Compute aggregate stats."""
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}

    def avg(key):
        return round(sum(r[key] for r in results) / n, 3)

    return {
        "label": label,
        "n": n,
        "avg_unique_pitches": avg("unique_pitches"),
        "avg_pitch_class_entropy": avg("pitch_class_entropy"),
        "avg_melodic_interval_frac": avg("melodic_interval_frac"),
        "avg_stepwise_motion_frac": avg("stepwise_motion_frac"),
        "avg_bigram_repeat_frac": avg("bigram_repeat_frac"),
        "avg_trigram_repeat_frac": avg("trigram_repeat_frac"),
        "avg_active_channels": avg("active_channels"),
        "avg_max_consecutive_waits": avg("max_consecutive_waits"),
        "avg_note_frac": avg("note_frac"),
        "avg_wait_frac": avg("wait_frac"),
        "avg_raw_frac": avg("raw_frac"),
    }


def main():
    import random

    base = Path(__file__).resolve().parents[1]

    # Generated v2 outputs
    gen_files = sorted(
        list((base / "output" / "v2_sonic").glob("*.vgm"))
        + list((base / "output" / "v2_tfiv").glob("*.vgm"))
    )

    # Real VGM files (sample 50 from the corpus)
    real_dir = base / "data" / "vgm"
    all_real = sorted(list(real_dir.glob("*.vgm")) + list(real_dir.glob("*.vgz")))
    random.seed(42)
    real_sample = random.sample(all_real, min(200, len(all_real)))

    print(f"Analyzing {len(gen_files)} generated + {len(real_sample)} real files...\n")

    gen_results = []
    for f in gen_files:
        r = analyze_file(f)
        if r:
            gen_results.append(r)

    real_results = []
    for f in real_sample:
        r = analyze_file(f)
        if r:
            real_results.append(r)

    gen_summary = summarize(gen_results, "GENERATED (v2)")
    real_summary = summarize(real_results, "REAL (corpus sample)")

    print(f"  Generated: {len(gen_results)}/{len(gen_files)} had note content")
    print(f"  Real: {len(real_results)}/{len(real_sample)} had note content")
    print()

    print("=" * 70)
    print(f"{'Metric':<35} {'REAL':>12} {'GENERATED':>12} {'Gap':>10}")
    print("=" * 70)

    metrics = [
        ("Unique pitches", "avg_unique_pitches", "higher=more varied"),
        ("Pitch class entropy (bits)", "avg_pitch_class_entropy", "3.58=uniform"),
        ("Melodic intervals (<= octave)", "avg_melodic_interval_frac", "higher=better"),
        ("Stepwise motion (1-2 semi)", "avg_stepwise_motion_frac", "higher=melodic"),
        ("Bigram repeat fraction", "avg_bigram_repeat_frac", "higher=structure"),
        ("Trigram repeat fraction", "avg_trigram_repeat_frac", "higher=structure"),
        ("Active FM channels", "avg_active_channels", "1-6"),
        ("Max consecutive waits", "avg_max_consecutive_waits", "lower=better"),
        ("Note token fraction", "avg_note_frac", ""),
        ("Wait token fraction", "avg_wait_frac", ""),
        ("Raw register fraction", "avg_raw_frac", ""),
    ]

    for name, key, note in metrics:
        rv = real_summary.get(key, 0)
        gv = gen_summary.get(key, 0)
        gap = gv - rv if isinstance(rv, (int, float)) and isinstance(gv, (int, float)) else ""
        if isinstance(gap, float):
            gap = f"{gap:+.3f}"
        label = f"{name}"
        if note:
            label += f" [{note}]"
        print(f"{label:<35} {rv:>12} {gv:>12} {gap:>10}")

    print("=" * 70)

    # Interpretation
    print("\nINTERPRETATION:")
    re = real_summary
    ge = gen_summary

    checks = []
    if ge.get("avg_pitch_class_entropy", 0) > 0.8 * re.get("avg_pitch_class_entropy", 1):
        checks.append("✓ Pitch variety is reasonable")
    else:
        checks.append("✗ Pitch variety too low — model stuck on few notes")

    if ge.get("avg_melodic_interval_frac", 0) > 0.6:
        checks.append("✓ Intervals mostly melodic (within octave)")
    else:
        checks.append("✗ Too many large jumps — not melodic")

    if ge.get("avg_stepwise_motion_frac", 0) > 0.15:
        checks.append("✓ Has stepwise motion (melody-like)")
    else:
        checks.append("✗ Lacks stepwise motion — sounds random")

    if ge.get("avg_bigram_repeat_frac", 0) > 0.3 * re.get("avg_bigram_repeat_frac", 1):
        checks.append("✓ Some note pattern repetition (structure)")
    else:
        checks.append("✗ Very little repetition — no structure")

    if ge.get("avg_active_channels", 0) >= 2:
        checks.append("✓ Multiple channels active (polyphony)")
    else:
        checks.append("✗ Too few channels — thin sound")

    for c in checks:
        print(f"  {c}")


if __name__ == "__main__":
    main()
