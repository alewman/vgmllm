"""Musical analysis of decoded VGM note events.

Provides:
  - Tempo detection via autocorrelation of note-onset intervals
  - Key / mode detection via Krumhansl-Kessler pitch-class profiles
  - Channel role classification (BASS / LEAD / HARM / DRUMS / PERC / UNK)
  - Corpus-level filtering helpers (SFX / jingle detection)

All functions operate on lists of NoteEvent (from ym2612.py) and return
plain Python values suitable for embedding in token sequences.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .ym2612 import (
    CH_DAC,
    CH_FM_0,
    CH_PSG_0,
    CH_PSG_NOISE,
    NoteEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44_100  # VGM sample rate (fixed)

# Krumhansl-Kessler pitch-class salience profiles
# Index 0 = C, 1 = C#, 2 = D, …, 11 = B
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# BPM bins used in the v4 token vocabulary
TEMPO_BINS = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240]

# Key names in chromatic order (index 0=C)
KEY_NAMES_MAJOR = ["C",  "C#", "D",  "D#", "E",  "F",
                   "F#", "G",  "G#", "A",  "A#", "B"]
KEY_NAMES_MINOR = ["Cm", "C#m","Dm", "D#m","Em", "Fm",
                   "F#m","Gm", "G#m","Am", "A#m","Bm"]

# Role names
ROLE_BASS    = "BASS"
ROLE_LEAD    = "LEAD"
ROLE_HARM    = "HARM"
ROLE_COUNTER = "COUNTER"
ROLE_DRUMS   = "DRUMS"
ROLE_PERC    = "PERC"
ROLE_UNK     = "UNK"


# ---------------------------------------------------------------------------
# Data class for analysis results
# ---------------------------------------------------------------------------

@dataclass
class MusicAnalysis:
    """Results of analysing a decoded VGM file."""
    tempo_bpm: float                    # detected BPM (or 120.0 fallback)
    tempo_token_idx: int                # index into TEMPO_BINS
    key_index: int                      # 0–11 = C–B
    is_minor: bool                      # True = minor, False = major
    key_name: str                       # e.g. "Am", "C#"
    meter_numerator: int = 4            # time signature numerator
    meter_denominator: int = 4          # time signature denominator
    channel_roles: dict[int, str] = None   # channel index → role string

    def __post_init__(self):
        if self.channel_roles is None:
            self.channel_roles = {}


# ---------------------------------------------------------------------------
# Tempo detection
# ---------------------------------------------------------------------------

def detect_tempo(
    note_events: list[NoteEvent],
    total_samples: int,
    bpm_min: float = 50.0,
    bpm_max: float = 300.0,
) -> float:
    """Estimate the dominant BPM from note-onset times.

    Uses autocorrelation of the note-onset histogram (one bin per 10 ms).
    Returns 120.0 if detection is unreliable (too few events, no clear peak).
    """
    onsets = sorted(
        e.sample_on for e in note_events
        if e.channel < CH_DAC and e.sample_on >= 0
    )
    if len(onsets) < 8:
        return 120.0

    # Build onset histogram at ~10 ms resolution (441 samples per bin)
    bin_size = 441
    duration_bins = max(1, total_samples // bin_size + 1)
    onset_hist = np.zeros(duration_bins, dtype=np.float32)
    for s in onsets:
        idx = min(s // bin_size, duration_bins - 1)
        onset_hist[idx] += 1.0

    # Autocorrelation
    acf = np.correlate(onset_hist, onset_hist, mode="full")
    acf = acf[len(acf) // 2:]   # keep non-negative lags

    # Convert BPM range to lag range (in bins)
    min_lag = max(1, int(SAMPLE_RATE * 60.0 / bpm_max / bin_size))
    max_lag = min(len(acf) - 1, int(SAMPLE_RATE * 60.0 / bpm_min / bin_size))

    if min_lag >= max_lag:
        return 120.0

    search = acf[min_lag:max_lag + 1]
    if search.max() == 0:
        return 120.0

    beat_lag_bins = min_lag + int(search.argmax())
    beat_samples  = beat_lag_bins * bin_size
    bpm           = 60.0 * SAMPLE_RATE / beat_samples

    # Clamp to plausible range
    return float(np.clip(bpm, bpm_min, bpm_max))


def quantize_tempo(bpm: float) -> tuple[float, int]:
    """Snap a BPM to the nearest TEMPO_BIN.

    Returns (snapped_bpm, token_index).
    """
    diffs = [abs(bpm - b) for b in TEMPO_BINS]
    idx   = int(np.argmin(diffs))
    return float(TEMPO_BINS[idx]), idx


# ---------------------------------------------------------------------------
# Key / mode detection
# ---------------------------------------------------------------------------

def detect_key(
    note_events: list[NoteEvent],
) -> tuple[int, bool, str]:
    """Detect the musical key using Krumhansl-Kessler profiles.

    Returns (key_index 0-11, is_minor, key_name).
    key_index 0 = C, 1 = C#/Db, …, 11 = B.
    """
    # Build pitch-class histogram from FM note events
    pc_hist = np.zeros(12, dtype=np.float64)
    for e in note_events:
        if e.pitch >= 0 and e.channel < CH_PSG_0:
            pc = e.pitch % 12
            # Weight by duration if available, else 1
            dur = e.duration_samples if e.is_closed else 735
            pc_hist[pc] += max(1, dur)

    if pc_hist.sum() == 0:
        return 0, False, "C"

    pc_hist /= pc_hist.sum()

    # Correlate against all 24 rotations
    best_score = -math.inf
    best_key   = 0
    best_minor = False

    for root in range(12):
        major_profile = np.roll(_KK_MAJOR, root)
        minor_profile = np.roll(_KK_MINOR, root)

        score_major = float(np.corrcoef(pc_hist, major_profile)[0, 1])
        score_minor = float(np.corrcoef(pc_hist, minor_profile)[0, 1])

        if score_major > best_score:
            best_score = score_major
            best_key   = root
            best_minor = False

        if score_minor > best_score:
            best_score = score_minor
            best_key   = root
            best_minor = True

    name = KEY_NAMES_MINOR[best_key] if best_minor else KEY_NAMES_MAJOR[best_key]
    return best_key, best_minor, name


# ---------------------------------------------------------------------------
# Channel role classification
# ---------------------------------------------------------------------------

def _channel_stats(
    events: list[NoteEvent],
) -> dict[int, dict]:
    """Compute per-channel statistics used for role classification."""
    by_channel: dict[int, list[NoteEvent]] = defaultdict(list)
    for e in events:
        by_channel[e.channel].append(e)

    stats = {}
    for ch, evs in by_channel.items():
        pitches = [e.pitch for e in evs if e.pitch >= 0]
        if not pitches:
            stats[ch] = {
                "count": len(evs),
                "mean_pitch": -1,
                "pitch_std": 0.0,
                "note_density": 0.0,
                "has_pitch": False,
            }
            continue

        durations = [e.sample_on for e in evs]
        if len(durations) > 1:
            total_span = max(durations) - min(durations)
            density    = len(evs) / max(1, total_span / SAMPLE_RATE)
        else:
            density = 0.0

        stats[ch] = {
            "count":       len(evs),
            "mean_pitch":  float(np.mean(pitches)),
            "pitch_std":   float(np.std(pitches)),
            "note_density": density,   # notes per second
            "has_pitch":   True,
        }
    return stats


def classify_channel_roles(
    note_events: list[NoteEvent],
    dac_enabled_channels: set[int] | None = None,
) -> dict[int, str]:
    """Assign a musical role to each active channel.

    Heuristics (in priority order):
      1. DAC channel (CH_DAC / CH6)                   → DRUMS
      2. PSG noise channel (CH_PSG_NOISE)             → PERC
      3. Mean pitch < MIDI 48 (C3)                    → BASS
      4. Mean pitch ≥ MIDI 60 (C5) AND density > 3/s  → LEAD
      5. Mean pitch ≥ MIDI 60 (C5) AND density ≤ 3/s  → COUNTER
      6. Otherwise                                     → HARM
      7. Channel with no pitch data                    → UNK
    """
    if dac_enabled_channels is None:
        dac_enabled_channels = set()

    stats  = _channel_stats(note_events)
    roles: dict[int, str] = {}

    for ch, s in stats.items():
        if ch == CH_DAC or ch in dac_enabled_channels:
            roles[ch] = ROLE_DRUMS
        elif ch == CH_PSG_NOISE:
            roles[ch] = ROLE_PERC
        elif not s["has_pitch"]:
            roles[ch] = ROLE_UNK
        elif s["mean_pitch"] < 48:
            roles[ch] = ROLE_BASS
        elif s["mean_pitch"] >= 60 and s["note_density"] >= 3.0:
            roles[ch] = ROLE_LEAD
        elif s["mean_pitch"] >= 60:
            roles[ch] = ROLE_COUNTER
        else:
            roles[ch] = ROLE_HARM

    return roles


# ---------------------------------------------------------------------------
# Corpus filtering
# ---------------------------------------------------------------------------

def should_discard(
    note_events: list[NoteEvent],
    total_samples: int,
    min_duration_s: float = 8.0,
    min_fm_channels: int = 2,
    min_unique_pitches: int = 5,
) -> tuple[bool, str]:
    """Return (True, reason) if this VGM should be excluded from training.

    Filters:
    - Too short (SFX, jingles)
    - Fewer than min_fm_channels active FM voices (single-channel drone/SFX)
    - Fewer than min_unique_pitches distinct pitches (note-variety check)
    """
    duration_s = total_samples / SAMPLE_RATE

    if duration_s < min_duration_s:
        return True, f"too short ({duration_s:.1f}s < {min_duration_s}s)"

    fm_channels_active = len({
        e.channel for e in note_events
        if 0 <= e.channel < 6 and e.pitch >= 0
    })
    if fm_channels_active < min_fm_channels:
        return (
            True,
            f"too few FM channels ({fm_channels_active} < {min_fm_channels})",
        )

    unique_pitches = len({
        e.pitch for e in note_events if e.pitch >= 0 and e.channel < 6
    })
    if unique_pitches < min_unique_pitches:
        return (
            True,
            f"too few unique pitches ({unique_pitches} < {min_unique_pitches})",
        )

    return False, ""


# ---------------------------------------------------------------------------
# Top-level analysis entry point
# ---------------------------------------------------------------------------

def analyse_vgm(
    note_events: list[NoteEvent],
    total_samples: int,
) -> MusicAnalysis:
    """Run all analysis passes and return a MusicAnalysis result.

    This is the single call used by the tokenizer.
    """
    bpm = detect_tempo(note_events, total_samples)
    _, tempo_idx = quantize_tempo(bpm)

    key_idx, is_minor, key_name = detect_key(note_events)

    # Detect DAC-enabled channels from NoteEvent list
    dac_channels = {e.channel for e in note_events if e.channel == CH_DAC}
    roles = classify_channel_roles(note_events, dac_channels)

    return MusicAnalysis(
        tempo_bpm       = bpm,
        tempo_token_idx = tempo_idx,
        key_index       = key_idx,
        is_minor        = is_minor,
        key_name        = key_name,
        channel_roles   = roles,
    )
