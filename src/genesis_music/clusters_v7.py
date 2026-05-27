"""v7 hardware-cluster classifier for Genesis VGM files.

Assigns each VGM file to one of 5 hardware-usage clusters based on
rule-based feature thresholds.  These cluster IDs are used by the v7
dataset pipeline for rare-cluster oversampling and by the trainer for
per-cluster validation loss logging.

Cluster definitions
-------------------
1 – Standard FM:        Mostly FM synthesis, low/no DAC streaming, simple PSG.
2 – Heavy PSG:          Significant PSG presence, low FM complexity.
3 – DAC-dominant:       PCM/DAC streaming heavily used (e.g. vocal samples).
4 – Complex envelope:   Mid-song patch changes and/or SSG-EG usage.
5 – Stereo / Special:   Non-default pan writes, LFO, CH3 special mode.

Feature vector (8 dimensions)
------------------------------
  0  pan_writes_per_sec      — YM2612 reg 0xB4-0xB6 writes / song duration
  1  max_feedback             — maximum feedback value 0-7 across all channels
  2  instrument_change_rate   — mid-song patch changes per second
  3  dac_fraction             — fraction of notes on CH_DAC vs total notes
  4  psg_fraction             — fraction of notes on PSG channels
  5  ssg_eg_usage             — any non-zero SSG-EG field present (0.0 or 1.0)
  6  lfo_active_fraction      — fraction of song duration with LFO active
  7  ch3_special_fraction     — fraction of song duration in CH3 special mode

Usage
-----
CLI:
    python -m genesis_music.clusters_v7 --vgm-dir data/vgm --out data/clusters_v7.json

Library:
    from genesis_music.clusters_v7 import extract_features, classify_song
    features = extract_features(vgm)
    cluster  = classify_song(features)   # 1-5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .vgm_parser import EventType, VgmFile, VgmHeader, load_vgm
from .ym2612 import CH_DAC, CH_PSG_0, CH_PSG_NOISE, decode_vgm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "pan_writes_per_sec",
    "max_feedback",
    "instrument_change_rate",
    "dac_fraction",
    "psg_fraction",
    "ssg_eg_usage",
    "lfo_active_fraction",
    "ch3_special_fraction",
)
N_FEATURES = len(FEATURE_NAMES)  # 8


def extract_features(vgm: VgmFile) -> np.ndarray:
    """Extract an 8-dimensional hardware-usage feature vector from a VgmFile.

    Returns a float32 array of shape (8,).  Safe on pathological files
    (empty event streams, zero-duration, etc.) — always returns finite values.
    """
    total_samples = max(1, getattr(vgm.header, 'total_samples', 0))
    duration_sec  = total_samples / 44100.0

    # -----------------------------------------------------------------------
    # Scan YM2612 register events
    # -----------------------------------------------------------------------
    pan_writes       = 0
    max_feedback     = 0
    lfo_on_samples   = 0   # accumulated samples where LFO is active
    ch3_special_samp = 0   # accumulated samples in CH3 special mode
    ssg_eg_nonzero   = False

    # For LFO/CH3 tracking: remember the sample_pos when each state was entered
    lfo_active      = False
    lfo_start_samp  = 0
    ch3_special     = False
    ch3_start_samp  = 0

    prev_sample_pos = 0

    for e in vgm.events:
        if e.type == EventType.WAIT:
            # Accumulate duration while in active states
            if lfo_active:
                lfo_on_samples += e.value
            if ch3_special:
                ch3_special_samp += e.value
            prev_sample_pos += e.value
            continue

        if e.type not in (EventType.YM2612_PORT0, EventType.YM2612_PORT1):
            continue

        reg = e.register
        val = e.value

        # Pan writes (regs 0xB4-0xB6, both ports)
        if 0xB4 <= reg <= 0xB6:
            pan_bits = (val >> 6) & 0x03
            if pan_bits != 3:   # non-center pan → interesting
                pan_writes += 1

        # Feedback (regs 0xB0-0xB2, bits 5:3)
        elif 0xB0 <= reg <= 0xB2:
            fb = (val >> 3) & 0x07
            if fb > max_feedback:
                max_feedback = fb

        # LFO control (reg 0x22, port 0)
        elif e.type == EventType.YM2612_PORT0 and reg == 0x22:
            now_active = bool(val & 0x08)
            if now_active and not lfo_active:
                lfo_active    = True
                lfo_start_samp = prev_sample_pos
            elif not now_active and lfo_active:
                lfo_on_samples += prev_sample_pos - lfo_start_samp
                lfo_active = False

        # CH3 mode (reg 0x27, port 0)
        elif e.type == EventType.YM2612_PORT0 and reg == 0x27:
            now_special = bool(val & 0x40)
            if now_special and not ch3_special:
                ch3_special    = True
                ch3_start_samp = prev_sample_pos
            elif not now_special and ch3_special:
                ch3_special_samp += prev_sample_pos - ch3_start_samp
                ch3_special = False

        # SSG-EG (regs 0x90-0x9F)
        elif 0x90 <= reg <= 0x9F and not ssg_eg_nonzero:
            if val & 0x0F:
                ssg_eg_nonzero = True

    # Close open LFO / CH3 windows
    if lfo_active:
        lfo_on_samples += total_samples - lfo_start_samp
    if ch3_special:
        ch3_special_samp += total_samples - ch3_start_samp

    # -----------------------------------------------------------------------
    # Note events for channel fractions and patch-change rate
    # -----------------------------------------------------------------------
    try:
        note_events, _patches = decode_vgm(vgm)
    except Exception:
        note_events = []

    n_total  = max(1, len(note_events))
    n_dac    = sum(1 for n in note_events if n.channel == CH_DAC)
    n_psg    = sum(1 for n in note_events
                   if CH_PSG_0 <= n.channel <= CH_PSG_NOISE)

    # Instrument change rate: count distinct (ch, fingerprint) pairs beyond first
    channel_fps: dict[int, set] = {}
    patch_changes = 0
    for n in sorted(note_events, key=lambda x: x.sample_on):
        if 0 <= n.channel <= 5 and n.patch is not None:
            fp = n.patch.to_fingerprint()
            seen = channel_fps.setdefault(n.channel, set())
            if fp not in seen:
                if seen:   # not first patch for this channel
                    patch_changes += 1
                seen.add(fp)

    # -----------------------------------------------------------------------
    # Assemble feature vector
    # -----------------------------------------------------------------------
    features = np.array([
        pan_writes / duration_sec,                             # 0 pan_writes_per_sec
        float(max_feedback),                                   # 1 max_feedback
        patch_changes / duration_sec,                          # 2 instrument_change_rate
        n_dac  / n_total,                                      # 3 dac_fraction
        n_psg  / n_total,                                      # 4 psg_fraction
        1.0 if ssg_eg_nonzero else 0.0,                        # 5 ssg_eg_usage
        lfo_on_samples / total_samples,                        # 6 lfo_active_fraction
        ch3_special_samp / total_samples,                      # 7 ch3_special_fraction
    ], dtype=np.float32)

    return features


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

def classify_song(features: np.ndarray) -> int:
    """Classify an 8-dim feature vector into cluster 1-5.

    Rule priority (first match wins):
      5 → Stereo/Special: heavy non-center pan OR LFO OR CH3 special
      4 → Complex envelope: frequent patch changes OR SSG-EG present
      3 → DAC-dominant: DAC fraction > 0.4
      2 → Heavy PSG: PSG fraction > 0.35
      1 → Standard FM: everything else
    """
    f = features
    pan_rate         = float(f[0])   # writes/sec
    max_fb           = float(f[1])
    ic_rate          = float(f[2])   # patch changes/sec
    dac_frac         = float(f[3])
    psg_frac         = float(f[4])
    ssg_eg           = float(f[5])
    lfo_frac         = float(f[6])
    ch3_frac         = float(f[7])

    # Cluster 5: significant non-default pan usage, active LFO, or CH3 special
    if pan_rate > 1.0 or lfo_frac > 0.10 or ch3_frac > 0.05:
        return 5

    # Cluster 4: noticeable instrument switching or SSG-EG looping envelopes
    if ic_rate > 0.5 or ssg_eg > 0.0:
        return 4

    # Cluster 3: DAC streaming is the dominant channel
    if dac_frac > 0.40:
        return 3

    # Cluster 2: heavy PSG, relatively light FM
    if psg_frac > 0.35:
        return 2

    # Cluster 1: standard FM
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _scan_corpus(
    vgm_dir: Path,
    out_path: Path,
    max_files: int | None = None,
) -> None:
    """Scan all VGM/VGZ files in *vgm_dir* and write a cluster JSON."""
    files = sorted(vgm_dir.rglob("*.vgm")) + sorted(vgm_dir.rglob("*.vgz"))
    if max_files:
        files = files[:max_files]
    log.info("Scanning %d VGM files in %s …", len(files), vgm_dir)

    results: dict[str, int] = {}
    cluster_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for i, path in enumerate(files):
        try:
            vgm = load_vgm(path)
            feat    = extract_features(vgm)
            cluster = classify_song(feat)
            results[str(path.resolve())] = cluster
            cluster_counts[cluster] += 1
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
        if (i + 1) % 500 == 0:
            log.info("  %d/%d processed", i + 1, len(files))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_counts": cluster_counts,
        "files": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Cluster distribution: %s", cluster_counts)
    log.info("Saved: %s (%d entries)", out_path, len(results))


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Classify VGM corpus into 5 hardware-usage clusters."
    )
    parser.add_argument("--vgm-dir",  required=True, type=Path,
                        help="Directory containing .vgm/.vgz files")
    parser.add_argument("--out",      required=True, type=Path,
                        help="Output JSON path (e.g. data/clusters_v7.json)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap number of files scanned (for testing)")
    args = parser.parse_args()
    _scan_corpus(args.vgm_dir, args.out, args.max_files)


if __name__ == "__main__":
    _main()
