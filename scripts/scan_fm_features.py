#!/usr/bin/env python3
"""scan_fm_features.py — Scan the VGM corpus for advanced YM2612 feature usage.

Checks each file for:
  - LFO enable  (reg 0x22, bits 3:0 = LFO freq, bit 3 = enable)
  - AMS / PMS   (regs 0xB4-0xB6, bits 5:4 = AMS, bits 2:0 = PMS)
  - SSG-EG      (regs 0x90-0x97, bit 3 = SSG-EG enable)
  - FM3 special mode  (reg 0x27, bits 7:6 — CH3 mode: 0=normal, 2=special)
  - DAC enable  (reg 0x2B, bit 7)
  - SN PSG noise/tone usage
  - Feedback > 0 and Algorithm variety

Reports per-feature usage counts and percentages across the corpus,
then lists the top examples of each feature for sampling/analysis.

Usage:
  cd d:\\dev\\genesis-music-ml
  python scripts/scan_fm_features.py [--vgm-dir data/vgm] [--limit N] [--top N]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from genesis_music.vgm_parser import load_vgm, EventType   # noqa: E402


# ── register constants ─────────────────────────────────────────────────────────
REG_LFO         = 0x22   # global LFO enable + freq
REG_CH3_MODE    = 0x27   # timer / CH3 mode
REG_DAC_ENABLE  = 0x2B   # DAC enable (bit 7)
REG_FNUM_LO     = range(0xA0, 0xA6)   # port 0+1 Fnum low
# SSG-EG: 0x90-0x97 (port 0), 0x90-0x97 (port 1) — bit 3 = SSG-EG on
REG_SSGEG_BASE  = 0x90
REG_SSGEG_END   = 0x98
# AMS/PMS: 0xB4-0xB6 (port 0), 0xB4-0xB6 (port 1)
REG_AMPMS_BASE  = 0xB4
REG_AMPMS_END   = 0xB7
# Algorithm / Feedback: 0xB0-0xB2
REG_ALGFB_BASE  = 0xB0
REG_ALGFB_END   = 0xB3


class FileFeatures(NamedTuple):
    path:           Path
    has_lfo:        bool   # 0x22 with bit 3 set and LFO freq > 0
    max_lfo_freq:   int    # 0-7
    has_pms:        bool   # any channel PMS > 0
    has_ams:        bool   # any channel AMS > 0
    has_ssgeg:      bool   # any operator SSG-EG enabled
    has_ch3_special:bool   # CH3 CSM or special mode
    has_dac:        bool   # DAC enabled at any point
    algorithms:     frozenset[int]
    feedbacks:      frozenset[int]
    has_sn_noise:   bool   # SN noise channel used
    n_ym_events:    int


def scan_file(path: Path) -> FileFeatures | None:
    try:
        vgm = load_vgm(path)
    except Exception:
        return None

    if not vgm.header.has_ym2612:
        return None

    has_lfo         = False
    max_lfo_freq    = 0
    has_pms         = False
    has_ams         = False
    has_ssgeg       = False
    has_ch3_special = False
    has_dac         = False
    has_sn_noise    = False
    algorithms:  set[int] = set()
    feedbacks:   set[int] = set()
    n_ym = 0

    for ev in vgm.events:
        t = ev.type
        r = ev.register
        v = ev.value

        if t in (EventType.YM2612_PORT0, EventType.YM2612_PORT1):
            n_ym += 1

            # LFO
            if r == REG_LFO:
                if v & 0x08:           # enable bit
                    has_lfo = True
                    max_lfo_freq = max(max_lfo_freq, v & 0x07)

            # CH3 special / CSM mode
            elif r == REG_CH3_MODE:
                mode = (v >> 6) & 0x03
                if mode in (1, 2, 3):  # 0=normal, 1=CSM, 2/3=special freq
                    has_ch3_special = True

            # DAC enable
            elif r == REG_DAC_ENABLE:
                if v & 0x80:
                    has_dac = True

            # SSG-EG
            elif REG_SSGEG_BASE <= r < REG_SSGEG_END:
                if v & 0x08:           # bit 3 = SSG-EG enable
                    has_ssgeg = True

            # AMS / PMS  (0xB4-0xB6)
            elif REG_AMPMS_BASE <= r < REG_AMPMS_END:
                pms = v & 0x07
                ams = (v >> 4) & 0x03
                if pms > 0:
                    has_pms = True
                if ams > 0:
                    has_ams = True

            # Algorithm + Feedback  (0xB0-0xB2)
            elif REG_ALGFB_BASE <= r < REG_ALGFB_END:
                algorithms.add(v & 0x07)
                feedbacks.add((v >> 3) & 0x07)

        elif t == EventType.SN76489:
            # SN noise channel: data byte with bits 7:4 == 0b1110 (0xE?) or 0xF?
            # Noise channel selected when bit 7=1, bit 5=1 (LATCH+DATA for channel 3)
            # Simplified: any SN byte with bits 7:6 = 11 and bits 5:4 = 10 → noise
            if (v & 0xE0) == 0xE0:
                has_sn_noise = True

    return FileFeatures(
        path            = path,
        has_lfo         = has_lfo,
        max_lfo_freq    = max_lfo_freq,
        has_pms         = has_pms,
        has_ams         = has_ams,
        has_ssgeg       = has_ssgeg,
        has_ch3_special = has_ch3_special,
        has_dac         = has_dac,
        algorithms      = frozenset(algorithms),
        feedbacks       = frozenset(feedbacks),
        has_sn_noise    = has_sn_noise,
        n_ym_events     = n_ym,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan VGM corpus for advanced YM2612 feature usage.")
    parser.add_argument("--vgm-dir", type=Path, default=_ROOT / "data" / "vgm", dest="vgm_dir")
    parser.add_argument("--limit",   type=int,  default=0,
                        help="Max files to scan (0 = all)")
    parser.add_argument("--top",     type=int,  default=10,
                        help="Number of top examples to list per feature")
    args = parser.parse_args()

    vgm_files = sorted(args.vgm_dir.rglob("*.vg[mz]"))
    if args.limit:
        vgm_files = vgm_files[:args.limit]

    total = len(vgm_files)
    print(f"Scanning {total} VGM/VGZ files in {args.vgm_dir} …\n")

    results: list[FileFeatures] = []
    errors = 0
    for i, p in enumerate(vgm_files, 1):
        if i % 1000 == 0:
            print(f"  {i}/{total} …", flush=True)
        f = scan_file(p)
        if f is None:
            errors += 1
        else:
            results.append(f)

    n = len(results)
    print(f"\nParsed {n} files  ({errors} errors / skipped)\n")
    print("=" * 70)
    print("YM2612 ADVANCED FEATURE USAGE REPORT")
    print("=" * 70)

    def pct(count: int) -> str:
        return f"{count:5d} / {n}  ({count/n*100:5.1f}%)"

    # ── feature counts ─────────────────────────────────────────────────────────
    lfo_files    = [f for f in results if f.has_lfo]
    pms_files    = [f for f in results if f.has_pms]
    ams_files    = [f for f in results if f.has_ams]
    ssgeg_files  = [f for f in results if f.has_ssgeg]
    ch3sp_files  = [f for f in results if f.has_ch3_special]
    dac_files    = [f for f in results if f.has_dac]
    snnoise_files= [f for f in results if f.has_sn_noise]

    print(f"\n  LFO enabled (0x22 bit3)          : {pct(len(lfo_files))}")
    print(f"  PMS > 0 (pitch vibrato sens)     : {pct(len(pms_files))}")
    print(f"  AMS > 0 (amplitude tremolo sens) : {pct(len(ams_files))}")
    print(f"  SSG-EG enabled (metallic env)    : {pct(len(ssgeg_files))}")
    print(f"  CH3 special/CSM mode             : {pct(len(ch3sp_files))}")
    print(f"  DAC enabled                      : {pct(len(dac_files))}")
    print(f"  SN noise channel used            : {pct(len(snnoise_files))}")

    # ── LFO freq distribution ──────────────────────────────────────────────────
    if lfo_files:
        freq_counts: dict[int, int] = defaultdict(int)
        for f in lfo_files:
            freq_counts[f.max_lfo_freq] += 1
        # LFO freq 0-7 → 3.98/5.56/6.02/6.37/6.88/9.40/11.75/23.64 Hz
        lfo_hz = [3.98, 5.56, 6.02, 6.37, 6.88, 9.40, 11.75, 23.64]
        print("\n  LFO frequency distribution (among LFO-enabled tracks):")
        for freq, count in sorted(freq_counts.items()):
            hz = lfo_hz[freq] if freq < len(lfo_hz) else "?"
            bar = "█" * (count * 30 // len(lfo_files))
            print(f"    freq={freq} ({hz:5.2f} Hz) : {count:4d}  {bar}")

    # ── algorithm distribution ─────────────────────────────────────────────────
    alg_counts: dict[int, int] = defaultdict(int)
    for f in results:
        for a in f.algorithms:
            alg_counts[a] += 1
    print("\n  Algorithm usage (files containing each ALG):")
    for alg in range(8):
        count = alg_counts.get(alg, 0)
        print(f"    ALG {alg} : {pct(count)}")

    # ── feedback distribution ──────────────────────────────────────────────────
    fb_counts: dict[int, int] = defaultdict(int)
    for f in results:
        for fb in f.feedbacks:
            fb_counts[fb] += 1
    print("\n  Feedback usage (files containing each FB level):")
    for fb in range(8):
        count = fb_counts.get(fb, 0)
        print(f"    FB  {fb} : {pct(count)}")

    # ── top examples per feature ───────────────────────────────────────────────
    top = args.top

    def _show_top(label: str, subset: list[FileFeatures]) -> None:
        if not subset:
            return
        print(f"\n  Top {min(top, len(subset))} examples — {label}:")
        for f in subset[:top]:
            rel = f.path.relative_to(args.vgm_dir)
            print(f"    {rel}")

    print(f"\n{'─'*70}")
    print(f"TOP {top} EXAMPLES PER FEATURE  (relative to {args.vgm_dir})")
    print(f"{'─'*70}")

    _show_top("LFO enabled",               lfo_files[:top])
    _show_top("SSG-EG enabled",            ssgeg_files[:top])
    _show_top("CH3 special/CSM mode",      ch3sp_files[:top])
    _show_top("PMS > 0 (pitch vibrato)",   pms_files[:top])
    _show_top("AMS > 0 (amplitude trem.)", ams_files[:top])

    # ── tokenizer coverage checklist ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOKENIZER COVERAGE CHECKLIST  (v6 gaps to fix in v7)")
    print(f"{'='*70}")
    features = [
        ("LFO global enable (reg 0x22)",          len(lfo_files),    n),
        ("PMS sensitivity (regs 0xB4-0xB6 b2:0)", len(pms_files),    n),
        ("AMS sensitivity (regs 0xB4-0xB6 b5:4)", len(ams_files),    n),
        ("SSG-EG mode (regs 0x90-0x97 b3)",       len(ssgeg_files),  n),
        ("CH3 special mode (reg 0x27 b7:6)",       len(ch3sp_files),  n),
    ]
    for name, count, total_n in features:
        priority = "HIGH" if count/total_n > 0.10 else ("MED" if count/total_n > 0.03 else "LOW")
        print(f"  [{priority:4s}] {name}")
        print(f"         {count} files ({count/total_n*100:.1f}%) — "
              f"{'likely missing from v6 token stream' if priority != 'LOW' else 'rare, lower priority'}")

    print()


if __name__ == "__main__":
    main()
