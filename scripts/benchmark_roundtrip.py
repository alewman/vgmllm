#!/usr/bin/env python3
"""benchmark_roundtrip.py — Tokenizer fidelity benchmark for landmark tracks.

For each of 29 curated top-tier Genesis tracks, this script:
  1. Decodes the original VGZ via the NoteEvent pipeline (ground truth)
  2. Runs a v4 (patch-library) roundtrip
  3. Runs a v6 (lossless FM) roundtrip
  4. Saves all three as .vgm files so you can listen in a VGM player
  5. Saves a 3-row comparison piano-roll PNG (original / v4 / v6)
  6. Prints a summary table with token counts, note retention %, duration delta

Output layout
-------------
  output/benchmark/
    {slug}/
      original.vgm
      v4_roundtrip.vgm
      v6_roundtrip.vgm
      comparison.png
    summary.txt

Usage
-----
  cd d:\\dev\\genesis-music-ml
  python scripts/benchmark_roundtrip.py
  python scripts/benchmark_roundtrip.py --only "Streets_of_Rage_2"
  python scripts/benchmark_roundtrip.py --no-audio   # PNGs + summary only, skip .vgm write
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm, CH_DAC, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE
from genesis_music.tokenizer_v4 import TokenizerV4, PatchLibrary
from genesis_music.tokenizer_v6 import TokenizerV6, ComposerMap, VOCAB_SIZE as V6_VOCAB_SIZE
from genesis_music.vgm_synth import synthesise_vgm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SAMPLE_RATE = 44_100

DATA_DIR   = _ROOT / "data"
PREPARED   = DATA_DIR / "prepared_v4"
VGM_DIR    = DATA_DIR / "vgm"
OUT_DIR    = _ROOT / "output" / "benchmark"

# ─── Landmark track list ───────────────────────────────────────────────────────
# (game_hint, track_hint, human_label)
# game_hint  : substring of the vgz filename (case-insensitive)
# track_hint : secondary filter on filename; None = pick the best numbered track
#              (skips common logo/jingle tracks 01 when hint provided)
LANDMARK_TRACKS: list[tuple[str, str | None, str]] = [
    ("Streets_of_Rage_2",               "Go_Straight",             "Streets of Rage 2 — Go Straight"),
    ("Sonic_the_Hedgehog_3",            "Ice_Cap",                 "Sonic 3 — Ice Cap Zone Act 1"),
    ("Thunder_Force_IV",                "Lightning_Strikes_Again", "Thunder Force IV — Lightning Strikes Again"),
    ("Revenge_of_Shinobi",              "Shinobi",                 "Revenge of Shinobi — The Shinobi"),
    ("Sonic_the_Hedgehog_2",            "Chemical_Plant",          "Sonic 2 — Chemical Plant Zone"),
    ("Gunstar_Heroes",                  "Empire",                  "Gunstar Heroes — Empire"),
    ("Ristar",                          "Star_Humming",            "Ristar — Star Humming"),
    ("Phantasy_Star_II__English",       "Place_of_Death",          "Phantasy Star II — Place of Death"),
    ("Streets_of_Rage__Bare_Knuckle__", "Street_of_Rage",          "Streets of Rage — The Street of Rage"),
    ("Ecco_the_Dolphin__Mega",          "Undercaves",              "Ecco the Dolphin — Undercaves"),
    ("Bloodlines",                      "Reincarnated",            "Castlevania: Bloodlines — Reincarnated Soul"),
    ("Shinobi_III",                     "Whirlwind",               "Shinobi III — Whirlwind"),
    ("Dragon_s_Fury",                   "Opening",                 "Dragon's Fury — Opening Theme"),
    ("Thunder_Force_III",               "Back_to_the_Fire",        "Thunder Force III — Back to the Fire"),
    ("Elemental_Master",                "Mountain_of_Doom",        "Elemental Master — Stage 1"),
    ("Alisia_Dragoon",                  "Stage_7",                 "Alisia Dragoon — Stage 7"),
    ("Contra_Hard_Corps",               "Overdrive",               "Contra Hard Corps — Contra Overdrive"),
    ("Sub-Terrania",                    "First_Floor",             "Sub-Terrania — First Floor Power"),
    ("Red_Zone",                        "Death",                   "Red Zone — Death by Drain"),
    ("Golden_Axe_II",                   "Battle",                  "Golden Axe II — Battle Field"),
    ("Rocket_Knight_Adventures",        "Kingdom",                 "Rocket Knight Adventures — Kingdom of Zebulos"),
    ("Alien_Soldier",                   "Runner",                  "Alien Soldier — Runner AD2025"),
    ("Vectorman__Mega",                 "Day_1",                   "Vectorman — Day 1 / Hydroponic Zone"),
    ("Beyond_Oasis",                    "Water_Shrine",            "Beyond Oasis — Water Shrine"),
    ("Comix_Zone__Mega",                "Welcome",                 "Comix Zone — Welcome to the Temple"),
    ("Gaiares",                         "Stage_3",                 "Gaiares — Stage 3"),
    ("Mega_Turrican",                   "Stage_1",                 "Mega Turrican — Stage 1-1"),
    ("Pulseman",                        "Shinjuku",                "Pulseman — Shinjuku"),
    ("Adventures_of_Batman",            "Amused",                  "Batman & Robin — Amused to Death"),
]

# ─── Channel colours (shared with vgm_piano_roll.py) ─────────────────────────
CHANNEL_INFO = {
    0: ("FM 1", "#e05c5c"), 1: ("FM 2", "#e08c3a"), 2: ("FM 3", "#d4c830"),
    3: ("FM 4", "#5cbf5c"), 4: ("FM 5", "#3ab0d4"), 5: ("FM 6", "#7b6fe0"),
    CH_DAC:       ("DAC",   "#888888"),
    CH_PSG_0:     ("PSG 1", "#d47bbf"), CH_PSG_1: ("PSG 2", "#bf9fd4"),
    CH_PSG_2:     ("PSG 3", "#94d4bf"), CH_PSG_NOISE: ("Noise", "#aaaaaa"),
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_vgz(hint: str, track_hint: str | None) -> Path | None:
    """Glob VGM_DIR for files matching hint (and optionally track_hint)."""
    pattern = re.compile(re.escape(hint), re.IGNORECASE)
    candidates = [p for p in VGM_DIR.rglob("*.vgz") if pattern.search(p.name)]
    if not candidates:
        return None
    if track_hint:
        th_pat = re.compile(re.escape(track_hint), re.IGNORECASE)
        filtered = [p for p in candidates if th_pat.search(p.name)]
        if filtered:
            return sorted(filtered)[0]
    # Fall back: prefer tracks numbered 02+ (skip logo tracks) if many candidates
    numbered = sorted(candidates)
    # skip tracks named *_01_* if there are others
    non_intro = [p for p in numbered if not re.search(r"_01_", p.name)]
    return (non_intro[0] if non_intro else numbered[0])


def notes_from_vgm(path: Path):
    vgm = load_vgm(path)
    notes, _ = decode_vgm(vgm)
    total_samples = vgm.header.total_samples
    return notes, total_samples, vgm


def render_roll(notes, total_samples: int, ax, title: str) -> None:
    """Draw a piano roll into ax (matches vgm_piano_roll.py style)."""
    total_sec = total_samples / SAMPLE_RATE
    pitched = [n for n in notes if n.pitch >= 0 and n.channel != CH_DAC]
    dac     = [n for n in notes if n.channel == CH_DAC]
    noise   = [n for n in notes if n.channel == CH_PSG_NOISE]

    if pitched:
        p_min = max(0,   min(n.pitch for n in pitched) - 2)
        p_max = min(127, max(n.pitch for n in pitched) + 2)
    else:
        p_min, p_max = 36, 84

    ax.set_facecolor("#1a1a2e")
    for oct_ in range(11):
        y = oct_ * 12
        if p_min <= y <= p_max:
            ax.axhline(y, color="#2a2a3e", linewidth=0.4, zorder=0)

    for note in pitched:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else total_sec
        dur   = max(t_off - t_on, 0.02)
        vel_a = 0.55 + 0.45 * (note.velocity / 15) if note.velocity > 0 else 0.65
        color = CHANNEL_INFO.get(note.channel, ("?", "#ffffff"))[1]
        ax.add_patch(mpatches.FancyBboxPatch(
            (t_on, note.pitch - 0.4), dur, 0.8,
            boxstyle="round,pad=0.01", linewidth=0, facecolor=color, alpha=vel_a, zorder=2,
        ))

    dac_y = p_min - 2.5
    for note in dac:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else t_on + 0.05
        ax.add_patch(mpatches.FancyBboxPatch(
            (t_on, dac_y - 0.45), max(t_off - t_on, 0.02), 0.9,
            boxstyle="round,pad=0.01", linewidth=0, facecolor="#888888", alpha=0.75, zorder=2,
        ))

    noise_y = p_min - 4.5
    for note in noise:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else t_on + 0.05
        ax.add_patch(mpatches.FancyBboxPatch(
            (t_on, noise_y - 0.45), max(t_off - t_on, 0.02), 0.9,
            boxstyle="round,pad=0.01", linewidth=0, facecolor="#aaaaaa", alpha=0.6, zorder=2,
        ))

    present = sorted(set(n.channel for n in notes))
    ax.legend(
        handles=[mpatches.Patch(color=CHANNEL_INFO.get(c, ("?", "#fff"))[1],
                                label=CHANNEL_INFO.get(c, (f"ch{c}", "#fff"))[0])
                 for c in present],
        loc="upper right", fontsize=6, framealpha=0.25,
        facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#cccccc",
    )
    y_bot = (noise_y if noise else dac_y if dac else p_min) - 1.5
    ax.set_xlim(0, total_sec)
    ax.set_ylim(y_bot, p_max + 1)
    ax.tick_params(colors="#cccccc", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.set_title(title, color="#eeeeff", fontsize=9, pad=4)
    ax.set_ylabel("MIDI pitch", color="#999999", fontsize=7)


class TrackResult(NamedTuple):
    label: str
    source: Path
    v4_tokens: int
    v6_tokens: int
    orig_notes: int
    v4_notes: int
    v6_notes: int
    orig_dur: float
    v4_dur: float
    v6_dur: float
    v4_ok: bool
    v6_ok: bool


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenizer fidelity benchmark")
    ap.add_argument("--only", metavar="HINT",
                    help="Run only tracks whose label/hint contains this string")
    ap.add_argument("--no-audio", action="store_true",
                    help="Skip writing .vgm output files (PNGs + summary only)")
    args = ap.parse_args()

    # ── Load shared support data ──────────────────────────────────────────────
    patch_lib = PatchLibrary.load(DATA_DIR / "patch_library_v4.json")

    dac_slot_map: dict[int, int] = {}
    dsl_path = PREPARED / "dac_slot_map_v4.json"
    if dsl_path.exists():
        dac_slot_map = {int(k): int(v) for k, v in json.loads(dsl_path.read_text()).items()}

    drum_kit: dict[int, bytes] | None = None
    dk_path = PREPARED / "dac_drum_kit_v4.json"
    if dk_path.exists():
        drum_kit = {int(k): bytes.fromhex(v) for k, v in json.loads(dk_path.read_text()).items()}
        log.info("Drum kit: %d slots", len(drum_kit))

    composer_map: ComposerMap | None = None
    cm_path = PREPARED / "composer_map_v4.json"
    if cm_path.exists():
        composer_map = ComposerMap.load(cm_path)

    tok_v4 = TokenizerV4(patch_lib, dac_slot_map=dac_slot_map)
    tok_v6 = TokenizerV6(composer_map=composer_map, dac_slot_map=dac_slot_map)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[TrackResult] = []

    for game_hint, track_hint, label in LANDMARK_TRACKS:
        if args.only and args.only.lower() not in (game_hint + label).lower():
            continue

        src = find_vgz(game_hint, track_hint)
        if src is None:
            log.warning("NOT FOUND: %s  (hint=%s, track=%s)", label, game_hint, track_hint)
            continue

        log.info("─── %s", label)
        log.info("    Source: %s", src.name)

        slug = re.sub(r"[^\w]+", "_", label).strip("_")
        out_track = OUT_DIR / slug
        out_track.mkdir(parents=True, exist_ok=True)

        # ── original ──────────────────────────────────────────────────────────
        try:
            orig_notes, orig_samples, orig_vgm = notes_from_vgm(src)
            orig_dur = orig_samples / SAMPLE_RATE
            n_orig = len([n for n in orig_notes if n.pitch >= 0 and n.channel != CH_DAC])
        except Exception as exc:
            log.error("Failed to decode original: %s", exc)
            continue

        if not args.no_audio:
            import shutil
            shutil.copy(src, out_track / "original.vgz")

        # ── v4 roundtrip ──────────────────────────────────────────────────────
        v4_tokens, v4_notes_list, v4_patch_map, v4_ok = 0, [], {}, False
        try:
            tokens_v4 = tok_v4.encode(orig_vgm)
            v4_tokens = len(tokens_v4)
            note_events_v4, v4_patch_map = tok_v4.decode(tokens_v4)
            v4_notes_list = note_events_v4
            n_v4 = len([n for n in v4_notes_list if n.pitch >= 0 and n.channel != CH_DAC])
            v4_dur_s = max((e.sample_off if e.sample_off >= 0 else e.sample_on)
                           for e in v4_notes_list) / SAMPLE_RATE if v4_notes_list else 0.0
            v4_ok = True
            log.info("    v4: %d tokens → %d pitched notes  (%.1f s)", v4_tokens, n_v4, v4_dur_s)
            if not args.no_audio and v4_notes_list:
                v4_total = int(v4_dur_s * SAMPLE_RATE) + SAMPLE_RATE
                vgm_bytes = synthesise_vgm(v4_notes_list, v4_total, v4_patch_map, drum_kit=drum_kit)
                (out_track / "v4_roundtrip.vgm").write_bytes(vgm_bytes)
        except Exception as exc:
            log.warning("    v4 failed: %s", exc)
            n_v4, v4_dur_s = 0, 0.0

        # ── v6 roundtrip ──────────────────────────────────────────────────────
        v6_tokens, v6_notes_list, v6_patch_map, v6_ok = 0, [], {}, False
        try:
            tokens_v6 = tok_v6.encode(orig_vgm)
            if tokens_v6 is None:
                log.warning("    v6: filtered out (too short / too few FM channels)")
                n_v6, v6_dur_s = 0, 0.0
            else:
                v6_tokens = len(tokens_v6)
                note_events_v6, v6_header = tok_v6.decode(tokens_v6)
                v6_notes_list = note_events_v6
                v6_patch_map = v6_header.get("channel_patches_direct", {})
                n_v6 = len([n for n in v6_notes_list if n.pitch >= 0 and n.channel != CH_DAC])
                v6_dur_s = max((e.sample_off if e.sample_off >= 0 else e.sample_on)
                               for e in v6_notes_list) / SAMPLE_RATE if v6_notes_list else 0.0
                v6_ok = True
                log.info("    v6: %d tokens → %d pitched notes  (%.1f s)", v6_tokens, n_v6, v6_dur_s)
                if not args.no_audio and v6_notes_list:
                    v6_total = int(v6_dur_s * SAMPLE_RATE) + SAMPLE_RATE
                    vgm_bytes = synthesise_vgm(v6_notes_list, v6_total, v6_patch_map, drum_kit=drum_kit)
                    (out_track / "v6_roundtrip.vgm").write_bytes(vgm_bytes)
        except Exception as exc:
            log.warning("    v6 failed: %s", exc)
            n_v6, v6_dur_s = 0, 0.0

        # ── piano roll comparison PNG ─────────────────────────────────────────
        try:
            fig, axes = plt.subplots(3, 1, figsize=(24, 12),
                                     facecolor="#111122", sharex=False)
            fig.suptitle(label, color="#eeeeff", fontsize=12, y=0.98)

            render_roll(orig_notes, orig_samples, axes[0],
                        f"ORIGINAL  ({n_orig} pitched notes, {orig_dur:.1f}s)")
            if v4_ok and v4_notes_list:
                v4_samp = int(v4_dur_s * SAMPLE_RATE)
                ret_v4 = 100 * n_v4 / n_orig if n_orig else 0
                render_roll(v4_notes_list, v4_samp, axes[1],
                            f"v4 ROUNDTRIP  {v4_tokens} tokens → {n_v4} notes "
                            f"({ret_v4:.0f}% retention, {v4_dur_s:.1f}s)")
            else:
                axes[1].set_facecolor("#1a1a2e")
                axes[1].text(0.5, 0.5, "v4 FAILED", color="#ff6666",
                             ha="center", va="center", transform=axes[1].transAxes, fontsize=14)
                axes[1].set_title("v4 ROUNDTRIP — FAILED", color="#ff6666", fontsize=9)

            if v6_ok and v6_notes_list:
                v6_samp = int(v6_dur_s * SAMPLE_RATE)
                ret_v6 = 100 * n_v6 / n_orig if n_orig else 0
                render_roll(v6_notes_list, v6_samp, axes[2],
                            f"v6 ROUNDTRIP  {v6_tokens} tokens → {n_v6} notes "
                            f"({ret_v6:.0f}% retention, {v6_dur_s:.1f}s)")
            else:
                axes[2].set_facecolor("#1a1a2e")
                axes[2].text(0.5, 0.5, "v6 FAILED", color="#ff6666",
                             ha="center", va="center", transform=axes[2].transAxes, fontsize=14)
                axes[2].set_title("v6 ROUNDTRIP — FAILED", color="#ff6666", fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            png_path = out_track / "comparison.png"
            fig.savefig(png_path, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            log.info("    PNG: %s", png_path)
        except Exception as exc:
            log.warning("    PNG render failed: %s", exc)

        results.append(TrackResult(
            label=label, source=src,
            v4_tokens=v4_tokens, v6_tokens=v6_tokens,
            orig_notes=n_orig, v4_notes=n_v4, v6_notes=n_v6,
            orig_dur=orig_dur, v4_dur=v4_dur_s, v6_dur=v6_dur_s,
            v4_ok=v4_ok, v6_ok=v6_ok,
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    if not results:
        log.warning("No tracks processed.")
        return

    header = (f"{'Label':<52} {'Orig':>5} {'v4tok':>6} {'v6tok':>6} "
              f"{'v4ret%':>7} {'v6ret%':>7} {'v4dur':>6} {'v6dur':>6}")
    sep = "─" * len(header)

    lines = [sep, header, sep]
    for r in results:
        ret4 = f"{100 * r.v4_notes / r.orig_notes:.0f}%" if r.orig_notes else "—"
        ret6 = f"{100 * r.v6_notes / r.orig_notes:.0f}%" if r.orig_notes else "—"
        dur4 = f"{r.v4_dur:.1f}s" if r.v4_ok else "FAIL"
        dur6 = f"{r.v6_dur:.1f}s" if r.v6_ok else "FAIL"
        lines.append(
            f"{r.label:<52} {r.orig_notes:>5} {r.v4_tokens:>6} {r.v6_tokens:>6} "
            f"{ret4:>7} {ret6:>7} {dur4:>6} {dur6:>6}"
        )
    lines.append(sep)

    summary = "\n".join(lines)
    print("\n" + summary + "\n")
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(summary + "\n")
    log.info("Summary written → %s", summary_path)
    log.info("All outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
