"""Round-trip a VGM through the v7 tokenizer (SSG-EG + hardware state tokens).

Encodes the file exactly as the v7 dataset pipeline will, then decodes and
synthesises back to VGM so you can hear what the model will be trained on.

New vs v6:
  - 44-token FM patch headers (adds SSG-EG per operator)
  - Hardware state tokens (PAN, LFO, CH3 special mode, DAC enable) inline
  - PSG vol_envelope and FM tl_envelope note-splitting
  - FM6 reclassification (CH_DAC + FM patch → CH_FM_5)

Usage:
    python scripts/hear_training_data_v7.py <path_to_vgm_or_vgz> [--output out.vgm]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis_music.vgm_parser import load_vgm
from genesis_music.tokenizer_v7 import TokenizerV7, ComposerMap, GameMap, VOCAB_SIZE_V7
from genesis_music.vgm_synth import synthesise_vgm

DATA_DIR  = Path(__file__).parent.parent / "data"
PREPARED_V7 = DATA_DIR / "prepared_v7"
PREPARED_V6 = DATA_DIR / "prepared_v6"   # fallback for maps
PREPARED_V4 = DATA_DIR / "prepared_v4"   # last-resort fallback


def _find_map(name: str, *dirs: Path) -> Path | None:
    for d in dirs:
        for suffix in (f"{name}_v7.json", f"{name}_v6.json", f"{name}_v4.json"):
            p = d / suffix
            if p.exists():
                return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip a VGM through the v7 tokenizer"
    )
    parser.add_argument("vgm", type=Path, help="Input VGM/VGZ file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output VGM path (default: <stem>_v7_roundtrip.vgm in output/roundtrip/)")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Bypass quality filter (useful for single-file debugging)")
    args = parser.parse_args()

    # --- Composer map ---
    composer_map: ComposerMap | None = None
    cmap_path = _find_map("composer_map", PREPARED_V7, PREPARED_V6, PREPARED_V4)
    if cmap_path:
        composer_map = ComposerMap.load(cmap_path)
        log.info("Loaded composer map from %s (%d composers)", cmap_path.name,
                 len(composer_map))

    # --- Game map ---
    game_map: GameMap | None = None
    gmap_path = _find_map("game_map", PREPARED_V7, PREPARED_V6, PREPARED_V4)
    if gmap_path:
        game_map = GameMap.load(gmap_path)
        log.info("Loaded game map from %s (%d games)", gmap_path.name, len(game_map))

    # --- DAC slot map ---
    dac_slot_map: dict[int, int] = {}
    dac_path = _find_map("dac_slot_map", PREPARED_V7, PREPARED_V6, PREPARED_V4)
    if dac_path:
        raw = json.loads(dac_path.read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
        log.info("Loaded DAC slot map from %s (%d slots)", dac_path.name,
                 len(dac_slot_map))

    # --- Drum kit ---
    drum_kit: dict[int, bytes] | None = None
    kit_path = _find_map("dac_drum_kit", PREPARED_V7, PREPARED_V6, PREPARED_V4)
    if kit_path:
        raw_kit = json.loads(kit_path.read_text())
        drum_kit = {int(k): bytes.fromhex(v) for k, v in raw_kit.items()}
        log.info("Loaded drum kit (%d slots)", len(drum_kit))

    tokenizer = TokenizerV7(
        composer_map=composer_map,
        game_map=game_map,
        dac_slot_map=dac_slot_map,
    )
    log.info("TokenizerV7 ready (vocab size %d)", VOCAB_SIZE_V7)

    # --- Encode ---
    log.info("Loading: %s", args.vgm)
    vgm = load_vgm(args.vgm)
    tokens = tokenizer.encode(vgm, skip_filter=args.skip_filter)
    if tokens is None:
        log.error("File was filtered out. Re-run with --skip-filter to force encode.")
        sys.exit(1)
    log.info("Encoded to %d tokens", len(tokens))

    # Report hardware state token counts
    from genesis_music.tokenizer_v7 import (
        PAN_OFF, PAN_CENTER, LFO_OFF, LFO_ON_BASE, CH3_SPECIAL_MODE,
        DAC_ENABLE, LOOP_POINT, INSTRUMENT_CHANGE,
    )
    hw_counts = {
        "pan":              sum(1 for t in tokens if PAN_OFF <= t <= PAN_CENTER),
        "lfo":              sum(1 for t in tokens if t == LFO_ON_BASE or (LFO_ON_BASE <= t <= LFO_ON_BASE + 7)),
        "ch3_special":      sum(1 for t in tokens if t == CH3_SPECIAL_MODE),
        "dac_enable":       sum(1 for t in tokens if t == DAC_ENABLE),
        "loop_point":       sum(1 for t in tokens if t == LOOP_POINT),
        "instrument_change":sum(1 for t in tokens if t == INSTRUMENT_CHANGE),
    }
    for name, count in hw_counts.items():
        if count:
            log.info("  HW tokens %-20s: %d", name, count)

    # --- Decode ---
    note_events, header = tokenizer.decode(tokens)
    log.info("Decoded to %d NoteEvents", len(note_events))

    n_patches = len(header.get("channel_patches_direct", {}))
    log.info("Patches recovered: %d FM channels (lossless)", n_patches)

    if not note_events:
        log.warning("No NoteEvents decoded.")
        sys.exit(1)

    total_samples = max(
        (e.sample_off if e.sample_off >= 0 else e.sample_on) for e in note_events
    )
    total_samples = max(total_samples + 44100, 44100)
    log.info("Duration: %.1f seconds", total_samples / 44100)

    # --- Synthesise ---
    patch_map = header.get("channel_patches_direct", {})
    vgm_bytes = synthesise_vgm(note_events, total_samples, patch_map, drum_kit=drum_kit)

    # --- Save ---
    if args.output is None:
        out_dir = Path(__file__).parent.parent / "output" / "roundtrip"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / (args.vgm.stem + "_v7_roundtrip.vgm")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_bytes(vgm_bytes)
    log.info("Saved: %s (%.1f KB)", args.output, len(vgm_bytes) / 1024)


if __name__ == "__main__":
    main()
