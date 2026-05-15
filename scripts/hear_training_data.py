"""Encode a VGM file exactly as the v4 dataset pipeline does, then decode it
back to VGM so you can hear what the model is actually trained on.

Usage:
    python scripts/hear_training_data.py <path_to_vgm_or_vgz> [--output out.vgm]
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
from genesis_music.tokenizer_v4 import TokenizerV4, PatchLibrary
from genesis_music.vgm_synth import synthesise_vgm

DATA_DIR = Path(__file__).parent.parent / "data"
PREPARED  = DATA_DIR / "prepared_v4"


def main():
    parser = argparse.ArgumentParser(description="Round-trip a VGM through the v4 tokenizer")
    parser.add_argument("vgm", type=Path, help="Input VGM/VGZ file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output VGM path (default: <input_stem>_roundtrip.vgm in output/roundtrip/)")
    args = parser.parse_args()

    # Load supporting data
    patch_lib = PatchLibrary.load(DATA_DIR / "patch_library_v4.json")
    dac_slot_map_path = PREPARED / "dac_slot_map_v4.json"
    dac_slot_map = {}
    if dac_slot_map_path.exists():
        raw = json.loads(dac_slot_map_path.read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
    drum_kit = None
    drum_kit_path = PREPARED / "dac_drum_kit_v4.json"
    if drum_kit_path.exists():
        raw_kit = json.loads(drum_kit_path.read_text())
        drum_kit = {int(k): bytes.fromhex(v) for k, v in raw_kit.items()}
        log.info("Loaded drum kit: %d slots", len(drum_kit))

    tokenizer = TokenizerV4(patch_lib, dac_slot_map=dac_slot_map)

    # Load and encode
    log.info("Loading: %s", args.vgm)
    vgm = load_vgm(args.vgm)
    tokens = tokenizer.encode(vgm)
    log.info("Encoded to %d tokens (vocab size 456)", len(tokens))

    # Decode back
    note_events, patch_map = tokenizer.decode(tokens)
    log.info("Decoded to %d NoteEvents", len(note_events))

    if not note_events:
        log.warning("No NoteEvents decoded — track may have been filtered (e.g. too much DAC)")
        sys.exit(1)

    total_samples = max(
        (e.sample_off if e.sample_off >= 0 else e.sample_on) for e in note_events
    )
    total_samples = max(total_samples + 44100, 44100)
    log.info("Duration: %.1f seconds", total_samples / 44100)

    # Synthesise
    vgm_bytes = synthesise_vgm(note_events, total_samples, patch_map, drum_kit=drum_kit)

    # Save
    if args.output is None:
        out_dir = Path(__file__).parent.parent / "output" / "roundtrip"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / (args.vgm.stem + "_roundtrip.vgm")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_bytes(vgm_bytes)
    log.info("Saved: %s (%.1f KB)", args.output, len(vgm_bytes) / 1024)


if __name__ == "__main__":
    main()
