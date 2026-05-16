"""Round-trip a VGM through the v6 tokenizer (lossless FM patch encoding).

Encodes the file exactly as the v6 dataset pipeline will, then decodes and
synthesises back to VGM so you can hear what the model will be trained on.

Usage:
    python scripts/hear_training_data_v6.py <path_to_vgm_or_vgz> [--output out.vgm]

Compare the output against scripts/hear_training_data.py (v4) to hear the
improvement from direct patch encoding vs 128-entry library lookup.
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
from genesis_music.tokenizer_v6 import TokenizerV6, ComposerMap, VOCAB_SIZE
from genesis_music.vgm_synth import synthesise_vgm

DATA_DIR = Path(__file__).parent.parent / "data"
PREPARED  = DATA_DIR / "prepared_v4"   # reuse v4 support files (same format)


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip a VGM through the v6 tokenizer (lossless FM patches)"
    )
    parser.add_argument("vgm", type=Path, help="Input VGM/VGZ file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output VGM path (default: <stem>_v6_roundtrip.vgm in output/roundtrip/)")
    args = parser.parse_args()

    # Optional: reuse v4 composer map (same token IDs)
    composer_map = None
    cmap_path = PREPARED / "composer_map_v4.json"
    if cmap_path.exists():
        composer_map = ComposerMap.load(cmap_path)
        log.info("Loaded composer map: %d composers", len(composer_map))

    # Reuse v4 DAC slot map
    dac_slot_map: dict[int, int] = {}
    dac_path = PREPARED / "dac_slot_map_v4.json"
    if dac_path.exists():
        raw = json.loads(dac_path.read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}

    # Drum kit for synthesis
    drum_kit: dict[int, bytes] | None = None
    drum_kit_path = PREPARED / "dac_drum_kit_v4.json"
    if drum_kit_path.exists():
        raw_kit = json.loads(drum_kit_path.read_text())
        drum_kit = {int(k): bytes.fromhex(v) for k, v in raw_kit.items()}
        log.info("Loaded drum kit: %d slots", len(drum_kit))

    tokenizer = TokenizerV6(composer_map=composer_map, dac_slot_map=dac_slot_map)
    log.info("TokenizerV6 ready (vocab size %d)", VOCAB_SIZE)

    # Encode
    log.info("Loading: %s", args.vgm)
    vgm = load_vgm(args.vgm)
    tokens = tokenizer.encode(vgm)
    if tokens is None:
        log.error("File was filtered out (too short / too few FM channels). "
                  "Re-run with --skip-filter if you want to force it.")
        sys.exit(1)
    log.info("Encoded to %d tokens", len(tokens))

    # Decode
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

    # Synthesise — pass direct patch dict as patch_map
    patch_map = header.get("channel_patches_direct", {})
    vgm_bytes = synthesise_vgm(note_events, total_samples, patch_map, drum_kit=drum_kit)

    # Save
    if args.output is None:
        out_dir = Path(__file__).parent.parent / "output" / "roundtrip"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / (args.vgm.stem + "_v6_roundtrip.vgm")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_bytes(vgm_bytes)
    log.info("Saved: %s (%.1f KB)", args.output, len(vgm_bytes) / 1024)


if __name__ == "__main__":
    main()
