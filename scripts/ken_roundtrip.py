"""
Round-trip Ken's Stage through the v6 tokenizer pipeline:
  VGM file → decode_vgm → NoteEvents
           → TokenizerV6.encode → tokens
           → TokenizerV6.decode → NoteEvents + header
           → synthesise_vgm     → VGM file

The output shows exactly what the model was trained on for this song —
the quantized, patch-reconstructed version of Ken's Stage.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add src to path
repo = Path(__file__).parents[1]
sys.path.insert(0, str(repo / "src"))

from genesis_music.tokenizer_v6 import TokenizerV6, GameMap
from genesis_music.vgm_parser import load_vgm
from genesis_music.vgm_synth import synthesise_vgm

INPUT_VGM  = repo / "data/vgm/Street_Fighter_II__-_Special_Champion_Edition__Mega_Drive__Genesis___11_-_Ken_s_Theme.vgz"
GAME_MAP   = repo / "data/prepared_v6/game_map_v6.json"
DAC_MAP    = repo / "data/prepared_v6/dac_slot_map_v6.json"
OUTPUT_DIR = repo / "output/v6_ken_25k_cont"
OUTPUT_VGM = OUTPUT_DIR / "ken_roundtrip.vgm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build tokenizer
import json
game_map = GameMap(json.loads(GAME_MAP.read_text()).get("games", []))
dac_slot_map = json.loads(DAC_MAP.read_text()) if DAC_MAP.exists() else None
tokenizer = TokenizerV6(game_map=game_map, dac_slot_map=dac_slot_map)

# Encode
log.info("Encoding %s ...", INPUT_VGM.name)
vgm = load_vgm(INPUT_VGM)
tokens = tokenizer.encode(vgm)
if tokens is None:
    log.error("Encoding returned None — file was filtered out")
    sys.exit(1)
log.info("Encoded to %d tokens", len(tokens))

# Decode
log.info("Decoding tokens back to NoteEvents ...")
note_events, header = tokenizer.decode(tokens)
log.info("Decoded to %d NoteEvents", len(note_events))
log.info("Header: tempo=%s  key=%s  loop=%s  channels=%s",
         header.get("tempo_bpm"), header.get("key"),
         header.get("loop_present"),
         list(header.get("channel_patches_direct", {}).keys()))

# Compute duration
if note_events:
    total_samples = max(
        (e.sample_off if e.sample_off >= 0 else e.sample_on)
        for e in note_events
    )
    total_samples = max(total_samples + 44100, 44100)
else:
    total_samples = 44100 * 10

duration = total_samples / 44100.0
log.info("Duration: %.1f seconds", duration)

# Synthesise
patch_map = header.get("channel_patches_direct", {})
log.info("Synthesising VGM ...")
vgm_bytes = synthesise_vgm(note_events, total_samples, patch_map)
OUTPUT_VGM.write_bytes(vgm_bytes)
log.info("Saved: %s (%.1f KB)", OUTPUT_VGM, len(vgm_bytes) / 1024)
