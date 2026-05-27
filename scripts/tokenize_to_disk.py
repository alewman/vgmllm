"""Tokenize VGM/VGZ files and save a compact per-song binary (.vtok) for
on-disk streaming.

The .vtok format is a minimal streamable binary:

    Bytes  0–3   Magic  b'VTOK'
    Byte   4     Version  0x01
    Bytes  5–6   VOCAB_SIZE as uint16 LE   (sanity / compatibility check)
    Bytes  7–10  Token count N as uint32 LE
    Bytes  11+   Tokens as uint16 LE array  (N × 2 bytes)

VOCAB_SIZE = 898, so every token fits in a uint16 (2 bytes).  A uint10 packing
is also shown in the size report for reference — it saves ~20% vs uint16 at
the cost of non-byte-aligned reads.

Usage examples::

    # Single file — save alongside source, print report
    python scripts/tokenize_to_disk.py data/vgm/MyPack/track.vgz

    # Multiple files
    python scripts/tokenize_to_disk.py data/vgm/SonicPack/*.vgz

    # Whole directory (recursive)
    python scripts/tokenize_to_disk.py --dir data/vgm/MetalSquad/

    # Custom output directory
    python scripts/tokenize_to_disk.py track.vgz --out-dir output/tokens/

    # Summary only — no files written
    python scripts/tokenize_to_disk.py track.vgz --dry-run

    # Read back a .vtok file and print token count + head
    python scripts/tokenize_to_disk.py --inspect output/tokens/track.vtok
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import struct
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis_music.tokenizer_v6 import (
    ComposerMap,
    GameMap,
    TokenizerV6,
    VOCAB_SIZE,
)
from genesis_music.vgm_parser import load_vgm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT  = Path(__file__).parent.parent
_DATA_DIR   = _REPO_ROOT / "data"
_PREP_V6    = _DATA_DIR / "prepared_v6"
_PREP_V4    = _DATA_DIR / "prepared_v4"   # fallback for composer / DAC maps

# ---------------------------------------------------------------------------
# .vtok format constants
# ---------------------------------------------------------------------------

_MAGIC   = b"VTOK"
_VERSION = 0x01
_HEADER_SIZE = 11   # magic(4) + version(1) + vocab(2) + count(4)


# ---------------------------------------------------------------------------
# .vtok read / write
# ---------------------------------------------------------------------------

def save_vtok(tokens: list[int] | np.ndarray, path: Path) -> None:
    """Write tokens to a .vtok file."""
    arr = np.asarray(tokens, dtype=np.uint16)
    n = len(arr)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<B", _VERSION))
        f.write(struct.pack("<H", VOCAB_SIZE))
        f.write(struct.pack("<I", n))
        f.write(arr.tobytes())


def load_vtok(path: Path) -> np.ndarray:
    """Read a .vtok file and return token array as uint16."""
    raw = path.read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"{path}: too short to be a .vtok file ({len(raw)} bytes)")
    magic   = raw[:4]
    version = raw[4]
    vocab   = struct.unpack_from("<H", raw, 5)[0]
    n       = struct.unpack_from("<I", raw, 7)[0]
    if magic != _MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version != _VERSION:
        log.warning("%s: version mismatch (file=%d, code=%d)", path.name, version, _VERSION)
    expected = _HEADER_SIZE + n * 2
    if len(raw) < expected:
        raise ValueError(
            f"{path}: truncated — expected {expected} bytes, got {len(raw)}"
        )
    tokens = np.frombuffer(raw[_HEADER_SIZE:_HEADER_SIZE + n * 2], dtype=np.uint16).copy()
    return tokens


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _packed_bits_size(n_tokens: int) -> int:
    """Bytes needed to pack n_tokens using ceil(log2(VOCAB_SIZE)) bits each."""
    bits_per_token = math.ceil(math.log2(VOCAB_SIZE))   # 10 for VOCAB_SIZE=898
    return _HEADER_SIZE + math.ceil(n_tokens * bits_per_token / 8)


def _vtok_size(n_tokens: int) -> int:
    return _HEADER_SIZE + n_tokens * 2


def _vgm_decompressed_size(path: Path) -> int:
    raw = path.read_bytes()
    if path.suffix.lower() == ".vgz":
        return len(gzip.decompress(raw))
    return len(raw)


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def _load_maps(
    prep_dir: Path,
) -> tuple[ComposerMap | None, GameMap | None, dict[int, int]]:
    composer_map: ComposerMap | None = None
    game_map: GameMap | None = None
    dac_slot_map: dict[int, int] = {}

    cmap_path = prep_dir / "composer_map_v6.json"
    if not cmap_path.exists():
        cmap_path = _PREP_V4 / "composer_map_v4.json"
    if cmap_path.exists():
        composer_map = ComposerMap.load(cmap_path)
        log.info("Loaded composer map (%d composers) from %s", len(composer_map), cmap_path.name)

    gmap_path = prep_dir / "game_map_v6.json"
    if gmap_path.exists():
        game_map = GameMap.load(gmap_path)
        log.info("Loaded game map (%d games) from %s", len(game_map), gmap_path.name)

    dac_path = prep_dir / "dac_slot_map_v6.json"
    if not dac_path.exists():
        dac_path = _PREP_V4 / "dac_slot_map_v4.json"
    if dac_path.exists():
        raw = json.loads(dac_path.read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
        log.info("Loaded DAC slot map (%d entries) from %s", len(dac_slot_map), dac_path.name)

    return composer_map, game_map, dac_slot_map


# ---------------------------------------------------------------------------
# Core: tokenize one file
# ---------------------------------------------------------------------------

def tokenize_file(
    path: Path,
    tokenizer: TokenizerV6,
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> dict | None:
    """
    Tokenize *path* and optionally write a .vtok file.

    Returns a dict with size stats, or None if the file was filtered out.
    """
    try:
        vgm = load_vgm(path)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path.name, exc)
        return None

    tokens = tokenizer.encode(vgm)
    if tokens is None:
        log.info("Filtered (SFX/jingle): %s", path.name)
        return None

    n_tokens = len(tokens)
    disk_size = path.stat().st_size
    vgm_size  = _vgm_decompressed_size(path)
    vtok_size = _vtok_size(n_tokens)
    packed_size = _packed_bits_size(n_tokens)

    out_path: Path | None = None
    if not dry_run:
        dest_dir = out_dir if out_dir else path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / (path.stem + ".vtok")
        save_vtok(tokens, out_path)

    return {
        "name":         path.name,
        "n_tokens":     n_tokens,
        "disk_bytes":   disk_size,
        "vgm_bytes":    vgm_size,
        "vtok_bytes":   vtok_size,
        "packed_bytes": packed_size,
        "out_path":     out_path,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_KB = 1024
_MB = 1024 * _KB

def _fmt(b: int) -> str:
    if b >= _MB:
        return f"{b/_MB:6.2f} MB"
    if b >= _KB:
        return f"{b/_KB:6.1f} KB"
    return f"{b:7d}  B"


def _pct(a: int, b: int) -> str:
    if b == 0:
        return "  n/a"
    return f"{100*a/b:5.1f}%"


def _print_report(results: list[dict]) -> None:
    if not results:
        print("No files processed.")
        return

    bits_per_token = math.ceil(math.log2(VOCAB_SIZE))

    header = (
        f"\n{'File':<40}  {'Tokens':>8}  {'VGZ/VGM':>9}  "
        f"{'Decomp VGM':>10}  {'vtok (u16)':>10}  {'Packed(%db)':>12}  "
        f"{'vtok/VGZ':>9}  {'packed/VGZ':>10}"
    )
    header = header.replace('%d', str(bits_per_token))
    print(header)
    print("-" * len(header))

    total_disk = total_vgm = total_vtok = total_packed = total_tokens = 0

    for r in results:
        name = r["name"]
        if len(name) > 38:
            name = "…" + name[-37:]
        print(
            f"  {name:<38}  {r['n_tokens']:>8,}  "
            f"{_fmt(r['disk_bytes']):>9}  "
            f"{_fmt(r['vgm_bytes']):>10}  "
            f"{_fmt(r['vtok_bytes']):>10}  "
            f"{_fmt(r['packed_bytes']):>12}  "
            f"{_pct(r['vtok_bytes'], r['disk_bytes']):>9}  "
            f"{_pct(r['packed_bytes'], r['disk_bytes']):>10}"
        )
        total_disk    += r["disk_bytes"]
        total_vgm     += r["vgm_bytes"]
        total_vtok    += r["vtok_bytes"]
        total_packed  += r["packed_bytes"]
        total_tokens  += r["n_tokens"]

    if len(results) > 1:
        print("-" * len(header))
        print(
            f"  {'TOTAL':<38}  {total_tokens:>8,}  "
            f"{_fmt(total_disk):>9}  "
            f"{_fmt(total_vgm):>10}  "
            f"{_fmt(total_vtok):>10}  "
            f"{_fmt(total_packed):>12}  "
            f"{_pct(total_vtok, total_disk):>9}  "
            f"{_pct(total_packed, total_disk):>10}"
        )

    print(
        f"\n  VOCAB_SIZE = {VOCAB_SIZE}  =>  {bits_per_token} bits/token minimum\n"
        f"  vtok format:  uint16 (2 bytes/token, {bits_per_token+6} bits used)\n"
        f"  packed format: {bits_per_token} bits/token -- theoretical minimum, shown for reference\n"
    )


# ---------------------------------------------------------------------------
# --inspect mode
# ---------------------------------------------------------------------------

def inspect_vtok(path: Path) -> None:
    tokens = load_vtok(path)
    n = len(tokens)
    file_size = path.stat().st_size
    print(f"\n{path.name}")
    print(f"  File size:   {_fmt(file_size)}")
    print(f"  Token count: {n:,}")
    print(f"  Bytes/token: {file_size / n:.2f}" if n else "  (empty)")
    print(f"  First 20 tokens: {tokens[:20].tolist()}")
    # Rough token type breakdown
    from genesis_music.tokenizer_v6 import (
        BOS, EOS, PAD, BAR, BEAT_BASE, NOTE_ON, NOTE_OFF,
        PITCH_BASE, NUM_PITCHES,
    )
    count_bos  = int(np.sum(tokens == BOS))
    count_eos  = int(np.sum(tokens == EOS))
    count_bars = int(np.sum(tokens == BAR))
    count_notes = int(np.sum((tokens >= NOTE_ON) & (tokens <= NOTE_OFF)))
    count_pitches = int(np.sum((tokens >= PITCH_BASE) & (tokens < PITCH_BASE + NUM_PITCHES)))
    count_beats = int(np.sum((tokens >= BEAT_BASE) & (tokens < BEAT_BASE + 16)))
    print(f"  BOS={count_bos}  EOS={count_eos}  BAR={count_bars}  "
          f"NOTE_ON/OFF={count_notes}  PITCH={count_pitches}  BEAT={count_beats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize VGM/VGZ → .vtok (streamable per-song token binary)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="VGM/VGZ files to tokenize (or .vtok files when --inspect is set)",
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="Recursively tokenize all VGM/VGZ files in this directory",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for .vtok files (default: same dir as source)",
    )
    parser.add_argument(
        "--prep-dir", type=Path, default=_PREP_V6,
        help=f"Path to prepared_v6 dir with map JSONs (default: {_PREP_V6})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report sizes only — do not write .vtok files",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect .vtok files instead of tokenizing",
    )
    args = parser.parse_args()

    # ---- inspect mode ----
    if args.inspect:
        targets = list(args.files)
        if not targets:
            parser.error("Provide .vtok files to inspect")
        for p in targets:
            try:
                inspect_vtok(p)
            except Exception as exc:
                log.error("%s: %s", p, exc)
        return

    # ---- collect input files ----
    input_files: list[Path] = list(args.files)
    if args.dir:
        input_files += sorted(args.dir.rglob("*.vgm")) + sorted(args.dir.rglob("*.vgz"))

    if not input_files:
        parser.error("No input files.  Pass VGM/VGZ files, or use --dir.")

    log.info("Found %d file(s) to tokenize", len(input_files))

    # ---- load maps ----
    composer_map, game_map, dac_slot_map = _load_maps(args.prep_dir)
    tokenizer = TokenizerV6(
        composer_map=composer_map,
        game_map=game_map,
        dac_slot_map=dac_slot_map,
    )

    # ---- process files ----
    results: list[dict] = []
    n_filtered = 0
    n_errors   = 0
    for path in input_files:
        result = tokenize_file(
            path,
            tokenizer,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
        if result is None:
            n_filtered += 1
        else:
            results.append(result)
            if not args.dry_run and result["out_path"]:
                log.info("Wrote %s  (%d tokens)",
                         result["out_path"].name, result["n_tokens"])

    # ---- print report ----
    _print_report(results)

    if n_filtered:
        print(f"  {n_filtered} file(s) filtered out (SFX / jingles / parse errors)")
    if n_errors:
        print(f"  {n_errors} file(s) failed with errors")


if __name__ == "__main__":
    main()
