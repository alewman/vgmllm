"""v6 dataset preparation — tokenize VGM corpus with augmentation.

Pipeline:
    1. Build / load ComposerMap  (fast GD3 header scan — no full parse)
    2. Build / load GameMap      (fast GD3 header scan — no full parse)
    3. Build / load DAC slot map (onset scan — same as v4)
    4. Fast GD3 game-name scan → assign each file to train / val / val_pack
       (split at FILE level, before augmentation, to prevent leakage)
    5. Encode each file with TokenizerV6 (filters SFX/jingles)
       — train files: augmented ×12 keys
       — val / val_pack files: base key only
    6. Save train.npy / val.npy / val_pack.npy + meta.json

No PatchLibrary is required — v6 encodes FM patches losslessly as direct
parameter tokens, so the expensive first-pass patch scan is eliminated.

Typical usage::

    python -m genesis_music.dataset_v6 \\
        --vgm-dir  data/vgm \\
        --out-dir  data/prepared_v6

Once prepared, training uses VgmDatasetV6 (or VgmDatasetV4 — same format).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .tokenizer_v6 import (
    PAD, DAC_HIT_BASE,
    ComposerMap, GameMap, TokenizerV6,
    VOCAB_SIZE, NUM_GAMES, NUM_DAC_SLOTS, UNK_GAME,
    _fast_read_gd3_fields,
)
from .vgm_parser import load_vgm

log = logging.getLogger(__name__)

# Files whose DAC-hit tokens exceed this fraction are treated as sample-based
# melodic tracks and excluded from FM training data (same threshold as v4).
_MAX_DAC_FRACTION: float = 0.50


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


class VgmDatasetV6(Dataset):
    """PyTorch Dataset of fixed-length v6 token windows.

    Same interface as VgmDatasetV4.  Each item is a dict:
        input_ids:  (seq_len,) int64
        labels:     (seq_len,) int64  — shifted by 1, PAD replaced by -100
    """

    def __init__(self, tokens: np.ndarray, seq_len: int = 16384) -> None:
        self.seq_len = seq_len
        chunk_size   = seq_len + 1
        n_chunks     = len(tokens) // chunk_size
        if n_chunks == 0:
            raise ValueError(
                f"Token array too short ({len(tokens)}) for seq_len={seq_len}"
            )
        self._tokens     = tokens
        self._chunk_size = chunk_size
        self._n_chunks   = n_chunks
        log.info(
            "VgmDatasetV6: %d chunks × %d tokens  (%.1fM total)",
            n_chunks, seq_len, n_chunks * seq_len / 1e6,
        )

    def __len__(self) -> int:
        return self._n_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start     = idx * self._chunk_size
        chunk     = torch.from_numpy(
            self._tokens[start : start + self._chunk_size].astype(np.int64)
        )
        input_ids = chunk[:-1]
        labels    = chunk[1:].clone()
        labels[labels == PAD] = -100
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Parallel worker helpers (module-level for Windows spawn)
# ---------------------------------------------------------------------------

def _dac_scan_chunk(paths_strs: list[str]) -> dict:
    """Count pcm_offset occurrences across DAC events in a file chunk."""
    from .ym2612 import Ym2612State, CH_DAC
    counter: Counter = Counter()
    for path_str in paths_strs:
        try:
            vgm     = load_vgm(path_str)
            decoder = Ym2612State()
            for note in decoder.process_vgm(vgm):
                if note.channel == CH_DAC and note.dac_sample_id >= 0:
                    counter[note.dac_sample_id] += 1
        except Exception:
            continue
    return dict(counter)


def _extract_pcm_samples(
    vgm_files: list,
    target_offsets: list[int],
    max_sample_bytes: int = 8192,
) -> dict[int, bytes]:
    """Extract raw PCM bytes for each target pcm_offset."""
    remaining = set(target_offsets)
    result: dict[int, bytes] = {}
    for path in vgm_files:
        if not remaining:
            break
        try:
            vgm = load_vgm(str(path))
            if not vgm.pcm_data:
                continue
            from .ym2612 import Ym2612State, CH_DAC
            decoder          = Ym2612State()
            found_this_file: set[int] = set()
            for note in decoder.process_vgm(vgm):
                if note.channel == CH_DAC and note.dac_sample_id in remaining:
                    found_this_file.add(note.dac_sample_id)
            for offset in found_this_file:
                if offset < len(vgm.pcm_data):
                    result[offset] = bytes(vgm.pcm_data[offset : offset + max_sample_bytes])
                    remaining.discard(offset)
        except Exception:
            continue
    return result


# Per-worker tokenizer — initialised once per worker process.
_worker_tok: "TokenizerV6 | None" = None


def _init_encode_worker(
    composer_map_path: str,
    game_map_path: str,
    dac_slot_map_path: str,
) -> None:
    """Pool initializer: load tokenizer artifacts once per worker."""
    global _worker_tok
    cmap = ComposerMap.load(composer_map_path)
    gmap = GameMap.load(game_map_path)
    dac_slot_map: dict[int, int] = {}
    if dac_slot_map_path and Path(dac_slot_map_path).exists():
        raw = json.loads(Path(dac_slot_map_path).read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
    _worker_tok = TokenizerV6(
        composer_map=cmap,
        game_map=gmap,
        dac_slot_map=dac_slot_map,
    )


def _encode_file(args: tuple) -> tuple[str, "list[np.ndarray] | None | str"]:
    """Second-pass worker: encode + (optionally) augment one VGM file.

    Returns:
        (path_str, list of int16 arrays)  — original + transpositions if augmented
        (path_str, None)   if the file was filtered out
        (path_str, 'error') if an exception occurred.
    """
    path_str, augment_keys = args
    try:
        vgm    = load_vgm(path_str)
        tokens = _worker_tok.encode(vgm)
        if tokens is None:
            return (path_str, None)
        # Skip sample-melody tracks where DAC dominates.
        dac_count = sum(
            1 for t in tokens
            if DAC_HIT_BASE <= t < DAC_HIT_BASE + NUM_DAC_SLOTS
        )
        if len(tokens) > 0 and dac_count / len(tokens) > _MAX_DAC_FRACTION:
            return (path_str, None)
        result = [np.array(tokens, dtype=np.int16)]
        if augment_keys:
            for s in range(1, 12):
                result.append(
                    np.array(_worker_tok.transpose(tokens, s), dtype=np.int16)
                )
        return (path_str, result)
    except Exception:
        return (path_str, "error")

# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_dataset_v6(
    vgm_dir: Path | str,
    output_dir: Path | str,
    composer_map_path: Path | str | None = None,
    game_map_path: Path | str | None = None,
    dac_slot_map_path: Path | str | None = None,
    seq_len: int = 16384,
    val_fraction: float = 0.05,
    augment_keys: bool = True,
    max_files: int | None = None,
    num_workers: int = 12,
    pack_holdout_games: int = 8,
) -> dict:
    """Tokenize and augment a VGM corpus for v6 training.

    The train/val/val_pack split is done at FILE level BEFORE augmentation to
    prevent leakage (all 12 key transpositions of a song land in the same split).

    Args:
        vgm_dir:             Directory containing .vgm / .vgz files.
        output_dir:          Where to save train.npy, val.npy, val_pack.npy, meta.json.
        composer_map_path:   Path for composer map JSON.
        game_map_path:       Path for game map JSON.
        dac_slot_map_path:   Path for DAC slot map JSON.
        seq_len:             Context window size (default 16384).
        val_fraction:        Fraction of (non-holdout) files held out for validation.
        augment_keys:        If True, transpose train sequences through all 12 keys.
        max_files:           Cap on VGM files (for debugging).
        num_workers:         Parallel worker processes (default 12).
        pack_holdout_games:  Number of entire game packs held out as val_pack for
                             true generalization measurement (0 to disable).

    Returns:
        Dict with summary statistics (also written to output_dir/meta.json).
    """
    vgm_dir    = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default paths inside the output directory.
    if composer_map_path is None:
        composer_map_path = output_dir / "composer_map_v6.json"
    else:
        composer_map_path = Path(composer_map_path)

    if game_map_path is None:
        game_map_path = output_dir / "game_map_v6.json"
    else:
        game_map_path = Path(game_map_path)

    if dac_slot_map_path is None:
        dac_slot_map_path = output_dir / "dac_slot_map_v6.json"
    else:
        dac_slot_map_path = Path(dac_slot_map_path)

    # ---- Collect VGM files -----------------------------------------------
    vgm_files = sorted(
        list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz"))
        + list(vgm_dir.rglob("*.vgm")) + list(vgm_dir.rglob("*.vgz"))
    )
    seen: set[Path] = set()
    unique = []
    for p in vgm_files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    vgm_files = unique
    if max_files is not None:
        vgm_files = vgm_files[:max_files]
    if not vgm_files:
        raise FileNotFoundError(f"No VGM files found under {vgm_dir}")
    log.info("Found %d VGM files", len(vgm_files))

    # ---- Build or load ComposerMap (fast GD3 scan) -----------------------
    if composer_map_path.exists():
        log.info("Loading existing composer map from %s", composer_map_path)
        composer_map = ComposerMap.load(composer_map_path)
    else:
        log.info("Building composer map (GD3 fast scan)…")
        composer_map = ComposerMap.build(vgm_files)
        composer_map.save(composer_map_path)
        log.info(
            "ComposerMap: %d composers → %s", len(composer_map), composer_map_path
        )

    # ---- Build or load GameMap (fast GD3 scan) ---------------------------
    if game_map_path.exists():
        log.info("Loading existing game map from %s", game_map_path)
        game_map = GameMap.load(game_map_path)
    else:
        log.info("Building game map (GD3 fast scan)…")
        game_map = GameMap.build(vgm_files)
        game_map.save(game_map_path)
        log.info("GameMap: %d games → %s", len(game_map), game_map_path)

    # ---- Build or load DAC slot map --------------------------------------
    if dac_slot_map_path.exists():
        log.info("Loading existing DAC slot map from %s", dac_slot_map_path)
        raw = json.loads(dac_slot_map_path.read_text())
        dac_slot_map: dict[int, int] = {int(k): int(v) for k, v in raw.items()}
        log.info("DAC slot map: %d unique offsets mapped", len(dac_slot_map))

        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v6.json"
        if not drum_kit_path.exists():
            log.info("Drum kit not found — extracting PCM bytes…")
            slot_to_offset = {v: int(k) for k, v in dac_slot_map.items()}
            top_offsets = [slot_to_offset[s] for s in sorted(slot_to_offset)]
            pcm_by_offset = _extract_pcm_samples(vgm_files, top_offsets)
            drum_kit: dict[int, str] = {}
            for slot, offset in enumerate(top_offsets):
                if offset in pcm_by_offset:
                    drum_kit[slot] = pcm_by_offset[offset].hex()
            drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
            log.info("Saved drum kit → %s", drum_kit_path)
        else:
            log.info("Drum kit already exists at %s", drum_kit_path)
    else:
        log.info("Building DAC slot map (%d workers)…", num_workers)
        paths_strs = [str(p) for p in vgm_files]
        chunk_size = max(1, len(paths_strs) // (num_workers * 4))
        chunks = [
            paths_strs[i : i + chunk_size]
            for i in range(0, len(paths_strs), chunk_size)
        ]
        merged_dac: Counter = Counter()
        with mp.Pool(num_workers) as pool:
            for i, cdict in enumerate(
                pool.imap_unordered(_dac_scan_chunk, chunks)
            ):
                merged_dac.update(cdict)
                if (i + 1) % max(1, len(chunks) // 10) == 0 or (i + 1) == len(chunks):
                    log.info(
                        "  DAC scan: %d/%d chunks done (%d unique offsets)",
                        i + 1, len(chunks), len(merged_dac),
                    )

        top_offsets  = [off for off, _ in merged_dac.most_common(NUM_DAC_SLOTS)]
        dac_slot_map = {off: slot for slot, off in enumerate(top_offsets)}
        log.info(
            "DAC slot map built: %d unique offsets, %d slots assigned",
            len(merged_dac), len(dac_slot_map),
        )
        for slot, off in enumerate(top_offsets):
            log.info("  Slot %d: pcm_offset=%d  count=%d", slot, off, merged_dac[off])

        log.info("Extracting PCM sample bytes for drum kit…")
        pcm_by_offset = _extract_pcm_samples(vgm_files, top_offsets)
        drum_kit = {}
        for slot, off in enumerate(top_offsets):
            if off in pcm_by_offset:
                drum_kit[slot] = pcm_by_offset[off].hex()
                log.info("  Slot %d: %d PCM bytes", slot, len(pcm_by_offset[off]))
            else:
                log.warning("  Slot %d: no PCM bytes found for offset %d", slot, off)

        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v6.json"
        drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
        dac_slot_map_path.write_text(
            json.dumps({str(k): v for k, v in dac_slot_map.items()}, indent=2)
        )
        log.info(
            "Saved DAC slot map → %s\nSaved drum kit → %s",
            dac_slot_map_path, drum_kit_path,
        )

    # ---- Fast GD3 game-name scan: assign each file to a split bucket --------
    log.info("Scanning game names for file-level split (%d files)…", len(vgm_files))
    file_game_tok: dict[str, int] = {}
    for p in vgm_files:
        try:
            fields = _fast_read_gd3_fields(Path(str(p)), 2)  # field 2 = game_name_en
            raw_game = fields[0].strip() if fields else ''
            file_game_tok[str(p)] = game_map.lookup(raw_game)
        except Exception:
            file_game_tok[str(p)] = UNK_GAME

    # Group files by game token (for pack holdout)
    game_to_paths: dict[int, list[str]] = defaultdict(list)
    for path_str, tok in file_game_tok.items():
        game_to_paths[tok].append(path_str)

    # Select pack holdout games: skip the top-5 most popular games (keep them
    # in training for good coverage), then take the next N.
    holdout_paths: set[str] = set()
    holdout_game_toks: set[int] = set()
    if pack_holdout_games > 0:
        candidates = sorted(
            ((tok, paths) for tok, paths in game_to_paths.items()
             if tok != UNK_GAME),
            key=lambda x: -len(x[1]),   # most tracks first
        )
        skip = min(5, len(candidates))
        for tok, paths in candidates[skip : skip + pack_holdout_games]:
            holdout_game_toks.add(tok)
            holdout_paths.update(paths)
        log.info(
            "Pack holdout: %d games, %d files", len(holdout_game_toks), len(holdout_paths)
        )

    # File-level train/val split on remaining files
    remaining = [str(p) for p in vgm_files if str(p) not in holdout_paths]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(remaining)
    n_val_files  = max(1, int(len(remaining) * val_fraction))
    val_path_set   = set(remaining[:n_val_files])
    train_path_set = set(remaining[n_val_files:])
    log.info(
        "File split: train=%d  val=%d  holdout=%d",
        len(train_path_set), len(val_path_set), len(holdout_paths),
    )

    # ---- Encode all files (parallel) ------------------------------------
    log.info("Encoding corpus (%d workers)…", num_workers)
    # Train files get key-augmentation; val / holdout files do not.
    task_args = (
        [(path, augment_keys) for path in train_path_set]
        + [(path, False)      for path in val_path_set]
        + [(path, False)      for path in holdout_paths]
    )
    _aug_factor = 12 if augment_keys else 1

    file_results: dict[str, list[np.ndarray]] = {}
    n_filtered = 0
    n_error    = 0

    init_args = (str(composer_map_path), str(game_map_path), str(dac_slot_map_path))
    with mp.Pool(
        num_workers,
        initializer=_init_encode_worker,
        initargs=init_args,
    ) as pool:
        for i, (path_str, result) in enumerate(
            pool.imap_unordered(_encode_file, task_args, chunksize=4)
        ):
            if result is None:
                n_filtered += 1
            elif result == "error":
                n_error += 1
            else:
                file_results[path_str] = result

            if (i + 1) % 500 == 0:
                log.info(
                    "  Encoded %d/%d  (kept=%d  filtered=%d  errors=%d)",
                    i + 1, len(task_args), len(file_results), n_filtered, n_error,
                )

    n_encoded_files = len(file_results)
    log.info(
        "Encoding complete: kept=%d  filtered=%d  errors=%d",
        n_encoded_files, n_filtered, n_error,
    )
    if not file_results:
        raise ValueError("No files survived filtering — check VGM directory.")

    # ---- Assign results to splits ----------------------------------------
    train_seqs: list[np.ndarray] = []
    val_seqs:   list[np.ndarray] = []
    pack_seqs:  list[np.ndarray] = []
    for path_str, seqs in file_results.items():
        if path_str in train_path_set:
            train_seqs.extend(seqs)         # all augmented keys
        elif path_str in val_path_set:
            val_seqs.extend(seqs)           # base key only (no augmentation)
        elif path_str in holdout_paths:
            pack_seqs.extend(seqs)          # base key only

    def _flat(seqs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(seqs) if seqs else np.array([], dtype=np.int16)

    train_arr = _flat(train_seqs)
    val_arr   = _flat(val_seqs)
    pack_arr  = _flat(pack_seqs)

    np.save(output_dir / "train.npy",    train_arr)
    np.save(output_dir / "val.npy",      val_arr)
    np.save(output_dir / "val_pack.npy", pack_arr)

    meta = {
        "tokenizer":            "v6",
        "vocab_size":           VOCAB_SIZE,
        "seq_len":              seq_len,
        "total_vgm_files":      len(vgm_files),
        "encoded_files":        n_encoded_files,
        "filtered_files":       n_filtered,
        "error_files":          n_error,
        "augment_keys":         augment_keys,
        "train_files":          len(train_path_set & set(file_results.keys())),
        "val_files":            len(val_path_set & set(file_results.keys())),
        "holdout_files":        len(holdout_paths & set(file_results.keys())),
        "holdout_games":        pack_holdout_games,
        "train_seqs":           len(train_seqs),
        "val_seqs":             len(val_seqs),
        "pack_seqs":            len(pack_seqs),
        "train_tokens":         int(len(train_arr)),
        "val_tokens":           int(len(val_arr)),
        "pack_tokens":          int(len(pack_arr)),
        "composer_map":         str(composer_map_path),
        "num_composers":        len(composer_map),
        "game_map":             str(game_map_path),
        "num_games":            len(game_map),
        "dac_slot_map":         str(dac_slot_map_path),
        "dac_slots_assigned":   len(dac_slot_map),
        "dac_drum_kit":         str(dac_slot_map_path.parent / "dac_drum_kit_v6.json"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info(
        "Saved: train=%dM tokens, val=%dM tokens, val_pack=%dM tokens → %s",
        len(train_arr) // 1_000_000, len(val_arr) // 1_000_000,
        len(pack_arr) // 1_000_000, output_dir,
    )
    return meta


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_datasets_v6(
    data_dir: Path | str,
    seq_len: int = 16384,
    batch_size: int = 4,
) -> tuple:
    """Load pre-prepared v6 train/val datasets and create DataLoaders.

    Returns:
        (train_loader, val_loader, val_pack_loader_or_None, meta_dict)
    """
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    train_tokens = np.load(data_dir / "train.npy", mmap_mode="r")
    val_tokens   = np.load(data_dir / "val.npy",   mmap_mode="r")

    train_ds = VgmDatasetV6(train_tokens, seq_len=seq_len)
    val_ds   = VgmDatasetV6(val_tokens,   seq_len=seq_len)

    # num_workers=0: memmap arrays cannot be pickled across spawned workers on
    # Windows (same constraint as v4).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=True, drop_last=False,
    )

    pack_loader = None
    pack_path = data_dir / "val_pack.npy"
    if pack_path.exists():
        try:
            pack_tokens = np.load(pack_path, mmap_mode="r")
            pack_ds     = VgmDatasetV6(pack_tokens, seq_len=seq_len)
            pack_loader = DataLoader(
                pack_ds, batch_size=batch_size,
                shuffle=False, num_workers=0,
                pin_memory=True, drop_last=False,
            )
        except ValueError:
            pass  # pack too small for one chunk — skip

    return train_loader, val_loader, pack_loader, meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Prepare v6 training dataset from a VGM corpus."
    )
    parser.add_argument("--vgm-dir",        required=True,
                        help="Directory containing .vgm/.vgz files")
    parser.add_argument("--out-dir",        required=True,
                        help="Output directory for train.npy / val.npy")
    parser.add_argument("--composer-map",   default=None,
                        help="Path for composer map JSON (built if absent; "
                             "defaults to out-dir/composer_map_v6.json)")
    parser.add_argument("--game-map",       default=None,
                        help="Path for game map JSON (built if absent; "
                             "defaults to out-dir/game_map_v6.json)")
    parser.add_argument("--dac-slot-map",   default=None,
                        help="Path for DAC slot map JSON (built if absent; "
                             "defaults to out-dir/dac_slot_map_v6.json)")
    parser.add_argument("--seq-len",        type=int, default=16384,
                        help="Context window size (default 16384)")
    parser.add_argument("--val-frac",       type=float, default=0.05,
                        help="Validation fraction (default 0.05)")
    parser.add_argument("--no-augment",     action="store_true",
                        help="Disable 12-key transposition augmentation")
    parser.add_argument("--max-files",      type=int, default=None,
                        help="Cap on VGM files (for debugging)")
    parser.add_argument("--num-workers",    type=int, default=12,
                        help="Parallel worker processes (default 12)")
    parser.add_argument("--pack-holdout",   type=int, default=8,
                        help="Number of whole game packs held out as val_pack (default 8; 0 to disable)")
    args = parser.parse_args()

    meta = prepare_dataset_v6(
        vgm_dir             = args.vgm_dir,
        output_dir          = args.out_dir,
        composer_map_path   = args.composer_map,
        game_map_path       = args.game_map,
        dac_slot_map_path   = args.dac_slot_map,
        seq_len             = args.seq_len,
        val_fraction        = args.val_frac,
        augment_keys        = not args.no_augment,
        max_files           = args.max_files,
        num_workers         = args.num_workers,
        pack_holdout_games  = args.pack_holdout,
    )

    print("\nDataset summary:")
    for k, v in meta.items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    _main()
