"""v7 dataset preparation — tokenize VGM corpus with 18× augmentation.

Pipeline:
    1. Build / load ComposerMap, GameMap, DAC slot map  (same as v6)
    2. File-level train/val split before augmentation
    3. Optionally load clusters_v7.json for cluster-based oversampling
    4. Encode each file with TokenizerV7
       — train files: 18× augmentation (12 key + 4 tempo + 2 velocity)
       — val / val_pack files: base only
    5. Save train.npy / val.npy / val_pack.npy + meta.json

Augmentation breakdown (additive, NOT multiplicative):
    - 12 key transpositions (semitones 0-11)
    -  4 tempo variants (×0.80, ×0.90, ×1.10, ×1.20 of original BPM)
    -  2 velocity variants (±1 on all VEL tokens)
    = 18 total variants per training song

Cluster oversampling (if clusters_v7.json is supplied):
    cluster5 → 10×  (stereo/special — rarest)
    cluster2 → 5×   (heavy PSG)
    others   → 1×   (no repeat)

Usage::

    python -m genesis_music.dataset_v7 \\
        --vgm-dir  data/vgm \\
        --out-dir  data/prepared_v7

Or with cluster oversampling::

    python -m genesis_music.dataset_v7 \\
        --vgm-dir      data/vgm \\
        --out-dir      data/prepared_v7 \\
        --cluster-map  data/clusters_v7.json
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .tokenizer_v7 import (
    PAD, TEMPO_BASE, VEL_BASE, DAC_HIT_BASE, SEP,
    ComposerMap, GameMap, TokenizerV7,
    VOCAB_SIZE_V7, NUM_GAMES, NUM_DAC_SLOTS, UNK_GAME, RARE_TOKEN_IDS,
    _fast_read_gd3_fields, build_curated_game_map,
)
from .music_analysis import TEMPO_BINS
from .vgm_parser import load_vgm

log = logging.getLogger(__name__)

# DAC-dominant track exclusion threshold (same as v6)
_MAX_DAC_FRACTION: float = 0.50

# Cluster → repeat count for oversampling
_CLUSTER_REPEAT: dict[int, int] = {1: 1, 2: 5, 3: 1, 4: 1, 5: 10}

# Tempo shift factors for augmentation (4 variants, original not repeated)
_TEMPO_FACTORS = (0.80, 0.90, 1.10, 1.20)

# Velocity shifts for augmentation
_VEL_SHIFTS = (-1, +1)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_tempo(tokens: list[int], factor: float) -> list[int]:
    """Shift the TEMPO token by applying *factor* to the BPM value."""
    n_bins = len(TEMPO_BINS)
    result = []
    for tok in tokens:
        if TEMPO_BASE <= tok < TEMPO_BASE + n_bins:
            bpm     = float(TEMPO_BINS[tok - TEMPO_BASE])
            new_bpm = bpm * factor
            diffs   = [abs(new_bpm - b) for b in TEMPO_BINS]
            result.append(TEMPO_BASE + int(min(range(len(diffs)), key=diffs.__getitem__)))
        else:
            result.append(tok)
    return result


def _augment_velocity(tokens: list[int], shift: int) -> list[int]:
    """Shift all VEL tokens by *shift*, clamping to [0, 15]."""
    result = []
    for tok in tokens:
        if VEL_BASE <= tok < VEL_BASE + 16:
            result.append(VEL_BASE + max(0, min(15, (tok - VEL_BASE) + shift)))
        else:
            result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


class VgmDatasetV7(Dataset):
    """PyTorch Dataset of fixed-length v7 token windows.

    Each item is a dict:
        input_ids:  (seq_len,) int64
        labels:     (seq_len,) int64  — shifted by 1, PAD replaced by -100
    """

    def __init__(self, tokens: np.ndarray, seq_len: int = 8192) -> None:
        self.seq_len    = seq_len
        chunk_size      = seq_len + 1
        n_chunks        = len(tokens) // chunk_size
        if n_chunks == 0:
            raise ValueError(
                f"Token array too short ({len(tokens)}) for seq_len={seq_len}"
            )
        self._tokens     = tokens
        self._chunk_size = chunk_size
        self._n_chunks   = n_chunks
        log.info(
            "VgmDatasetV7: %d chunks × %d tokens  (%.1fM total)",
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
# Parallel worker helpers  (module-level for Windows spawn)
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
            decoder = Ym2612State()
            for note in decoder.process_vgm(vgm):
                if note.channel == CH_DAC and note.dac_sample_id in remaining:
                    offset = note.dac_sample_id
                    if offset < len(vgm.pcm_data):
                        result[offset] = bytes(
                            vgm.pcm_data[offset : offset + max_sample_bytes]
                        )
                        remaining.discard(offset)
        except Exception:
            continue
    return result


# Per-worker state (initialised once per process)
_worker_tok:      "TokenizerV7 | None" = None
_worker_clusters: "dict[str, int] | None" = None  # path → cluster 1-5


def _init_encode_worker(
    composer_map_path: str,
    game_map_path: str,
    dac_slot_map_path: str,
    cluster_map_path: str,
) -> None:
    global _worker_tok, _worker_clusters
    cmap = ComposerMap.load(composer_map_path)
    gmap = GameMap.load(game_map_path)
    dac_slot_map: dict[int, int] = {}
    if dac_slot_map_path and Path(dac_slot_map_path).exists():
        raw = json.loads(Path(dac_slot_map_path).read_text())
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
    _worker_tok = TokenizerV7(
        composer_map=cmap,
        game_map=gmap,
        dac_slot_map=dac_slot_map,
    )
    _worker_clusters = {}
    if cluster_map_path and Path(cluster_map_path).exists():
        raw_cl = json.loads(Path(cluster_map_path).read_text())
        _worker_clusters = raw_cl.get("files", {})


def _encode_file(args: tuple) -> tuple[str, "list[np.ndarray] | None | str"]:
    """Encode one VGM file and return all augmented variants.

    For train files: 18 variants (12 key + 4 tempo + 2 vel).
    For non-train files: 1 variant (base key/tempo/vel).
    """
    path_str, augment, is_train = args
    try:
        vgm    = load_vgm(path_str)
        tokens = _worker_tok.encode(vgm)
        if tokens is None:
            return (path_str, None)

        dac_count = sum(
            1 for t in tokens
            if DAC_HIT_BASE <= t < DAC_HIT_BASE + NUM_DAC_SLOTS
        )
        if len(tokens) > 0 and dac_count / len(tokens) > _MAX_DAC_FRACTION:
            return (path_str, None)

        result: list[np.ndarray] = []

        if augment and is_train:
            # 12 key transpositions (semitones 0-11)
            for s in range(12):
                variant = _worker_tok.transpose(tokens, s)
                result.append(np.array(variant, dtype=np.int16))

            # 4 tempo variants (at original key)
            for factor in _TEMPO_FACTORS:
                variant = _augment_tempo(tokens, factor)
                result.append(np.array(variant, dtype=np.int16))

            # 2 velocity variants (at original key/tempo)
            for shift in _VEL_SHIFTS:
                variant = _augment_velocity(tokens, shift)
                result.append(np.array(variant, dtype=np.int16))
        else:
            result.append(np.array(tokens, dtype=np.int16))

        return (path_str, result)
    except Exception:
        return (path_str, "error")


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_dataset_v7(
    vgm_dir: Path | str,
    output_dir: Path | str,
    composer_map_path: Path | str | None = None,
    game_map_path: Path | str | None = None,
    curated_game_map_path: Path | str | None = None,
    dac_slot_map_path: Path | str | None = None,
    cluster_map_path: Path | str | None = None,
    seq_len: int = 8192,
    val_fraction: float = 0.05,
    augment: bool = True,
    max_files: int | None = None,
    num_workers: int = 12,
    pack_holdout_games: int = 8,
) -> dict:
    """Tokenize and augment a VGM corpus for v7 training.

    Args:
        vgm_dir:             Directory containing .vgm / .vgz files.
        output_dir:          Where to save train.npy, val.npy, val_pack.npy, meta.json.
        cluster_map_path:    Optional path to clusters_v7.json for oversampling.
        seq_len:             Context window size (default 8192).
        augment:             If True, apply 18× augmentation to training files.
        (other args identical to prepare_dataset_v6)
    """
    vgm_dir    = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if composer_map_path is None:
        composer_map_path = output_dir / "composer_map_v7.json"
    if game_map_path is None:
        game_map_path = output_dir / "game_map_v7.json"
    if dac_slot_map_path is None:
        dac_slot_map_path = output_dir / "dac_slot_map_v7.json"

    composer_map_path = Path(composer_map_path)
    game_map_path     = Path(game_map_path)
    dac_slot_map_path = Path(dac_slot_map_path)

    # ---- Collect VGM files -----------------------------------------------
    vgm_files = sorted(
        list(vgm_dir.rglob("*.vgm")) + list(vgm_dir.rglob("*.vgz"))
    )
    seen: set[Path] = set()
    unique: list[Path] = []
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

    # ---- ComposerMap -------------------------------------------------------
    if composer_map_path.exists():
        composer_map = ComposerMap.load(composer_map_path)
        log.info("Loaded composer map (%d composers)", len(composer_map))
    else:
        composer_map = ComposerMap.build(vgm_files)
        composer_map.save(composer_map_path)
        log.info("Built composer map → %s", composer_map_path)

    # ---- GameMap -----------------------------------------------------------
    if game_map_path.exists():
        game_map = GameMap.load(game_map_path)
        log.info("Loaded game map (%d games)", len(game_map))
    elif curated_game_map_path is not None:
        game_map = build_curated_game_map(curated_game_map_path)
        game_map.save(game_map_path)
        log.info("Built curated game map (%d games) → %s", len(game_map), game_map_path)
    else:
        game_map = GameMap.build(vgm_files)
        game_map.save(game_map_path)
        log.info("Built game map → %s", game_map_path)

    # ---- DAC slot map ------------------------------------------------------
    if dac_slot_map_path.exists():
        raw = json.loads(dac_slot_map_path.read_text())
        dac_slot_map: dict[int, int] = {int(k): int(v) for k, v in raw.items()}
        log.info("Loaded DAC slot map (%d slots)", len(dac_slot_map))

        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v7.json"
        if not drum_kit_path.exists():
            log.info("Building drum kit…")
            slot_to_offset = {v: int(k) for k, v in dac_slot_map.items()}
            top_offsets    = [slot_to_offset[s] for s in sorted(slot_to_offset)]
            pcm_by_offset  = _extract_pcm_samples(vgm_files, top_offsets)
            drum_kit = {slot: pcm_by_offset[off].hex()
                        for slot, off in enumerate(top_offsets)
                        if off in pcm_by_offset}
            drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
    else:
        log.info("Building DAC slot map (%d workers)…", num_workers)
        paths_strs = [str(p) for p in vgm_files]
        chunk_size = max(1, len(paths_strs) // (num_workers * 4))
        chunks = [paths_strs[i:i+chunk_size] for i in range(0, len(paths_strs), chunk_size)]
        merged_dac: Counter = Counter()
        with mp.Pool(num_workers) as pool:
            for cdict in pool.imap_unordered(_dac_scan_chunk, chunks):
                merged_dac.update(cdict)

        top_offsets  = [off for off, _ in merged_dac.most_common(NUM_DAC_SLOTS)]
        dac_slot_map = {off: slot for slot, off in enumerate(top_offsets)}
        log.info("Built DAC slot map: %d offsets, %d slots", len(merged_dac), len(dac_slot_map))

        pcm_by_offset = _extract_pcm_samples(vgm_files, top_offsets)
        drum_kit = {slot: pcm_by_offset[off].hex()
                    for slot, off in enumerate(top_offsets)
                    if off in pcm_by_offset}
        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v7.json"
        drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
        dac_slot_map_path.write_text(
            json.dumps({str(k): v for k, v in dac_slot_map.items()}, indent=2)
        )

    # ---- Load cluster map (optional) ----------------------------------------
    cluster_file_map: dict[str, int] = {}
    if cluster_map_path is not None and Path(cluster_map_path).exists():
        raw_cl = json.loads(Path(cluster_map_path).read_text())
        cluster_file_map = {str(Path(k).resolve()): v
                            for k, v in raw_cl.get("files", {}).items()}
        log.info("Loaded cluster map: %d entries", len(cluster_file_map))
    else:
        log.info("No cluster map — cluster oversampling disabled")

    # ---- File-level train/val split ----------------------------------------
    log.info("Scanning game names for split…")
    file_game_tok: dict[str, int] = {}
    for p in vgm_files:
        try:
            fields = _fast_read_gd3_fields(Path(str(p)), 2)
            raw_game = fields[0].strip() if fields else ''
            file_game_tok[str(p)] = game_map.lookup(raw_game)
        except Exception:
            file_game_tok[str(p)] = UNK_GAME

    game_to_paths: dict[int, list[str]] = defaultdict(list)
    for path_str, tok in file_game_tok.items():
        game_to_paths[tok].append(path_str)

    holdout_paths:    set[str] = set()
    holdout_game_toks: set[int] = set()
    if pack_holdout_games > 0:
        candidates = sorted(
            ((tok, paths) for tok, paths in game_to_paths.items() if tok != UNK_GAME),
            key=lambda x: -len(x[1]),
        )
        skip = min(5, len(candidates))
        for tok, paths in candidates[skip:skip + pack_holdout_games]:
            holdout_game_toks.add(tok)
            holdout_paths.update(paths)
        log.info("Pack holdout: %d games, %d files", len(holdout_game_toks), len(holdout_paths))

    remaining = [str(p) for p in vgm_files if str(p) not in holdout_paths]
    rng       = np.random.default_rng(seed=42)
    rng.shuffle(remaining)
    n_val_files    = max(1, int(len(remaining) * val_fraction))
    val_path_set   = set(remaining[:n_val_files])
    train_path_set = set(remaining[n_val_files:])
    log.info("File split: train=%d  val=%d  holdout=%d",
             len(train_path_set), len(val_path_set), len(holdout_paths))

    # ---- Encode all files (parallel) ----------------------------------------
    log.info("Encoding corpus (%d workers, 18× aug)…", num_workers)
    task_args = (
        [(path, augment, True)  for path in train_path_set]
        + [(path, False, False) for path in val_path_set]
        + [(path, False, False) for path in holdout_paths]
    )

    file_results: dict[str, list[np.ndarray]] = {}
    n_filtered = 0
    n_error    = 0

    init_args = (
        str(composer_map_path), str(game_map_path), str(dac_slot_map_path),
        str(cluster_map_path) if cluster_map_path else "",
    )
    with mp.Pool(num_workers, initializer=_init_encode_worker, initargs=init_args) as pool:
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
                log.info("  Encoded %d/%d  (kept=%d  filtered=%d  err=%d)",
                         i + 1, len(task_args), len(file_results), n_filtered, n_error)

    log.info("Encoding done: kept=%d  filtered=%d  err=%d",
             len(file_results), n_filtered, n_error)
    if not file_results:
        raise ValueError("No files survived filtering.")

    # ---- Build splits with cluster oversampling ----------------------------
    train_seqs: list[np.ndarray] = []
    val_seqs:   list[np.ndarray] = []
    pack_seqs:  list[np.ndarray] = []

    for path_str, seqs in file_results.items():
        if path_str in train_path_set:
            cluster = cluster_file_map.get(str(Path(path_str).resolve()), 1)
            repeat  = _CLUSTER_REPEAT.get(cluster, 1)
            for _ in range(repeat):
                train_seqs.extend(seqs)
        elif path_str in val_path_set:
            val_seqs.extend(seqs)
        elif path_str in holdout_paths:
            pack_seqs.extend(seqs)

    def _flat(seqs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(seqs) if seqs else np.array([], dtype=np.int16)

    train_arr = _flat(train_seqs)
    val_arr   = _flat(val_seqs)
    pack_arr  = _flat(pack_seqs)

    np.save(output_dir / "train.npy",    train_arr)
    np.save(output_dir / "val.npy",      val_arr)
    np.save(output_dir / "val_pack.npy", pack_arr)

    meta = {
        "tokenizer_version":      "v7",
        "vocab_size":             VOCAB_SIZE_V7,
        "seq_len":                seq_len,
        "rare_token_ids":         sorted(RARE_TOKEN_IDS),
        "total_vgm_files":        len(vgm_files),
        "encoded_files":          len(file_results),
        "filtered_files":         n_filtered,
        "error_files":            n_error,
        "augmentation_summary":   "12-key + 4-tempo + 2-vel = 18x (additive)",
        "augmented":              augment,
        "train_files":            len(train_path_set & set(file_results.keys())),
        "val_files":              len(val_path_set & set(file_results.keys())),
        "holdout_files":          len(holdout_paths & set(file_results.keys())),
        "holdout_games":          pack_holdout_games,
        "cluster_oversampling":   cluster_map_path is not None,
        "cluster_repeat_map":     _CLUSTER_REPEAT,
        "train_seqs":             len(train_seqs),
        "val_seqs":               len(val_seqs),
        "pack_seqs":              len(pack_seqs),
        "train_tokens":           int(len(train_arr)),
        "val_tokens":             int(len(val_arr)),
        "pack_tokens":            int(len(pack_arr)),
        "composer_map":           str(composer_map_path),
        "num_composers":          len(composer_map),
        "game_map":               str(game_map_path),
        "num_games":              len(game_map),
        "dac_slot_map":           str(dac_slot_map_path),
        "dac_slots_assigned":     len(dac_slot_map),
        "dac_drum_kit":           str(dac_slot_map_path.parent / "dac_drum_kit_v7.json"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "Saved: train=%dM tokens, val=%dM tokens, pack=%dM tokens → %s",
        len(train_arr) // 1_000_000, len(val_arr) // 1_000_000,
        len(pack_arr) // 1_000_000, output_dir,
    )
    return meta


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_datasets_v7(
    data_dir: Path | str,
    seq_len: int = 8192,
    batch_size: int = 4,
) -> tuple:
    """Load pre-prepared v7 train/val datasets and create DataLoaders.

    Returns:
        (train_loader, val_loader, val_pack_loader_or_None, meta_dict)
    """
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    train_tokens = np.load(data_dir / "train.npy", mmap_mode="r")
    val_tokens   = np.load(data_dir / "val.npy",   mmap_mode="r")

    train_ds = VgmDatasetV7(train_tokens, seq_len=seq_len)
    val_ds   = VgmDatasetV7(val_tokens,   seq_len=seq_len)

    # num_workers=0: memmap pickle issue on Windows spawn
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True, drop_last=False,
    )

    pack_loader = None
    pack_path   = data_dir / "val_pack.npy"
    if pack_path.exists():
        try:
            pack_tokens = np.load(pack_path, mmap_mode="r")
            pack_ds     = VgmDatasetV7(pack_tokens, seq_len=seq_len)
            pack_loader = DataLoader(
                pack_ds, batch_size=batch_size,
                shuffle=False, num_workers=0, pin_memory=True, drop_last=False,
            )
        except ValueError:
            pass  # pack too small — skip

    return train_loader, val_loader, pack_loader, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Prepare v7 training dataset from a VGM corpus."
    )
    parser.add_argument("--vgm-dir",      required=True, type=Path)
    parser.add_argument("--out-dir",      required=True, type=Path)
    parser.add_argument("--composer-map", default=None,  type=Path)
    parser.add_argument("--game-map",     default=None,  type=Path)
    parser.add_argument("--curated-games", default=None,  type=Path,
                        help="curated_games_v7.json for deterministic GameMap assignment (optional)")
    parser.add_argument("--dac-slot-map", default=None,  type=Path)
    parser.add_argument("--cluster-map",  default=None,  type=Path,
                        help="clusters_v7.json for oversampling (optional)")
    parser.add_argument("--seq-len",      type=int, default=8192)
    parser.add_argument("--val-frac",     type=float, default=0.05)
    parser.add_argument("--no-augment",   action="store_true")
    parser.add_argument("--max-files",    type=int, default=None)
    parser.add_argument("--num-workers",  type=int, default=12)
    parser.add_argument("--pack-holdout", type=int, default=8)
    args = parser.parse_args()

    meta = prepare_dataset_v7(
        vgm_dir           = args.vgm_dir,
        output_dir        = args.out_dir,
        composer_map_path = args.composer_map,
        game_map_path     = args.game_map,
        curated_game_map_path = args.curated_games,
        dac_slot_map_path = args.dac_slot_map,
        cluster_map_path  = args.cluster_map,
        seq_len           = args.seq_len,
        val_fraction      = args.val_frac,
        augment           = not args.no_augment,
        max_files         = args.max_files,
        num_workers       = args.num_workers,
        pack_holdout_games= args.pack_holdout,
    )

    print("\nDataset summary:")
    for k, v in meta.items():
        if not isinstance(v, dict) and not isinstance(v, list):
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    mp.freeze_support()
    _main()
