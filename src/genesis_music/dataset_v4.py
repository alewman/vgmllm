"""v4 dataset preparation — tokenize VGM corpus with augmentation.

Pipeline:
    1. First pass: build PatchLibrary from all VGM files
    2. Second pass: encode each file with TokenizerV4 (filters SFX/jingles)
    3. Augmentation: transpose each sequence across all 12 keys
    4. Shuffle and split into train/val
    5. Save as .npy files in the output directory (same format as v3)

Typical usage::

    python -m genesis_music.dataset_v4 \\
        --vgm-dir  data/vgm \\
        --out-dir  data/prepared_v4 \\
        --patch-lib data/patch_library_v4.json

Once prepared, training uses the same VgmDataset class with seq_len=16384.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .tokenizer_v4 import (
    PAD, BOS, EOS, DAC_HIT_BASE,
    ComposerMap, PatchLibrary, TokenizerV4,
    VOCAB_SIZE, NUM_PATCHES, NUM_DAC_SLOTS,
)
from .vgm_parser import load_vgm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class (reusable by train.py for v4)
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


class VgmDatasetV4(Dataset):
    """PyTorch Dataset of fixed-length v4 token windows.

    Same interface as VgmDataset in dataset.py but with the v4 PAD/EOS tokens.
    Each item is a dict:
        input_ids:  (seq_len,) int64
        labels:     (seq_len,) int64  — shifted by 1, PAD replaced by -100
    """

    def __init__(self, tokens: np.ndarray, seq_len: int = 16384) -> None:
        self.seq_len     = seq_len
        chunk_size = seq_len + 1
        n_chunks   = len(tokens) // chunk_size
        if n_chunks == 0:
            raise ValueError(f"Token array too short ({len(tokens)}) for seq_len={seq_len}")

        self._tokens     = tokens
        self._chunk_size = chunk_size
        self._n_chunks   = n_chunks

        log.info(
            "VgmDatasetV4: %d chunks × %d tokens  (%.1fM total)",
            n_chunks, seq_len, n_chunks * seq_len / 1e6,
        )

    def __len__(self) -> int:
        return self._n_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start  = idx * self._chunk_size
        chunk  = torch.from_numpy(
            self._tokens[start : start + self._chunk_size].astype(np.int64)
        )
        input_ids = chunk[:-1]
        labels    = chunk[1:].clone()
        labels[labels == PAD] = -100
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Parallel worker functions (must be module-level for Windows spawn)
# ---------------------------------------------------------------------------

def _patch_scan_chunk(paths_strs: list[str]) -> tuple[dict, dict]:
    """First-pass worker: extract FM patches from a chunk of VGM files.

    Returns (counter_dict, fingerprint→patch_dict) where patches are
    serialised as plain dicts (picklable on Windows spawn).
    """
    counter: Counter = Counter()
    fp_to_patch: dict = {}
    for path_str in paths_strs:
        try:
            from .ym2612 import Ym2612State
            vgm = load_vgm(path_str)
            decoder = Ym2612State()
            list(decoder.process_vgm(vgm))
            for patch in decoder.last_patches.values():
                fp = patch.to_fingerprint()
                counter[fp] += 1
                if fp not in fp_to_patch:
                    fp_to_patch[fp] = PatchLibrary._patch_to_dict(patch)
        except Exception:
            continue
    return dict(counter), fp_to_patch


def _dac_scan_chunk(paths_strs: list[str]) -> dict:
    """First-pass worker: count pcm_offset occurrences across DAC events.

    Returns a dict mapping pcm_offset (int) → count (int).
    This is used to rank-assign the 8 DAC drum identity slots.
    """
    from .ym2612 import Ym2612State, CH_DAC
    from .vgm_parser import EventType
    counter: Counter = Counter()
    for path_str in paths_strs:
        try:
            vgm = load_vgm(path_str)
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
    """Extract raw PCM bytes for each target pcm_offset.

    Scans VGM files until all target offsets have been found.
    Returns dict mapping pcm_offset → bytes (up to max_sample_bytes each).
    """
    remaining = set(target_offsets)
    result: dict[int, bytes] = {}

    for path in vgm_files:
        if not remaining:
            break
        try:
            vgm = load_vgm(str(path))
            if not vgm.pcm_data:
                continue
            # Check if this file uses any of the remaining offsets
            from .ym2612 import Ym2612State, CH_DAC
            decoder = Ym2612State()
            found_this_file: set[int] = set()
            for note in decoder.process_vgm(vgm):
                if note.channel == CH_DAC and note.dac_sample_id in remaining:
                    found_this_file.add(note.dac_sample_id)
            # Extract bytes for each found offset
            for offset in found_this_file:
                if offset < len(vgm.pcm_data):
                    # Read up to max_sample_bytes starting at this offset.
                    # Stop at the next seek boundary (heuristic: first byte
                    # returning to 0x80 center after a run of non-center values).
                    sample_bytes = vgm.pcm_data[offset:offset + max_sample_bytes]
                    result[offset] = bytes(sample_bytes)
                    remaining.discard(offset)
        except Exception:
            continue

    return result


# Per-worker tokenizer — initialised once per worker process by _init_encode_worker.
_worker_tok: "TokenizerV4 | None" = None


def _init_encode_worker(patch_lib_path: str, composer_map_path: str, dac_slot_map_path: str) -> None:
    """Pool initializer: load tokenizer once per worker process."""
    global _worker_tok
    lib  = PatchLibrary.load(patch_lib_path)
    cmap = ComposerMap.load(composer_map_path)
    dac_slot_map: dict[int, int] = {}
    if dac_slot_map_path and Path(dac_slot_map_path).exists():
        raw = json.loads(Path(dac_slot_map_path).read_text())
        # JSON keys are strings; convert back to int
        dac_slot_map = {int(k): int(v) for k, v in raw.items()}
    _worker_tok = TokenizerV4(lib, composer_map=cmap, dac_slot_map=dac_slot_map)


# Maximum fraction of tokens that may be DAC_HIT events before a file is
# considered a sample-heavy track and filtered from training.  Games like
# Addams Family Values stream continuous PCM melody through the DAC channel;
# after the onset-detection fix each of those rapid seeks still produces a
# token, so such tracks can still be 60-95 % DAC.  Dropping them keeps the
# dataset FM-centric without losing typical drumming games (10-25 % DAC).
_MAX_DAC_FRACTION: float = 0.50


def _encode_file(args: tuple) -> "list[np.ndarray] | None | str":
    """Second-pass worker: encode + augment one VGM file.

    Returns:
        list of int16 numpy arrays (original + transpositions) on success,
        None if the file was filtered out,
        'error' if an exception occurred.
    """
    path_str, augment_keys = args
    try:
        vgm    = load_vgm(path_str)
        tokens = _worker_tok.encode(vgm)
        if tokens is None:
            return None
        # Filter files where DAC hits dominate the token stream.  These are
        # typically sample-based melodic tracks, not FM music with drumming.
        dac_count = sum(
            1 for t in tokens
            if DAC_HIT_BASE <= t < DAC_HIT_BASE + NUM_DAC_SLOTS
        )
        if len(tokens) > 0 and dac_count / len(tokens) > _MAX_DAC_FRACTION:
            return None
        result = [np.array(tokens, dtype=np.int16)]
        if augment_keys:
            for s in range(1, 12):
                result.append(np.array(_worker_tok.transpose(tokens, s), dtype=np.int16))
        return result
    except Exception:
        return "error"


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------

def prepare_dataset_v4(
    vgm_dir: Path | str,
    output_dir: Path | str,
    patch_lib_path: Path | str,
    composer_map_path: Path | str | None = None,
    dac_slot_map_path: Path | str | None = None,
    seq_len: int = 16384,
    val_fraction: float = 0.05,
    augment_keys: bool = True,
    max_files: int | None = None,
    num_workers: int = 12,
) -> dict:
    """Tokenize and augment a VGM corpus for v4 training.

    Args:
        vgm_dir:            Directory containing .vgm / .vgz files.
        output_dir:         Where to save train.npy, val.npy, meta.json.
        patch_lib_path:     Path to an existing patch library JSON, or a path
                            where a newly-built library should be saved.
        composer_map_path:  Path for composer map JSON.  If None, defaults to
                            output_dir/composer_map_v4.json.
        dac_slot_map_path:  Path for DAC slot map JSON (pcm_offset -> slot 0-7).
                            If None, defaults to output_dir/dac_slot_map_v4.json.
        seq_len:            Context window size (default 16384 for v4).
        val_fraction:       Fraction of *files* held out for validation.
        augment_keys:       If True, transpose each sequence through all 12 keys
                            (12x augmentation factor).
        max_files:          Cap on VGM files to process (for debugging).
        num_workers:        Number of parallel worker processes (default 12).

    Returns:
        Dict with summary statistics.
    """
    vgm_dir    = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_lib_path = Path(patch_lib_path)
    if composer_map_path is None:
        composer_map_path = output_dir / "composer_map_v4.json"
    else:
        composer_map_path = Path(composer_map_path)
    if dac_slot_map_path is None:
        dac_slot_map_path = output_dir / "dac_slot_map_v4.json"
    else:
        dac_slot_map_path = Path(dac_slot_map_path)

    # ---- Collect VGM files -----------------------------------------------
    vgm_files = sorted(
        list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz"))
        + list(vgm_dir.rglob("*.vgm")) + list(vgm_dir.rglob("*.vgz"))
    )
    # Deduplicate (rglob and glob may overlap at top level)
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

    # ---- Build or load patch library (first pass) ----------------------
    if patch_lib_path.exists():
        log.info("Loading existing patch library from %s", patch_lib_path)
        lib = PatchLibrary.load(patch_lib_path)
    else:
        log.info("Building patch library (%d workers, first pass)…", num_workers)
        paths_strs = [str(p) for p in vgm_files]
        chunk_size = max(1, len(paths_strs) // (num_workers * 4))
        chunks = [paths_strs[i:i + chunk_size]
                  for i in range(0, len(paths_strs), chunk_size)]

        merged_counter: Counter = Counter()
        merged_fp_to_patch: dict = {}
        with mp.Pool(num_workers) as pool:
            for i, (cdict, fp_patch) in enumerate(
                pool.imap_unordered(_patch_scan_chunk, chunks)
            ):
                merged_counter.update(cdict)
                for fp, pdict in fp_patch.items():
                    if fp not in merged_fp_to_patch:
                        merged_fp_to_patch[fp] = pdict
                if (i + 1) % max(1, len(chunks) // 10) == 0 or (i + 1) == len(chunks):
                    log.info("  PatchLibrary: %d/%d chunks done (%d unique patches)",
                             i + 1, len(chunks), len(merged_counter))

        ranked_fps = [fp for fp, _ in merged_counter.most_common(NUM_PATCHES)]
        ranked_patches = [PatchLibrary._dict_to_patch(merged_fp_to_patch[fp])
                          for fp in ranked_fps]
        lib = PatchLibrary(ranked_patches)
        log.info("PatchLibrary: %d unique patches, keeping top %d",
                 len(merged_counter), len(lib))
        lib.save(patch_lib_path)

    # ---- Build or load composer map (fast GD3 header scan) --------------
    if composer_map_path.exists():
        log.info("Loading existing composer map from %s", composer_map_path)
        composer_map = ComposerMap.load(composer_map_path)
    else:
        log.info("Building composer map (GD3 fast scan)...")
        composer_map = ComposerMap.build(vgm_files)
        composer_map.save(composer_map_path)

    # ---- Build or load DAC slot map (drum identity scan) ----------------
    # Counts pcm_offset occurrences across the corpus (using onset detection),
    # then maps the top-NUM_DAC_SLOTS offsets to slot indices 0..7.
    # Slot 0 = most frequent sample (typically kick drum).
    if dac_slot_map_path.exists():
        log.info("Loading existing DAC slot map from %s", dac_slot_map_path)
        raw = json.loads(dac_slot_map_path.read_text())
        dac_slot_map: dict[int, int] = {int(k): int(v) for k, v in raw.items()}
        log.info("DAC slot map: %d unique offsets mapped", len(dac_slot_map))

        # Build drum kit if not already present alongside the slot map
        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v4.json"
        if not drum_kit_path.exists():
            log.info("Drum kit not found — extracting PCM bytes for each slot...")
            # slot map maps pcm_offset -> slot; invert to get ordered offsets
            slot_to_offset = {v: int(k) for k, v in dac_slot_map.items()}
            top_offsets = [slot_to_offset[s] for s in sorted(slot_to_offset)]
            pcm_by_offset = _extract_pcm_samples(vgm_files, top_offsets)
            drum_kit: dict[int, str] = {}
            for slot, offset in enumerate(top_offsets):
                if offset in pcm_by_offset:
                    drum_kit[slot] = pcm_by_offset[offset].hex()
                    log.info("  Slot %d: extracted %d PCM bytes", slot, len(pcm_by_offset[offset]))
                else:
                    log.warning("  Slot %d: no PCM bytes found for offset %d", slot, offset)
            drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
            log.info("Saved drum kit -> %s", drum_kit_path)
        else:
            log.info("Drum kit already exists at %s", drum_kit_path)
    else:
        log.info("Building DAC slot map (%d workers, DAC scan)...", num_workers)
        paths_strs = [str(p) for p in vgm_files]
        chunk_size = max(1, len(paths_strs) // (num_workers * 4))
        chunks = [paths_strs[i:i + chunk_size]
                  for i in range(0, len(paths_strs), chunk_size)]

        merged_dac: Counter = Counter()
        with mp.Pool(num_workers) as pool:
            for i, cdict in enumerate(
                pool.imap_unordered(_dac_scan_chunk, chunks)
            ):
                merged_dac.update(cdict)
                if (i + 1) % max(1, len(chunks) // 10) == 0 or (i + 1) == len(chunks):
                    log.info("  DAC scan: %d/%d chunks done (%d unique offsets)",
                             i + 1, len(chunks), len(merged_dac))

        # Assign slot 0 to most common, slot 1 to second most common, etc.
        top_offsets = [offset for offset, _ in merged_dac.most_common(NUM_DAC_SLOTS)]
        dac_slot_map = {offset: slot for slot, offset in enumerate(top_offsets)}
        log.info("DAC slot map built: %d unique offsets, assigning %d slots",
                 len(merged_dac), len(dac_slot_map))
        for slot, offset in enumerate(top_offsets):
            log.info("  Slot %d: pcm_offset=%d  count=%d", slot, offset, merged_dac[offset])

        # Extract actual PCM bytes for each slot (for VGM synthesis / drum kit)
        log.info("Extracting PCM sample bytes for each drum slot...")
        pcm_by_offset = _extract_pcm_samples(vgm_files, top_offsets)
        drum_kit: dict[int, str] = {}  # slot → hex-encoded bytes
        for slot, offset in enumerate(top_offsets):
            if offset in pcm_by_offset:
                drum_kit[slot] = pcm_by_offset[offset].hex()
                log.info("  Slot %d: extracted %d PCM bytes", slot, len(pcm_by_offset[offset]))
            else:
                log.warning("  Slot %d: no PCM bytes found for offset %d", slot, offset)

        # Save drum kit alongside slot map
        drum_kit_path = dac_slot_map_path.parent / "dac_drum_kit_v4.json"
        drum_kit_path.write_text(json.dumps(drum_kit, indent=2))
        log.info("Saved drum kit -> %s", drum_kit_path)

        # Persist slot map as JSON (string keys for JSON compatibility)
        dac_slot_map_path.write_text(
            json.dumps({str(k): v for k, v in dac_slot_map.items()}, indent=2)
        )
        log.info("Saved DAC slot map -> %s", dac_slot_map_path)

    # ---- Encode all files with inline augmentation (second pass) --------
    # Workers load the tokenizer once via the pool initializer; results come
    # back as lists of int16 numpy arrays (original + transpositions).
    log.info("Encoding corpus (%d workers, second pass)...", num_workers)
    _aug_factor = 12 if augment_keys else 1
    task_args = [(str(p), augment_keys) for p in vgm_files]

    all_seqs: list[np.ndarray] = []
    n_filtered = 0
    n_error    = 0

    init_args = (str(patch_lib_path), str(composer_map_path), str(dac_slot_map_path))
    with mp.Pool(num_workers,
                 initializer=_init_encode_worker,
                 initargs=init_args) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_encode_file, task_args, chunksize=4)
        ):
            if result is None:
                n_filtered += 1
            elif result == "error":
                n_error += 1
            else:
                all_seqs.extend(result)

            if (i + 1) % 500 == 0:
                n_encoded = len(all_seqs) // _aug_factor
                log.info("  Encoded %d/%d  (kept=%d  filtered=%d  errors=%d)",
                         i + 1, len(vgm_files), n_encoded, n_filtered, n_error)

    n_encoded_files = len(all_seqs) // _aug_factor
    log.info(
        "Encoding complete: kept=%d  filtered=%d  errors=%d  augmented_seqs=%d",
        n_encoded_files, n_filtered, n_error, len(all_seqs),
    )

    if not all_seqs:
        raise ValueError("No files survived filtering — check VGM directory.")

    # ---- Shuffle + train/val split (at file level) ----------------------
    rng     = np.random.default_rng(seed=42)
    indices = rng.permutation(len(all_seqs))

    n_val        = max(1, int(len(all_seqs) * val_fraction))
    val_set      = set(indices[:n_val].tolist())

    train_seqs: list[np.ndarray] = []
    val_seqs:   list[np.ndarray] = []
    for i, seq in enumerate(all_seqs):
        (val_seqs if i in val_set else train_seqs).append(seq)

    # ---- Flatten to 1-D arrays and save ---------------------------------
    def _flat(seqs: list[np.ndarray]) -> np.ndarray:
        # Sequences are already int16 numpy arrays; concatenate directly.
        return np.concatenate(seqs) if seqs else np.array([], dtype=np.int16)

    train_arr = _flat(train_seqs)
    val_arr   = _flat(val_seqs)

    train_path = output_dir / "train.npy"
    val_path   = output_dir / "val.npy"
    np.save(train_path, train_arr)
    np.save(val_path,   val_arr)

    meta = {
        "tokenizer":          "v4",
        "vocab_size":         VOCAB_SIZE,
        "seq_len":            seq_len,
        "total_vgm_files":    len(vgm_files),
        "encoded_files":      n_encoded_files,
        "filtered_files":     n_filtered,
        "error_files":        n_error,
        "augment_keys":       augment_keys,
        "augmented_seqs":     len(all_seqs),
        "train_seqs":         len(train_seqs),
        "val_seqs":           len(val_seqs),
        "train_tokens":       int(len(train_arr)),
        "val_tokens":         int(len(val_arr)),
        "patch_library":      str(patch_lib_path),
        "composer_map":       str(composer_map_path),
        "num_composers":      len(composer_map),
        "dac_slot_map":       str(dac_slot_map_path),
        "dac_slots_assigned": len(dac_slot_map),
        "dac_drum_kit":       str(dac_slot_map_path.parent / "dac_drum_kit_v4.json"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info(
        "Saved: train=%dM tokens, val=%dM tokens → %s",
        len(train_arr) // 1_000_000, len(val_arr) // 1_000_000, output_dir,
    )
    return meta


# ---------------------------------------------------------------------------
# DataLoader factory (mirrors load_datasets from dataset.py)
# ---------------------------------------------------------------------------

def load_datasets_v4(
    data_dir: Path | str,
    seq_len: int = 16384,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple:
    """Load pre-prepared v4 train/val datasets and create DataLoaders.

    Args:
        data_dir:    Directory with train.npy, val.npy, meta.json (from
                     prepare_dataset_v4).
        seq_len:     Context window size (default 16384 for v4).
        batch_size:  Batch size for training.
        num_workers: DataLoader workers (0 = main process; required on
                     Windows with memmap).

    Returns:
        (train_loader, val_loader, meta_dict)
    """
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    train_tokens = np.load(data_dir / "train.npy", mmap_mode="r")
    val_tokens   = np.load(data_dir / "val.npy",   mmap_mode="r")

    train_ds = VgmDatasetV4(train_tokens, seq_len=seq_len)
    val_ds   = VgmDatasetV4(val_tokens,   seq_len=seq_len)

    # num_workers=0 required on Windows: memmap arrays can't be pickled across
    # spawned worker processes (OSError/UnpicklingError). The dataset is already
    # memory-mapped so main-process loading is fast enough.
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
    return train_loader, val_loader, meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Prepare v4 training dataset from a VGM corpus."
    )
    parser.add_argument("--vgm-dir",   required=True,
                        help="Directory containing .vgm/.vgz files")
    parser.add_argument("--out-dir",   required=True,
                        help="Output directory for train.npy / val.npy")
    parser.add_argument("--patch-lib", required=True,
                        help="Path for patch library JSON (built if absent)")
    parser.add_argument("--composer-map", default=None,
                        help="Path for composer map JSON (built if absent; "
                             "defaults to out-dir/composer_map_v4.json)")
    parser.add_argument("--dac-slot-map", default=None,
                        help="Path for DAC slot map JSON (built if absent; "
                             "defaults to out-dir/dac_slot_map_v4.json)")
    parser.add_argument("--seq-len",   type=int, default=16384,
                        help="Context window size (default 16384)")
    parser.add_argument("--val-frac",  type=float, default=0.05,
                        help="Validation fraction (default 0.05)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable 12-key transposition augmentation")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on VGM files (for debugging)")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Parallel worker processes (default 12)")
    args = parser.parse_args()

    meta = prepare_dataset_v4(
        vgm_dir            = args.vgm_dir,
        output_dir         = args.out_dir,
        patch_lib_path     = args.patch_lib,
        composer_map_path  = args.composer_map,
        dac_slot_map_path  = args.dac_slot_map,
        seq_len            = args.seq_len,
        val_fraction       = args.val_frac,
        augment_keys       = not args.no_augment,
        max_files          = args.max_files,
        num_workers        = args.num_workers,
    )

    print("\nDataset summary:")
    for k, v in meta.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    _main()
