"""Dataset for training: tokenized VGM sequences chunked to fixed windows.

Handles the full pipeline from raw VGM files to training-ready tensors:
    VGM files → tokenize → chunk into windows → train/val split → DataLoader
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .vgm_parser import load_vgm
from .tokenizer import Vocab, encode_vgm, PAD, BOS, EOS
from .tokenizer_v2 import VocabV2, encode_vgm_v2, PAD as PAD_V2
from .tokenizer_v2 import (
    encode_events_v2, name_to_midi, midi_to_name,
    BOS as BOS_V2, EOS as EOS_V2, UNK as UNK_V2,
)

log = logging.getLogger(__name__)


class VgmDataset(Dataset):
    """PyTorch Dataset of fixed-length token windows from VGM files.

    Each item is a dict with:
        input_ids:  (seq_len,) int64 tensor — the token sequence
        labels:     (seq_len,) int64 tensor — shifted by 1 for next-token prediction
                    (input_ids[1:] with padding, labels ignore PAD via -100)
    """

    def __init__(
        self,
        tokens: np.ndarray,
        seq_len: int = 4096,
    ):
        """Create a dataset from a flat token array.

        Args:
            tokens: 1-D numpy array of concatenated token sequences
                (each sequence already has BOS/EOS).  May be a memmap.
            seq_len: Context window size. Sequences are chunked with
                stride = seq_len (no overlap) for training efficiency.
        """
        self.seq_len = seq_len

        # Chunk into non-overlapping windows of (seq_len + 1) tokens
        # The +1 gives us the target for the last position
        chunk_size = seq_len + 1
        n_chunks = len(tokens) // chunk_size
        if n_chunks == 0:
            raise ValueError(
                f"Token array too short ({len(tokens)}) for seq_len={seq_len}"
            )

        # Keep a reference to the flat array (may be a memmap — zero-copy)
        self._tokens = tokens
        self._chunk_size = chunk_size
        self._n_chunks = n_chunks

        log.info(
            "Dataset: %d chunks of %d tokens (%.1fM total tokens)",
            n_chunks, seq_len, n_chunks * seq_len / 1e6,
        )

    def __len__(self) -> int:
        return self._n_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self._chunk_size
        chunk = torch.from_numpy(
            self._tokens[start : start + self._chunk_size].astype(np.int64)
        )
        input_ids = chunk[:-1]   # first seq_len tokens
        labels = chunk[1:].clone()  # shifted by 1

        # Mask PAD tokens in labels so loss ignores them
        labels[labels == PAD] = -100

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _tokenize_one_file(
    fpath: Path,
    *,
    vocab_path: Path | None,
    vocab: Vocab | None,
    include_dac: bool,
    min_tokens: int,
) -> list[int] | None:
    """Worker: load + tokenize a single VGM file. Returns tokens or None."""
    try:
        vgm = load_vgm(fpath)
        tokens = encode_vgm(vgm, vocab, include_dac=include_dac)
        if len(tokens) < min_tokens:
            return None
        return tokens
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset preparation: tokenize corpus → numpy arrays → train/val split
# ---------------------------------------------------------------------------

def prepare_dataset(
    vgm_dir: Path | str,
    vocab: Vocab,
    output_dir: Path | str,
    seq_len: int = 4096,
    val_fraction: float = 0.05,
    include_dac: bool = False,
    min_tokens: int = 32,
    max_files: int | None = None,
) -> dict:
    """Tokenize all VGM files and save as memory-mapped numpy arrays.

    Args:
        vgm_dir: Directory containing .vgm/.vgz files.
        vocab: Built vocabulary for encoding.
        output_dir: Where to save train.npy, val.npy, and metadata.
        seq_len: Context window size for chunking.
        val_fraction: Fraction of sequences for validation.
        include_dac: Whether to include DAC_WRITE tokens.
        min_tokens: Skip files that produce fewer tokens than this.
        max_files: Limit number of files (for testing).

    Returns:
        Dict with dataset statistics.
    """
    vgm_dir = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all VGM files
    vgm_files = sorted(
        list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz"))
    )
    if max_files is not None:
        vgm_files = vgm_files[:max_files]

    if not vgm_files:
        raise FileNotFoundError(f"No VGM files in {vgm_dir}")

    n_workers = max(1, min(os.cpu_count() or 1, 12))
    log.info("Tokenizing %d VGM files with %d workers...",
             len(vgm_files), n_workers)

    # Tokenize all files in parallel
    all_tokens: list[list[int]] = []
    total_tokens = 0
    skipped = 0
    done = 0

    worker = partial(_tokenize_one_file,
                     vocab_path=None, vocab=vocab,
                     include_dac=include_dac, min_tokens=min_tokens)

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker, vgm_files, chunksize=64):
            done += 1
            if result is None:
                skipped += 1
            else:
                all_tokens.append(result)
                total_tokens += len(result)

            if done % 2000 == 0:
                log.info(
                    "  Tokenized %d/%d files (%d tokens so far)",
                    done, len(vgm_files), total_tokens,
                )

    log.info(
        "Tokenized %d files (%d skipped), %d total tokens",
        len(all_tokens), skipped, total_tokens,
    )

    if not all_tokens:
        raise ValueError("No files produced valid token sequences")

    # Shuffle sequences for train/val split (at file level)
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(all_tokens))

    n_val = max(1, int(len(all_tokens) * val_fraction))
    val_indices = set(indices[:n_val])

    train_seqs: list[list[int]] = []
    val_seqs: list[list[int]] = []
    for i, seq in enumerate(all_tokens):
        if i in val_indices:
            val_seqs.append(seq)
        else:
            train_seqs.append(seq)

    # Concatenate into flat arrays
    train_tokens = np.concatenate(
        [np.array(s, dtype=np.int32) for s in train_seqs]
    )
    val_tokens = np.concatenate(
        [np.array(s, dtype=np.int32) for s in val_seqs]
    )

    # Save
    train_path = output_dir / "train.npy"
    val_path = output_dir / "val.npy"
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    # Save metadata
    meta = {
        "total_files": len(all_tokens),
        "skipped_files": skipped,
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_files": len(train_seqs),
        "val_files": len(val_seqs),
        "vocab_size": vocab.size,
        "seq_len": seq_len,
        "include_dac": include_dac,
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        "Saved: train=%d tokens, val=%d tokens → %s",
        len(train_tokens), len(val_tokens), output_dir,
    )

    return meta


# ---------------------------------------------------------------------------
# v2 worker + prepare_dataset_v2
# ---------------------------------------------------------------------------

def _tokenize_one_file_v2(
    fpath: Path,
    *,
    vocab: VocabV2 | None,
    include_dac: bool,
    min_tokens: int,
) -> list[int] | None:
    """Worker: load + tokenize a single VGM file with v2 tokenizer."""
    try:
        vgm = load_vgm(fpath)
        tokens = encode_vgm_v2(vgm, vocab, include_dac=include_dac)
        if len(tokens) < min_tokens:
            return None
        return tokens
    except Exception:
        return None


def prepare_dataset_v2(
    vgm_dir: Path | str,
    vocab: VocabV2,
    output_dir: Path | str,
    seq_len: int = 4096,
    val_fraction: float = 0.05,
    include_dac: bool = False,
    min_tokens: int = 32,
    max_files: int | None = None,
) -> dict:
    """Tokenize all VGM files with v2 tokenizer and save as numpy arrays."""
    vgm_dir = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vgm_files = sorted(
        list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz"))
    )
    if max_files is not None:
        vgm_files = vgm_files[:max_files]

    if not vgm_files:
        raise FileNotFoundError(f"No VGM files in {vgm_dir}")

    n_workers = max(1, min(os.cpu_count() or 1, 12))
    log.info("Tokenizing %d VGM files with v2 tokenizer (%d workers)...",
             len(vgm_files), n_workers)

    all_tokens: list[list[int]] = []
    total_tokens = 0
    skipped = 0
    done = 0

    worker = partial(_tokenize_one_file_v2,
                     vocab=vocab,
                     include_dac=include_dac, min_tokens=min_tokens)

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker, vgm_files, chunksize=64):
            done += 1
            if result is None:
                skipped += 1
            else:
                all_tokens.append(result)
                total_tokens += len(result)

            if done % 2000 == 0:
                log.info(
                    "  Tokenized %d/%d files (%d tokens so far)",
                    done, len(vgm_files), total_tokens,
                )

    log.info(
        "v2 tokenized %d files (%d skipped), %d total tokens",
        len(all_tokens), skipped, total_tokens,
    )

    if not all_tokens:
        raise ValueError("No files produced valid token sequences")

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(all_tokens))

    n_val = max(1, int(len(all_tokens) * val_fraction))
    val_indices = set(indices[:n_val])

    train_seqs: list[list[int]] = []
    val_seqs: list[list[int]] = []
    for i, seq in enumerate(all_tokens):
        if i in val_indices:
            val_seqs.append(seq)
        else:
            train_seqs.append(seq)

    train_tokens = np.concatenate(
        [np.array(s, dtype=np.int32) for s in train_seqs]
    )
    val_tokens = np.concatenate(
        [np.array(s, dtype=np.int32) for s in val_seqs]
    )

    np.save(output_dir / "train.npy", train_tokens)
    np.save(output_dir / "val.npy", val_tokens)

    meta = {
        "total_files": len(all_tokens),
        "skipped_files": skipped,
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_files": len(train_seqs),
        "val_files": len(val_seqs),
        "vocab_size": vocab.size,
        "seq_len": seq_len,
        "include_dac": include_dac,
        "tokenizer_version": 2,
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    log.info(
        "v2 saved: train=%d tokens, val=%d tokens → %s",
        len(train_tokens), len(val_tokens), output_dir,
    )

    return meta


class ResumableSampler(torch.utils.data.Sampler):
    """Shuffling sampler that can seek to a position without loading data.

    Uses a fixed seed to produce a deterministic permutation, then slices
    it at ``start_idx`` on resume. This means the data order is identical
    whether training runs continuously or is interrupted and resumed.
    """

    def __init__(self, n: int, seed: int = 42, start_idx: int = 0):
        self.n = n
        self.seed = seed
        self.start_idx = start_idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(self.n, generator=g).tolist()
        return iter(indices[self.start_idx:])

    def __len__(self) -> int:
        return self.n - self.start_idx


def load_datasets(
    data_dir: Path | str,
    seq_len: int = 4096,
    batch_size: int = 8,
    num_workers: int = 2,
    start_batch: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Load pre-prepared train/val datasets and create DataLoaders.

    Args:
        data_dir: Directory containing train.npy, val.npy, meta.json.
        seq_len: Context window size.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        start_batch: Number of batches already consumed (for resuming).
            The sampler will seek past these without loading any data.

    Returns:
        (train_loader, val_loader, metadata_dict)
    """
    data_dir = Path(data_dir)

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    train_tokens = np.load(data_dir / "train.npy", mmap_mode="r")
    val_tokens = np.load(data_dir / "val.npy", mmap_mode="r")

    train_ds = VgmDataset(train_tokens, seq_len=seq_len)
    val_ds = VgmDataset(val_tokens, seq_len=seq_len)

    # ResumableSampler seeks to the right position instantly (no data loaded)
    # by slicing a deterministic permutation. start_idx = consumed samples.
    start_idx = start_batch * batch_size
    sampler = ResumableSampler(len(train_ds), seed=42, start_idx=start_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # memmap can't be pickled on Windows
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, meta


# ---------------------------------------------------------------------------
# v3: filtered + transposition augmentation
# ---------------------------------------------------------------------------

def _transpose_token_strs(token_strs: list[str], semitones: int) -> list[str]:
    """Shift all note tokens by `semitones` semitones.

    Tokens like CH1:ON:C4, CH3:PITCH:Eb5 get their note shifted.
    Other tokens pass through unchanged.
    Returns None if any transposed note goes out of valid MIDI range [21, 108].
    """
    result = []
    for t in token_strs:
        if ":ON:" in t:
            parts = t.split(":")
            note_name = parts[2]
            if note_name == "X":
                result.append(t)
                continue
            midi = name_to_midi(note_name) + semitones
            if midi < 21 or midi > 108:
                return None  # out of range, skip this transposition
            parts[2] = midi_to_name(midi)
            result.append(":".join(parts))
        elif ":PITCH:" in t:
            parts = t.split(":")
            note_name = parts[2]
            midi = name_to_midi(note_name) + semitones
            if midi < 21 or midi > 108:
                return None
            parts[2] = midi_to_name(midi)
            result.append(":".join(parts))
        else:
            result.append(t)
    return result


def _tokenize_one_file_v3(
    fpath: Path,
    *,
    vocab: VocabV2 | None,
    include_dac: bool,
    min_tokens: int,
    transpose_range: tuple[int, int],
) -> tuple[str, list[np.ndarray]] | None:
    """Worker: tokenize with v2, return (path, list of int32 arrays)."""
    try:
        vgm = load_vgm(fpath)
        # Get string tokens first for transposition
        token_strs = encode_events_v2(vgm.events, include_dac=include_dac)

        # Convert string tokens to int32 numpy array
        def strs_to_ids(strs):
            ids = [BOS_V2]
            for t in strs:
                tid = vocab.encode(t)
                if tid != UNK_V2:
                    ids.append(tid)
            ids.append(EOS_V2)
            return np.array(ids, dtype=np.int32)

        original = strs_to_ids(token_strs)
        if len(original) < min_tokens:
            return None

        results = [original]

        # Transposition augmentation
        lo, hi = transpose_range
        for semitones in range(lo, hi + 1):
            if semitones == 0:
                continue
            shifted = _transpose_token_strs(token_strs, semitones)
            if shifted is not None:
                ids = strs_to_ids(shifted)
                if len(ids) >= min_tokens:
                    results.append(ids)

        return (str(fpath), results)
    except Exception:
        return None


def _file_is_val(fpath_str: str, val_fraction: float) -> bool:
    """Deterministic train/val assignment by file path (hash-based)."""
    h = int(hashlib.md5(fpath_str.encode()).hexdigest()[:8], 16)
    return (h % 10000) < int(val_fraction * 10000)


def _raw_to_npy(raw_path: Path, npy_path: Path, n_tokens: int):
    """Convert raw int32 binary to .npy via memory-mapped write (low RAM)."""
    out = np.lib.format.open_memmap(
        str(npy_path), mode='w+', dtype=np.int32, shape=(n_tokens,)
    )
    chunk_size = 10_000_000  # 10M tokens = 40 MB per chunk
    with open(raw_path, 'rb') as f:
        offset = 0
        while offset < n_tokens:
            n_read = min(chunk_size, n_tokens - offset)
            raw = f.read(n_read * 4)
            chunk = np.frombuffer(raw, dtype=np.int32)
            out[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
    out.flush()
    del out


def prepare_dataset_v3(
    vgm_dir: Path | str,
    vocab: VocabV2,
    output_dir: Path | str,
    seq_len: int = 16384,
    val_fraction: float = 0.05,
    include_dac: bool = False,
    min_tokens: int = 32,
    max_files: int | None = None,
    filter_file: Path | str | None = None,
    min_duration: float = 5.0,
    max_duration: float | None = None,
    require_fm_notes: bool = True,
    transpose_range: tuple[int, int] = (-5, 6),
) -> dict:
    """Tokenize VGM files with v2 tokenizer, SFX filtering, and transposition.

    Streams results to disk to avoid OOM with large augmented datasets.

    Args:
        filter_file: Path to corpus_analysis.json for duration/SFX filtering.
        min_duration: Minimum duration in seconds (files shorter are skipped).
        max_duration: Maximum duration in seconds (files longer are skipped).
        require_fm_notes: If True, skip files with no FM key-on events.
        transpose_range: (lo, hi) semitone range for augmentation.
            E.g. (-5, 6) gives up to 11 transpositions + original = 12x.
    """
    vgm_dir = Path(vgm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filter data
    filter_data = None
    if filter_file is not None:
        filter_file = Path(filter_file)
        if filter_file.exists():
            with open(filter_file) as f:
                filter_data = json.load(f)
            log.info("Loaded filter data for %d files", len(filter_data))

    # Gather and filter VGM files
    vgm_files = sorted(
        list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz"))
    )

    if filter_data is not None:
        pre_filter = len(vgm_files)
        filtered = []
        for f in vgm_files:
            key = str(f)
            if key not in filter_data:
                filtered.append(f)  # no filter data = include
                continue
            info = filter_data[key]
            if require_fm_notes and not info.get("has_fm_notes", True):
                continue
            dur = info.get("duration", 999)
            if dur < min_duration:
                continue
            if max_duration is not None and dur > max_duration:
                continue
            filtered.append(f)
        vgm_files = filtered
        log.info("Filtered: %d → %d files (removed %d SFX/short/long)",
                 pre_filter, len(vgm_files), pre_filter - len(vgm_files))

    if max_files is not None:
        vgm_files = vgm_files[:max_files]

    if not vgm_files:
        raise FileNotFoundError(f"No VGM files after filtering in {vgm_dir}")

    n_workers = max(1, min(os.cpu_count() or 1, 8))
    log.info("Tokenizing %d VGM files with v3 pipeline (%d workers), "
             "transpose=(%d,%d)...",
             len(vgm_files), n_workers, transpose_range[0], transpose_range[1])

    total_tokens = 0
    train_token_count = 0
    val_token_count = 0
    n_seqs = 0
    train_seqs = 0
    val_seqs = 0
    skipped = 0
    done = 0
    n_augmented = 0

    worker = partial(_tokenize_one_file_v3,
                     vocab=vocab,
                     include_dac=include_dac,
                     min_tokens=min_tokens,
                     transpose_range=transpose_range)

    train_bin = output_dir / "_train.bin"
    val_bin = output_dir / "_val.bin"

    with open(train_bin, "wb") as tf, open(val_bin, "wb") as vf:
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(worker, vgm_files, chunksize=1):
                done += 1
                if result is None:
                    skipped += 1
                else:
                    fpath_str, sequences = result
                    is_val = _file_is_val(fpath_str, val_fraction)
                    for arr in sequences:
                        data = arr.tobytes()
                        n_tok = len(arr)
                        if is_val:
                            vf.write(data)
                            val_token_count += n_tok
                            val_seqs += 1
                        else:
                            tf.write(data)
                            train_token_count += n_tok
                            train_seqs += 1
                        total_tokens += n_tok
                        n_seqs += 1
                    if len(sequences) > 1:
                        n_augmented += len(sequences) - 1

                if done % 500 == 0:
                    log.info(
                        "  Tokenized %d/%d files (%d seqs, %s tokens)",
                        done, len(vgm_files), n_seqs, f"{total_tokens:,}",
                    )

    log.info(
        "v3 tokenized %d files → %d sequences (%d augmented, %d skipped), "
        "%s total tokens",
        done - skipped, n_seqs, n_augmented, skipped, f"{total_tokens:,}",
    )

    if total_tokens == 0:
        raise ValueError("No files produced valid token sequences")

    # Convert streaming binary files to .npy via memory-mapped writes
    log.info("Writing train.npy (%s tokens)...", f"{train_token_count:,}")
    _raw_to_npy(train_bin, output_dir / "train.npy", train_token_count)
    train_bin.unlink()

    log.info("Writing val.npy (%s tokens)...", f"{val_token_count:,}")
    _raw_to_npy(val_bin, output_dir / "val.npy", val_token_count)
    val_bin.unlink()

    meta = {
        "total_files": done - skipped,
        "total_sequences": n_seqs,
        "augmented_sequences": n_augmented,
        "skipped_files": skipped,
        "total_tokens": total_tokens,
        "train_tokens": train_token_count,
        "val_tokens": val_token_count,
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "vocab_size": vocab.size,
        "seq_len": seq_len,
        "include_dac": include_dac,
        "tokenizer_version": 3,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "require_fm_notes": require_fm_notes,
        "transpose_range": list(transpose_range),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    log.info(
        "v3 saved: train=%s tokens, val=%s tokens → %s",
        f"{train_token_count:,}", f"{val_token_count:,}", output_dir,
    )

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Prepare VGM training dataset")
    sub = parser.add_subparsers(dest="command")

    # prepare
    prep = sub.add_parser("prepare", help="Tokenize VGMs into train/val arrays")
    prep.add_argument("--vgm-dir", type=Path, default=Path("data/vgm"))
    prep.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    prep.add_argument("--output", type=Path, default=Path("data/prepared"))
    prep.add_argument("--seq-len", type=int, default=4096)
    prep.add_argument("--val-fraction", type=float, default=0.05)
    prep.add_argument("--include-dac", action="store_true")
    prep.add_argument("--max-files", type=int, default=None)

    # prepare v2
    prep2 = sub.add_parser("prepare-v2", help="Tokenize VGMs with v2 tokenizer")
    prep2.add_argument("--vgm-dir", type=Path, default=Path("data/vgm"))
    prep2.add_argument("--vocab", type=Path, default=Path("data/vocab_v2.json"))
    prep2.add_argument("--output", type=Path, default=Path("data/prepared_v2"))
    prep2.add_argument("--seq-len", type=int, default=4096)
    prep2.add_argument("--val-fraction", type=float, default=0.05)
    prep2.add_argument("--include-dac", action="store_true")
    prep2.add_argument("--max-files", type=int, default=None)

    # prepare v3
    prep3 = sub.add_parser("prepare-v3",
                           help="v2 tokenizer + SFX filter + transposition")
    prep3.add_argument("--vgm-dir", type=Path, default=Path("data/vgm"))
    prep3.add_argument("--vocab", type=Path, default=Path("data/vocab_v2.json"))
    prep3.add_argument("--output", type=Path, default=Path("data/prepared_v3"))
    prep3.add_argument("--seq-len", type=int, default=16384)
    prep3.add_argument("--val-fraction", type=float, default=0.05)
    prep3.add_argument("--include-dac", action="store_true")
    prep3.add_argument("--max-files", type=int, default=None)
    prep3.add_argument("--filter-file", type=Path,
                       default=Path("data/corpus_analysis.json"))
    prep3.add_argument("--min-duration", type=float, default=5.0)
    prep3.add_argument("--no-require-fm-notes", action="store_true")
    prep3.add_argument("--max-duration", type=float, default=None)
    prep3.add_argument("--transpose-lo", type=int, default=-5)
    prep3.add_argument("--transpose-hi", type=int, default=6)

    # info
    info = sub.add_parser("info", help="Show prepared dataset info")
    info.add_argument("--data-dir", type=Path, default=Path("data/prepared"))

    args = parser.parse_args()

    if args.command == "prepare":
        vocab = Vocab.load(args.vocab)
        prepare_dataset(
            vgm_dir=args.vgm_dir,
            vocab=vocab,
            output_dir=args.output,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            include_dac=args.include_dac,
            max_files=args.max_files,
        )

    elif args.command == "prepare-v2":
        vocab = VocabV2.load(args.vocab)
        prepare_dataset_v2(
            vgm_dir=args.vgm_dir,
            vocab=vocab,
            output_dir=args.output,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            include_dac=args.include_dac,
            max_files=args.max_files,
        )

    elif args.command == "prepare-v3":
        vocab = VocabV2.load(args.vocab)
        prepare_dataset_v3(
            vgm_dir=args.vgm_dir,
            vocab=vocab,
            output_dir=args.output,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            include_dac=args.include_dac,
            max_files=args.max_files,
            filter_file=args.filter_file,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            require_fm_notes=not args.no_require_fm_notes,
            transpose_range=(args.transpose_lo, args.transpose_hi),
        )

    elif args.command == "info":
        meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
        for k, v in meta.items():
            if isinstance(v, int) and v > 10000:
                print(f"  {k:20s}: {v:>12,}")
            else:
                print(f"  {k:20s}: {v}")

        # Show chunk counts
        train_tokens = np.load(args.data_dir / "train.npy")
        val_tokens = np.load(args.data_dir / "val.npy")
        sl = meta["seq_len"]
        print(f"\n  Train chunks ({sl}): {len(train_tokens) // (sl + 1):,}")
        print(f"  Val chunks ({sl}):   {len(val_tokens) // (sl + 1):,}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
