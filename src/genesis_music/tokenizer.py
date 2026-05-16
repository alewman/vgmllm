"""Tokenizer for VGM event sequences (v3 — legacy).

.. deprecated::
    This is the v3 (data-driven, large-vocab) tokenizer used for the original
    proof-of-concept training runs.  New work should use ``tokenizer_v6.py``
    which uses a smaller, lossless musical-concept vocabulary (~660 tokens).
    This module is kept for reproducibility of v3 checkpoints.

Converts VGM events (register writes, waits, DAC) into integer token
sequences suitable for transformer training, and decodes them back.

Vocabulary is built data-driven from a corpus of parsed VGM files.
Wait times are quantized into logarithmic bins.
"""

from __future__ import annotations

import json
import math
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np

from .vgm_parser import EventType, VgmEvent, VgmFile, load_vgm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special token IDs (always first in vocab)
# ---------------------------------------------------------------------------
PAD = 0
BOS = 1
EOS = 2
UNK = 3

SPECIAL_TOKENS = {
    "<PAD>": PAD,
    "<BOS>": BOS,
    "<EOS>": EOS,
    "<UNK>": UNK,
}

# ---------------------------------------------------------------------------
# Wait-time quantization
# ---------------------------------------------------------------------------

def _build_wait_bins(n_bins: int = 64, max_samples: int = 3_000_000) -> np.ndarray:
    """Build logarithmically-spaced wait bins from 1 to *max_samples*.

    Returns an array of bin *edges* (length n_bins).  To quantize a wait
    value, find the nearest bin edge.
    """
    bins = np.unique(
        np.round(np.geomspace(1, max_samples, n_bins)).astype(np.int64)
    )
    return bins


def quantize_wait(samples: int, bins: np.ndarray) -> int:
    """Return the index of the nearest bin for *samples*."""
    idx = int(np.searchsorted(bins, samples))
    if idx >= len(bins):
        return len(bins) - 1
    if idx > 0:
        # Pick whichever bin edge is closer
        if abs(int(bins[idx]) - samples) > abs(int(bins[idx - 1]) - samples):
            return idx - 1
    return idx


def dequantize_wait(bin_idx: int, bins: np.ndarray) -> int:
    """Return the sample count for a given bin index."""
    return int(bins[min(bin_idx, len(bins) - 1)])


# ---------------------------------------------------------------------------
# Event token: compact representation of a register-write event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventToken:
    """Hashable key for a non-wait VGM event."""
    event_type: str   # EventType.name
    register: int     # 0 for SN76489 / DAC
    value: int

    def to_str(self) -> str:
        return f"{self.event_type}:{self.register:02X}:{self.value:02X}"

    @staticmethod
    def from_str(s: str) -> EventToken:
        parts = s.split(":")
        return EventToken(parts[0], int(parts[1], 16), int(parts[2], 16))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    """Token vocabulary mapping between symbolic events and integer IDs."""
    wait_bins: np.ndarray
    # Forward mapping: token key → id
    token_to_id: dict[str, int] = field(default_factory=dict)
    # Reverse mapping: id → token key
    id_to_token: dict[int, str] = field(default_factory=dict)

    # Derived ranges (set after build)
    wait_offset: int = 0   # first wait token ID
    event_offset: int = 0  # first event token ID

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def n_wait_tokens(self) -> int:
        return len(self.wait_bins)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        obj = {
            "wait_bins": self.wait_bins.tolist(),
            "token_to_id": self.token_to_id,
            "wait_offset": self.wait_offset,
            "event_offset": self.event_offset,
        }
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Vocab:
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        bins = np.array(obj["wait_bins"], dtype=np.int64)
        v = cls(wait_bins=bins)
        v.token_to_id = obj["token_to_id"]
        v.id_to_token = {int(i): t for t, i in v.token_to_id.items()}
        v.wait_offset = obj["wait_offset"]
        v.event_offset = obj["event_offset"]
        return v

    def encode_event(self, event: VgmEvent) -> int | None:
        """Map a single VgmEvent to a token ID, or None if unknown."""
        if event.type == EventType.WAIT:
            idx = quantize_wait(event.value, self.wait_bins)
            return self.wait_offset + idx
        if event.type == EventType.END:
            return None  # handled as EOS at sequence level

        tok = EventToken(
            event_type=event.type.name,
            register=event.register,
            value=event.value,
        )
        tid = self.token_to_id.get(tok.to_str())
        return tid if tid is not None else UNK

    def decode_token(self, token_id: int) -> VgmEvent | str:
        """Decode a token ID back to a VgmEvent or special token name.

        Returns a VgmEvent for data tokens, or a string like '<BOS>'
        for special tokens.
        """
        if token_id in (PAD, BOS, EOS, UNK):
            return self.id_to_token.get(token_id, "<UNK>")

        # Wait token?
        if self.wait_offset <= token_id < self.event_offset:
            bin_idx = token_id - self.wait_offset
            samples = dequantize_wait(bin_idx, self.wait_bins)
            return VgmEvent(type=EventType.WAIT, value=samples)

        # Event token
        key = self.id_to_token.get(token_id)
        if key is None:
            return "<UNK>"
        tok = EventToken.from_str(key)
        etype = EventType[tok.event_type]
        return VgmEvent(type=etype, register=tok.register, value=tok.value)


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _extract_event_tokens_from_file(
    fpath: Path | str,
    *,
    include_dac: bool,
    include_sn76489: bool,
) -> dict[str, int] | None:
    """Worker: load one VGM and return a counter dict of event token strings."""
    try:
        vgm = load_vgm(fpath)
    except Exception:
        return None

    counts: dict[str, int] = {}
    for ev in vgm.events:
        if ev.type in (EventType.WAIT, EventType.END):
            continue
        if ev.type == EventType.DAC_WRITE and not include_dac:
            continue
        if ev.type == EventType.SN76489 and not include_sn76489:
            continue

        tok = EventToken(
            event_type=ev.type.name,
            register=ev.register,
            value=ev.value,
        )
        key = tok.to_str()
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(
    vgm_files: Sequence[Path | str],
    n_wait_bins: int = 64,
    min_count: int = 2,
    include_dac: bool = False,
    include_sn76489: bool = True,
    max_tokens: int | None = None,
) -> Vocab:
    """Scan VGM files and build a data-driven vocabulary.

    Args:
        vgm_files: Paths to .vgm / .vgz files.
        n_wait_bins: Number of logarithmic wait-time bins.
        min_count: Minimum times a (type, reg, val) must appear to get
            its own token.  Rare events map to UNK.
        include_dac: Whether to include DAC_WRITE events as tokens.
        include_sn76489: Whether to include SN76489 writes.
        max_tokens: Optional cap on event tokens (by frequency).

    Returns:
        A fully-built Vocab.
    """
    event_counts: Counter[str] = Counter()
    n_workers = max(1, min(os.cpu_count() or 1, 12))
    log.info("  Using %d workers", n_workers)

    worker = partial(_extract_event_tokens_from_file,
                     include_dac=include_dac,
                     include_sn76489=include_sn76489)
    done = 0

    with Pool(n_workers) as pool:
        for file_counts in pool.imap_unordered(worker, vgm_files, chunksize=64):
            done += 1
            if file_counts is not None:
                for tok_str, cnt in file_counts.items():
                    event_counts[tok_str] += cnt

            if done % 500 == 0:
                log.info("  Scanned %d/%d files, %d unique tokens so far",
                         done, len(vgm_files), len(event_counts))

    # Filter by min_count
    filtered = {k: c for k, c in event_counts.items() if c >= min_count}
    log.info("Corpus tokens: %d total unique, %d after min_count=%d filter",
             len(event_counts), len(filtered), min_count)

    # Sort by frequency (most common first)
    sorted_tokens = sorted(filtered, key=lambda k: filtered[k], reverse=True)

    # Apply max_tokens cap
    if max_tokens is not None and len(sorted_tokens) > max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]
        log.info("Capped to top %d event tokens", max_tokens)

    # Build wait bins
    wait_bins = _build_wait_bins(n_wait_bins)

    # Assign IDs
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    # Special tokens first
    for name, tid in SPECIAL_TOKENS.items():
        token_to_id[name] = tid
        id_to_token[tid] = name

    # Wait tokens
    wait_offset = len(SPECIAL_TOKENS)
    for i in range(len(wait_bins)):
        name = f"<WAIT:{i}>"
        tid = wait_offset + i
        token_to_id[name] = tid
        id_to_token[tid] = name

    # Event tokens
    event_offset = wait_offset + len(wait_bins)
    for i, tok_str in enumerate(sorted_tokens):
        tid = event_offset + i
        token_to_id[tok_str] = tid
        id_to_token[tid] = tok_str

    vocab = Vocab(
        wait_bins=wait_bins,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        wait_offset=wait_offset,
        event_offset=event_offset,
    )

    log.info("Vocab built: %d total tokens (%d special + %d wait + %d event)",
             vocab.size, len(SPECIAL_TOKENS), len(wait_bins), len(sorted_tokens))

    return vocab


# ---------------------------------------------------------------------------
# Encode / decode full sequences
# ---------------------------------------------------------------------------

def encode_vgm(vgm: VgmFile, vocab: Vocab, include_dac: bool = False) -> list[int]:
    """Encode a parsed VGM file into a list of token IDs.

    Wraps the sequence with BOS / EOS tokens.
    Events that map to UNK are dropped (not inserted).
    Consecutive waits are merged after filtering (e.g. DAC removal).
    """
    tokens = [BOS]
    pending_wait = 0  # accumulate wait samples to merge after DAC removal

    for ev in vgm.events:
        if ev.type == EventType.END:
            break
        if ev.type == EventType.DAC_WRITE and not include_dac:
            continue

        if ev.type == EventType.WAIT:
            pending_wait += ev.value
            continue

        # Flush any accumulated wait before this non-wait event
        if pending_wait > 0:
            tid = vocab.encode_event(VgmEvent(type=EventType.WAIT, value=pending_wait))
            if tid is not None:
                tokens.append(tid)
            pending_wait = 0

        tid = vocab.encode_event(ev)
        if tid is not None and tid != UNK:
            tokens.append(tid)

    # Flush trailing wait
    if pending_wait > 0:
        tid = vocab.encode_event(VgmEvent(type=EventType.WAIT, value=pending_wait))
        if tid is not None:
            tokens.append(tid)

    tokens.append(EOS)
    return tokens


def decode_tokens(token_ids: list[int], vocab: Vocab) -> list[VgmEvent]:
    """Decode a token sequence back to VgmEvents.

    Strips BOS/EOS/PAD/UNK tokens and returns the event list.
    """
    events = []
    for tid in token_ids:
        result = vocab.decode_token(tid)
        if isinstance(result, VgmEvent):
            events.append(result)
    # Add END event
    events.append(VgmEvent(type=EventType.END))
    return events


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

    parser = argparse.ArgumentParser(
        description="Build or inspect VGM tokenizer vocabulary"
    )
    sub = parser.add_subparsers(dest="command")

    # build-vocab
    bv = sub.add_parser("build-vocab", help="Scan VGM corpus and build vocabulary")
    bv.add_argument("--vgm-dir", type=Path, default=Path("data/vgm"))
    bv.add_argument("--output", type=Path, default=Path("data/vocab.json"))
    bv.add_argument("--n-wait-bins", type=int, default=64)
    bv.add_argument("--min-count", type=int, default=2)
    bv.add_argument("--max-tokens", type=int, default=None)
    bv.add_argument("--include-dac", action="store_true")
    bv.add_argument("--include-sn76489", action="store_true", default=True)

    # inspect
    ins = sub.add_parser("inspect", help="Print vocabulary stats")
    ins.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))

    # encode
    enc = sub.add_parser("encode", help="Encode a VGM file and print token stats")
    enc.add_argument("file", type=Path)
    enc.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))

    args = parser.parse_args()

    if args.command == "build-vocab":
        files = sorted(args.vgm_dir.glob("*.vg*"))
        if not files:
            log.error("No VGM files found in %s", args.vgm_dir)
            return
        log.info("Building vocab from %d files...", len(files))
        vocab = build_vocab(
            files,
            n_wait_bins=args.n_wait_bins,
            min_count=args.min_count,
            include_dac=args.include_dac,
            include_sn76489=args.include_sn76489,
            max_tokens=args.max_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(args.output)
        log.info("Saved vocabulary to %s", args.output)

    elif args.command == "inspect":
        vocab = Vocab.load(args.vocab)
        print(f"Vocabulary size:  {vocab.size}")
        print(f"Wait bins:        {vocab.n_wait_tokens}")
        print(f"Wait offset:      {vocab.wait_offset}")
        print(f"Event offset:     {vocab.event_offset}")
        print(f"Event tokens:     {vocab.size - vocab.event_offset}")
        print(f"\nWait bin edges (samples):")
        for i, b in enumerate(vocab.wait_bins):
            ms = b / 44.1
            print(f"  [{i:3d}] {b:>10,} samples  ({ms:>10.1f} ms)")

    elif args.command == "encode":
        vocab = Vocab.load(args.vocab)
        vgm = load_vgm(args.file)
        tokens = encode_vgm(vgm, vocab)
        print(f"File:     {args.file}")
        print(f"Events:   {len(vgm.events):,}")
        print(f"Tokens:   {len(tokens):,}")
        print(f"Ratio:    {len(tokens)/len(vgm.events):.2f}x")
        print(f"Duration: {vgm.header.duration_seconds:.1f}s")

        # Token type breakdown
        n_wait = sum(1 for t in tokens if vocab.wait_offset <= t < vocab.event_offset)
        n_event = sum(1 for t in tokens if t >= vocab.event_offset)
        n_special = sum(1 for t in tokens if t < vocab.wait_offset)
        print(f"\nBreakdown: {n_special} special, {n_wait} wait, {n_event} event")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
