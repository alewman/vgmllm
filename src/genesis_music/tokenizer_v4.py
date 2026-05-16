"""v4 tokenizer — musical note-event token sequences.

Converts decoded VGM NoteEvents into compact integer token sequences where
every token represents a human-understandable musical concept (pitch, beat
position, channel role, FM patch reference, etc.).

Vocabulary: 449 tokens (320 musical + 128 composer + 1 UNK_COMPOSER).
Information density: ~8× higher per token than v3.

Typical usage::

    # One-time: build patch library and composer map from corpus
    lib = PatchLibrary.build(vgm_files)
    lib.save("data/patch_library_v4.json")
    cmap = ComposerMap.build(vgm_files)
    cmap.save("data/composer_map_v4.json")

    # Per-file: encode
    lib = PatchLibrary.load("data/patch_library_v4.json")
    cmap = ComposerMap.load("data/composer_map_v4.json")
    tok = TokenizerV4(lib, composer_map=cmap)
    tokens = tok.encode(vgm_file)

    # Per-file: decode back to VGM
    vgm_file = tok.decode(tokens, reference_vgm=prompt_vgm)
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .music_analysis import (
    ROLE_BASS, ROLE_COUNTER, ROLE_DRUMS, ROLE_HARM,
    ROLE_LEAD, ROLE_PERC, ROLE_UNK,
    SAMPLE_RATE, TEMPO_BINS,
    analyse_vgm, should_discard,
)
from .vgm_parser import VgmFile, load_vgm
from .ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    NoteEvent, Ym2612Patch, decode_vgm,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary layout  (320 tokens total)
# ---------------------------------------------------------------------------
#
# IDs 0–3    Special
# IDs 4–19   Tempo (16 bins)
# IDs 20–43  Key (24: 12 major + 12 minor)
# IDs 44–47  Meter
# IDs 48     BAR
# IDs 49–64  BEAT_0 … BEAT_15 (16th-note positions)
# IDs 65     PHRASE_END
# IDs 66–71  CH_FM_0 … CH_FM_5
# IDs 72     CH_DAC_TOK
# IDs 73–75  CH_PSG_0_TOK … CH_PSG_2_TOK
# IDs 76–82  ROLE tokens (7)
# IDs 83–210 PATCH_0 … PATCH_127
# IDs 211–213 NOTE_ON / NOTE_OFF / NOTE_HOLD
# IDs 214–301 PITCH_0 … PITCH_87  (MIDI 24–111)
# IDs 302–317 VEL_0 … VEL_15
# IDs 318–325 DAC_HIT_0 … DAC_HIT_7 (drum identity slots; 0 = most common)
# IDs 326    PSG_NOISE_HIT
# IDs 327–454 COMPOSER_0 … COMPOSER_127
# IDs 455    UNK_COMPOSER
# VOCAB_SIZE  456
# ---------------------------------------------------------------------------

PAD = 0
BOS = 1
EOS = 2
UNK = 3

TEMPO_BASE = 4          # IDs 4–19  (16 bins)
KEY_BASE   = 20         # IDs 20–43 (24 keys: 12 major then 12 minor)
METER_44   = 44
METER_34   = 45
METER_68   = 46
METER_24   = 47

BAR        = 48
BEAT_BASE  = 49         # IDs 49–64 (positions 0–15 within a bar)
PHRASE_END = 65

CH_FM_BASE  = 66        # IDs 66–71 (FM channels 0–5)
CH_DAC_TOK  = 72
CH_PSG_BASE = 73        # IDs 73–75 (PSG tone channels 0–2)

ROLE_TOKEN_BASE = 76    # IDs 76–82
_ROLE_ORDER = [ROLE_BASS, ROLE_LEAD, ROLE_HARM,
               ROLE_COUNTER, ROLE_DRUMS, ROLE_PERC, ROLE_UNK]
ROLE_TO_TOKEN = {r: ROLE_TOKEN_BASE + i for i, r in enumerate(_ROLE_ORDER)}
TOKEN_TO_ROLE = {v: k for k, v in ROLE_TO_TOKEN.items()}

PATCH_BASE = 83         # IDs 83–210  (128 patch references)
NUM_PATCHES = 128

NOTE_ON   = 211
NOTE_OFF  = 212
NOTE_HOLD = 213

PITCH_BASE     = 214    # IDs 214–301
PITCH_MIN_MIDI = 24     # C1
PITCH_MAX_MIDI = 111    # B7
NUM_PITCHES    = PITCH_MAX_MIDI - PITCH_MIN_MIDI + 1   # 88

VEL_BASE   = 302        # IDs 302–317  (16 levels)

# DAC drum identity slots: 8 tokens for different drum samples (kick, snare, etc.)
# Assigned by frequency rank of pcm_offset in the corpus (slot 0 = most common).
# IDs 318–325  DAC_HIT_0 … DAC_HIT_7
NUM_DAC_SLOTS = 8
DAC_HIT_BASE  = 318
DAC_HIT_UNK   = DAC_HIT_BASE   # fallback when sample not in slot map
# Legacy alias kept so any existing code that imports DAC_HIT still compiles;
# it maps to slot 0 (most common drum, typically kick).
DAC_HIT       = DAC_HIT_BASE

PSG_NOISE_HIT = 326             # shifted up by 7 (was 319)

COMPOSER_BASE = 327             # IDs 327–454  (128 named composer slots; was 320)
NUM_COMPOSERS = 128
UNK_COMPOSER  = COMPOSER_BASE + NUM_COMPOSERS  # ID 455

VOCAB_SIZE = 456                # was 449

# ---------------------------------------------------------------------------
# Composer name helpers
# ---------------------------------------------------------------------------

def _split_composers(raw: str) -> list[str]:
    """Split a compound author string like 'A, B, C' into individual names."""
    parts = re.split(r',|;|\band\b', raw)
    names = []
    for p in parts:
        n = p.strip()
        if n and len(n) > 1 and not n.startswith('('):
            n = re.sub(r'\s*\(.*?\)\s*', '', n).strip()
            if n:
                names.append(n)
    return names


def _fast_read_author(path: Path) -> str:
    """Read only the GD3 author_en field from a .vgm/.vgz file.  Returns '' on error."""
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == '.vgz':
            raw = gzip.decompress(raw)
        if len(raw) < 0x20 or raw[:4] != b'Vgm ':
            return ''
        gd3_rel = struct.unpack_from('<I', raw, 0x14)[0]
        if gd3_rel == 0:
            return ''
        gd3_abs = gd3_rel + 0x14
        if gd3_abs + 12 > len(raw) or raw[gd3_abs:gd3_abs + 4] != b'Gd3 ':
            return ''
        pos = gd3_abs + 12  # skip magic(4) + version(4) + size(4)
        # Read 7 strings: track_en, track_jp, game_en, game_jp, sys_en, sys_jp, author_en
        def _read_utf16le(data: bytes, p: int) -> tuple[str, int]:
            chars = []
            while p + 1 < len(data):
                cp = struct.unpack_from('<H', data, p)[0]
                p += 2
                if cp == 0:
                    break
                chars.append(chr(cp))
            return ''.join(chars), p
        for _ in range(6):  # skip 6 strings before author_en
            _, pos = _read_utf16le(raw, pos)
        author_en, _ = _read_utf16le(raw, pos)
        return author_en.strip()
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Token helper functions
# ---------------------------------------------------------------------------

def tempo_to_token(bpm: float) -> int:
    diffs = [abs(bpm - b) for b in TEMPO_BINS]
    return TEMPO_BASE + int(min(range(len(diffs)), key=diffs.__getitem__))


def key_to_token(key_index: int, is_minor: bool) -> int:
    """key_index 0–11 (C–B), is_minor flag → token ID."""
    offset = 12 if is_minor else 0
    return KEY_BASE + offset + (key_index % 12)


def token_to_key(token: int) -> tuple[int, bool]:
    """Inverse of key_to_token.  Returns (key_index, is_minor)."""
    v = token - KEY_BASE
    return v % 12, v >= 12


def pitch_to_token(midi: int) -> int | None:
    """MIDI note number → PITCH token, or None if out of range."""
    if PITCH_MIN_MIDI <= midi <= PITCH_MAX_MIDI:
        return PITCH_BASE + (midi - PITCH_MIN_MIDI)
    return None


def token_to_pitch(token: int) -> int:
    """PITCH token → MIDI note number."""
    return PITCH_MIN_MIDI + (token - PITCH_BASE)


def channel_to_token(ch: int) -> int | None:
    """Logical channel index → channel-select token, or None."""
    if CH_FM_0 <= ch <= 5:
        return CH_FM_BASE + ch
    if ch == CH_DAC:
        return CH_DAC_TOK
    if ch in (CH_PSG_0, CH_PSG_1, CH_PSG_2):
        return CH_PSG_BASE + (ch - CH_PSG_0)
    return None


def vel_to_token(vel: int) -> int:
    return VEL_BASE + max(0, min(15, vel))


# ---------------------------------------------------------------------------
# Composer map
# ---------------------------------------------------------------------------

class ComposerMap:
    """Maps composer names (from GD3 tags) to token IDs in range [320, 447].

    Build from a corpus once, save to JSON, then load for each training run.
    The top ``NUM_COMPOSERS`` individual composers (after splitting compound
    strings like "A, B, C") are assigned sequential token IDs starting at
    ``COMPOSER_BASE``.  All others map to ``UNK_COMPOSER``.
    """

    def __init__(self, composers: list[str]) -> None:
        """
        Parameters
        ----------
        composers : list of str
            Ordered list of composer names.  Index 0 → token COMPOSER_BASE,
            index 1 → COMPOSER_BASE+1, etc.  At most NUM_COMPOSERS entries.
        """
        self.composers: list[str] = composers[:NUM_COMPOSERS]
        # Case-insensitive lookup: normalized name → token ID
        self._lookup: dict[str, int] = {
            name.lower(): COMPOSER_BASE + i
            for i, name in enumerate(self.composers)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, raw_author: str) -> int:
        """Return token ID for an author string, or UNK_COMPOSER.

        First tries the full string; then splits on commas/semicolons and
        returns the first individual name that matches.
        """
        if not raw_author:
            return UNK_COMPOSER
        # Try full string (normalized)
        tok = self._lookup.get(raw_author.lower().strip())
        if tok is not None:
            return tok
        # Try individual names from compound string
        for name in _split_composers(raw_author):
            tok = self._lookup.get(name.lower())
            if tok is not None:
                return tok
        return UNK_COMPOSER

    def name(self, token_id: int) -> str:
        """Reverse lookup: token ID → composer name string."""
        idx = token_id - COMPOSER_BASE
        if 0 <= idx < len(self.composers):
            return self.composers[idx]
        return "Unknown"

    def __len__(self) -> int:
        return len(self.composers)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"composers": self.composers}, indent=2))
        log.info("ComposerMap: saved %d composers → %s", len(self.composers), path)

    @classmethod
    def load(cls, path: str | Path) -> "ComposerMap":
        data = json.loads(Path(path).read_text())
        obj = cls(data["composers"])
        log.info("ComposerMap: loaded %d composers from %s", len(obj.composers), path)
        return obj

    # ------------------------------------------------------------------
    # Corpus builder
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        vgm_files: Sequence["VgmFile | Path | str"],
        top_n: int = NUM_COMPOSERS,
    ) -> "ComposerMap":
        """Build a ComposerMap by scanning GD3 author tags in the corpus.

        Uses a fast header-only reader when given file paths to avoid the
        cost of full VGM event parsing.
        """
        counter: Counter[str] = Counter()

        for item in vgm_files:
            if hasattr(item, 'gd3'):
                # Already a VgmFile
                gd3 = getattr(item, 'gd3', None)
                raw = gd3.author_en.strip() if gd3 else ''
            else:
                raw = _fast_read_author(Path(str(item)))

            if raw and raw.lower() not in ('unknown', ''):
                for name in _split_composers(raw):
                    counter[name] += 1

        top = [name for name, _ in counter.most_common(top_n)]
        log.info(
            "ComposerMap.build: %d unique individual composers, keeping top %d",
            len(counter), len(top),
        )
        return cls(top)


# ---------------------------------------------------------------------------
# Patch library
# ---------------------------------------------------------------------------

class PatchLibrary:
    """Frequency-ranked library of YM2612 FM patches extracted from a corpus.

    Maintains up to ``max_patches`` entries (default 128).  Patches not in
    the library are mapped to the nearest entry by L1 distance on a
    normalised parameter vector.
    """

    def __init__(self, patches: list[Ym2612Patch], max_patches: int = NUM_PATCHES) -> None:
        self.patches: list[Ym2612Patch] = patches[:max_patches]
        self._index: dict[tuple, int] = {
            p.to_fingerprint(): i for i, p in enumerate(self.patches)
        }
        # Pre-compute normalised parameter vectors for nearest-neighbour lookup
        self._vectors = np.array([self._to_vector(p) for p in self.patches], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, patch: Ym2612Patch | None) -> int:
        """Return the library index (0–127) for a patch.

        Exact match first; falls back to L1 nearest neighbour.
        Returns 0 if library is empty or patch is None.
        """
        if patch is None or not self.patches:
            return 0
        exact = self._index.get(patch.to_fingerprint())
        if exact is not None:
            return exact
        # Nearest neighbour
        vec = self._to_vector(patch)
        dists = np.abs(self._vectors - vec).sum(axis=1)
        return int(dists.argmin())

    def get(self, idx: int) -> Ym2612Patch | None:
        """Return the patch at library index idx."""
        if 0 <= idx < len(self.patches):
            return self.patches[idx]
        return None

    def __len__(self) -> int:
        return len(self.patches)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [self._patch_to_dict(p) for p in self.patches]
        path.write_text(json.dumps(data, indent=2))
        log.info("PatchLibrary: saved %d patches → %s", len(self.patches), path)

    @classmethod
    def load(cls, path: str | Path) -> "PatchLibrary":
        data = json.loads(Path(path).read_text())
        patches = [cls._dict_to_patch(d) for d in data]
        lib = cls(patches)
        log.info("PatchLibrary: loaded %d patches from %s", len(patches), path)
        return lib

    # ------------------------------------------------------------------
    # Corpus builder
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        vgm_files: Sequence[VgmFile | Path | str],
        max_patches: int = NUM_PATCHES,
    ) -> "PatchLibrary":
        """Build a patch library from a list of VGM files or paths.

        Extracts all FM patches encountered, frequency-ranks them, and keeps
        the top ``max_patches``.
        """
        from .ym2612 import Ym2612State

        counter: Counter[tuple] = Counter()
        fingerprint_to_patch: dict[tuple, Ym2612Patch] = {}

        for i, item in enumerate(vgm_files):
            try:
                vgm = item if isinstance(item, VgmFile) else load_vgm(str(item))
                decoder = Ym2612State()
                list(decoder.process_vgm(vgm))
                for patch in decoder.last_patches.values():
                    fp = patch.to_fingerprint()
                    counter[fp] += 1
                    fingerprint_to_patch[fp] = patch
            except Exception:
                continue
            if (i + 1) % 1000 == 0:
                log.info("  PatchLibrary: scanned %d/%d files (%d unique patches)",
                         i + 1, len(vgm_files), len(counter))

        ranked = [fingerprint_to_patch[fp] for fp, _ in counter.most_common(max_patches)]
        log.info("PatchLibrary.build: %d unique patches, keeping top %d",
                 len(counter), max_patches)
        return cls(ranked, max_patches)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector(p: Ym2612Patch) -> np.ndarray:
        """Full-parameter float vector for nearest-neighbour comparison.

        Algorithm is one-hot encoded (8 dims, weight=4.0) so the nearest
        neighbour can never cross algorithm boundaries — a different algorithm
        changes the entire operator topology and always sounds completely wrong.
        All other parameters are normalised to [0, 1].
        """
        # One-hot algorithm (8 bins, scaled so any algo mismatch dominates)
        algo_oh = [0.0] * 8
        algo_oh[p.algorithm & 7] = 4.0

        vec = algo_oh + [p.feedback / 7.0]
        for i in range(4):
            vec.append(p.tl[i]  / 127.0)
            vec.append(p.ar[i]  / 31.0)
            vec.append(p.dr[i]  / 31.0)
            vec.append(p.sr[i]  / 31.0)
            vec.append(p.rr[i]  / 15.0)
            vec.append(p.sl[i]  / 15.0)
            vec.append(p.mul[i] / 15.0)
            vec.append(p.dt[i]  / 7.0)
            vec.append(p.ks[i]  / 3.0)
        vec.append(p.ams / 3.0)
        vec.append(p.fms / 7.0)
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def _patch_to_dict(p: Ym2612Patch) -> dict:
        return {
            "algorithm": p.algorithm, "feedback": p.feedback,
            "tl": list(p.tl), "ar": list(p.ar), "dr": list(p.dr),
            "sr": list(p.sr), "rr": list(p.rr), "sl": list(p.sl),
            "mul": list(p.mul), "dt": list(p.dt),
            "ams": p.ams, "fms": p.fms,
        }

    @staticmethod
    def _dict_to_patch(d: dict) -> Ym2612Patch:
        return Ym2612Patch(
            algorithm=d["algorithm"], feedback=d["feedback"],
            tl=tuple(d["tl"]), ar=tuple(d["ar"]), dr=tuple(d["dr"]),
            sr=tuple(d["sr"]), rr=tuple(d["rr"]), sl=tuple(d["sl"]),
            mul=tuple(d["mul"]), dt=tuple(d["dt"]),
            ams=d.get("ams", 0), fms=d.get("fms", 0),
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TokenizerV4:
    """Encode VGM files to v4 token sequences and decode back to NoteEvents.

    Parameters
    ----------
    patch_library : PatchLibrary
        Required for encoding (patch fingerprint → token ID) and decoding
        (token ID → FM register values).
    beats_per_bar : int
        Default 4 (4/4 time).  Affects BAR token placement.
    subdivisions : int
        16th-note grid resolution within a bar.  Default 16.
    """

    def __init__(
        self,
        patch_library: PatchLibrary,
        composer_map: ComposerMap | None = None,
        beats_per_bar: int = 4,
        subdivisions: int = 16,
        dac_slot_map: dict[int, int] | None = None,
    ) -> None:
        self.library       = patch_library
        self.composer_map  = composer_map
        self.beats_per_bar = beats_per_bar
        self.subdivisions  = subdivisions  # 16th notes per bar
        # Maps pcm_offset (int) → DAC slot index 0–7.
        # Built by a corpus scan in dataset_v4.py and saved as JSON.
        # If None, all DAC events map to DAC_HIT_BASE (slot 0 / DAC_HIT_UNK).
        self._dac_slot_map: dict[int, int] = dac_slot_map or {}

    # ------------------------------------------------------------------
    # Public: encode
    # ------------------------------------------------------------------

    def encode(
        self,
        vgm: VgmFile,
        *,
        skip_filter: bool = False,
    ) -> list[int] | None:
        """Encode a VGM file to a list of integer tokens.

        Returns None if the file should be filtered out (SFX / jingle).
        """
        note_events, patch_map = decode_vgm(vgm)
        total_samples = vgm.header.total_samples or (
            max((e.sample_on for e in note_events), default=0) + SAMPLE_RATE
        )

        # Corpus quality filter
        if not skip_filter:
            discard, reason = should_discard(note_events, total_samples)
            if discard:
                log.debug("Filtered %s: %s", vgm.source_path, reason)
                return None

        analysis = analyse_vgm(note_events, total_samples)

        tokens: list[int] = [BOS]

        # ---- File header ----
        tokens.append(tempo_to_token(analysis.tempo_bpm))
        tokens.append(key_to_token(analysis.key_index, analysis.is_minor))
        tokens.append(METER_44)   # TODO: extend to detect 3/4, 6/8

        # ---- Composer conditioning token ----
        if self.composer_map is not None:
            gd3 = getattr(vgm, 'gd3', None)
            raw_author = gd3.author_en.strip() if gd3 else ''
            tokens.append(self.composer_map.lookup(raw_author))
        else:
            tokens.append(UNK_COMPOSER)

        # ---- Channel assignments ----
        active_channels = sorted(analysis.channel_roles.keys())
        for ch in active_channels:
            ch_tok = channel_to_token(ch)
            if ch_tok is None:
                continue
            role = analysis.channel_roles[ch]
            role_tok = ROLE_TO_TOKEN.get(role, ROLE_TO_TOKEN[ROLE_UNK])
            patch_id = self.library.lookup(patch_map.get(ch))
            tokens.extend([ch_tok, role_tok, PATCH_BASE + patch_id])

        # ---- Note events grouped by beat position ----
        tokens.extend(self._encode_note_stream(note_events, analysis))

        tokens.append(EOS)
        return tokens

    def _encode_note_stream(
        self,
        note_events: list[NoteEvent],
        analysis,
    ) -> list[int]:
        """Encode all note on/off events ordered by time → tokens."""
        if not note_events:
            return []

        bpm           = analysis.tempo_bpm
        bar_samples   = SAMPLE_RATE * 60.0 / bpm * self.beats_per_bar
        beat_samples  = bar_samples / self.subdivisions

        # Separate on and off events
        events_by_time: list[tuple[int, str, NoteEvent]] = []
        for e in note_events:
            if e.sample_on >= 0:
                events_by_time.append((e.sample_on, "on", e))
            if e.sample_off >= 0:
                events_by_time.append((e.sample_off, "off", e))

        events_by_time.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

        if not events_by_time:
            return []

        tokens: list[int] = []
        current_bar  = -1
        current_beat = -1

        for sample_pos, kind, event in events_by_time:
            bar  = int(sample_pos / bar_samples)
            beat = int((sample_pos % bar_samples) / beat_samples)
            beat = min(beat, self.subdivisions - 1)

            # Emit BAR token when bar changes
            if bar != current_bar:
                tokens.append(BAR)
                current_bar  = bar
                current_beat = -1

            # Emit BEAT token when beat position changes
            if beat != current_beat:
                tokens.append(BEAT_BASE + beat)
                current_beat = beat

            # Emit the note event itself
            ch = event.channel

            if kind == "on":
                if ch == CH_DAC:
                    slot = self._dac_slot_map.get(event.dac_sample_id, 0)
                    tokens.append(DAC_HIT_BASE + slot)
                elif ch == CH_PSG_NOISE:
                    tokens.append(PSG_NOISE_HIT)
                else:
                    ch_tok = channel_to_token(ch)
                    if ch_tok is None:
                        continue
                    pitch_tok = pitch_to_token(event.pitch) if event.pitch >= 0 else None
                    if pitch_tok is None:
                        continue
                    tokens.extend([
                        ch_tok,
                        NOTE_ON,
                        pitch_tok,
                        vel_to_token(event.velocity),
                    ])
            else:  # "off"
                if ch in (CH_DAC, CH_PSG_NOISE):
                    continue  # no explicit off for percussion
                ch_tok = channel_to_token(ch)
                if ch_tok is None:
                    continue
                tokens.extend([ch_tok, NOTE_OFF])

        return tokens

    # ------------------------------------------------------------------
    # Public: decode
    # ------------------------------------------------------------------

    def decode(
        self,
        tokens: list[int],
        tempo_bpm: float | None = None,
    ) -> tuple[list[NoteEvent], dict]:
        """Decode a token sequence back to NoteEvents and metadata.

        Returns (note_events, header_info_dict).
        The caller passes these to vgm_synth.synthesise() to produce a VGM.

        Parameters
        ----------
        tokens : list[int]
            Token sequence (may include BOS / EOS).
        tempo_bpm : float | None
            Override tempo; if None, reads from TEMPO token in header.
        """
        it = iter(tokens)
        header, it = self._decode_header(it)
        if tempo_bpm is not None:
            header["tempo_bpm"] = tempo_bpm

        bpm          = header.get("tempo_bpm", 120.0)
        bar_samples  = int(SAMPLE_RATE * 60.0 / bpm * self.beats_per_bar)
        beat_samples = bar_samples // self.subdivisions

        note_events: list[NoteEvent] = []
        open_notes: dict[int, NoteEvent] = {}   # ch → open NoteEvent

        current_bar  = 0
        current_beat = 0

        def current_sample() -> int:
            return current_bar * bar_samples + current_beat * beat_samples

        for tok in it:
            if tok in (EOS, PAD):
                break

            if tok == BAR:
                current_bar  += 1
                current_beat  = 0

            elif BEAT_BASE <= tok < BEAT_BASE + self.subdivisions:
                current_beat = tok - BEAT_BASE

            elif tok == PHRASE_END:
                pass  # structural hint, no sample-level effect

            elif DAC_HIT_BASE <= tok < DAC_HIT_BASE + NUM_DAC_SLOTS:
                slot = tok - DAC_HIT_BASE
                e = NoteEvent(
                    channel=CH_DAC, pitch=-1, velocity=15,
                    sample_on=current_sample(),
                    sample_off=current_sample() + beat_samples,
                    dac_sample_id=slot,  # slot index preserved for synthesis
                )
                note_events.append(e)

            elif tok == PSG_NOISE_HIT:
                e = NoteEvent(
                    channel=CH_PSG_NOISE, pitch=-1, velocity=12,
                    sample_on=current_sample(),
                    sample_off=current_sample() + beat_samples,
                )
                note_events.append(e)

            # --- Channel-prefixed events ---
            elif CH_FM_BASE <= tok <= CH_FM_BASE + 5:
                ch = tok - CH_FM_BASE
                self._decode_ch_event(it, ch, current_sample(),
                                      open_notes, note_events, header)

            elif tok == CH_DAC_TOK:
                pass  # DAC channel events are DAC_HIT tokens directly

            elif CH_PSG_BASE <= tok < CH_PSG_BASE + 3:
                ch = CH_PSG_0 + (tok - CH_PSG_BASE)
                self._decode_ch_event(it, ch, current_sample(),
                                      open_notes, note_events, header)

        # Close any still-open notes
        final_sample = current_sample()
        for ch, note in open_notes.items():
            note.sample_off = final_sample
            note_events.append(note)

        note_events.sort(key=lambda e: (e.sample_on, e.channel))
        return note_events, header

    def _decode_ch_event(
        self,
        it,
        ch: int,
        sample: int,
        open_notes: dict,
        note_events: list,
        header: dict,
    ) -> None:
        """Consume NOTE_ON / NOTE_OFF tokens for a given channel."""
        try:
            action = next(it)
        except StopIteration:
            return

        if action == NOTE_ON:
            try:
                pitch_tok = next(it)
                vel_tok   = next(it)
            except StopIteration:
                return
            if not (PITCH_BASE <= pitch_tok < PITCH_BASE + NUM_PITCHES):
                return
            if not (VEL_BASE <= vel_tok < VEL_BASE + 16):
                return

            midi  = token_to_pitch(pitch_tok)
            vel   = vel_tok - VEL_BASE

            patch = self.library.get(
                header.get("channel_patches", {}).get(ch, 0)
            )

            # Close existing note on this channel
            if ch in open_notes:
                old = open_notes.pop(ch)
                old.sample_off = sample
                note_events.append(old)

            note = NoteEvent(
                channel=ch, pitch=midi, velocity=vel,
                sample_on=sample, patch=patch,
            )
            open_notes[ch] = note

        elif action == NOTE_OFF:
            if ch in open_notes:
                note = open_notes.pop(ch)
                note.sample_off = sample
                note_events.append(note)

        elif action == NOTE_HOLD:
            pass  # note continues, no action needed

    def _decode_header(self, it) -> tuple[dict, any]:
        """Consume header tokens and return (header_dict, remaining_iterator)."""
        import itertools
        # We need to peek ahead; collect all tokens first
        all_remaining = list(it)
        pos = 0

        header: dict = {
            "tempo_bpm": 120.0,
            "key_index": 0,
            "is_minor": False,
            "meter": (4, 4),
            "composer_token": UNK_COMPOSER,
            "channel_roles": {},
            "channel_patches": {},
        }

        if pos < len(all_remaining) and all_remaining[pos] == BOS:
            pos += 1

        # Tempo
        if pos < len(all_remaining) and TEMPO_BASE <= all_remaining[pos] < TEMPO_BASE + len(TEMPO_BINS):
            header["tempo_bpm"] = float(TEMPO_BINS[all_remaining[pos] - TEMPO_BASE])
            pos += 1

        # Key
        if pos < len(all_remaining) and KEY_BASE <= all_remaining[pos] < KEY_BASE + 24:
            key_idx, is_minor = token_to_key(all_remaining[pos])
            header["key_index"] = key_idx
            header["is_minor"]  = is_minor
            pos += 1

        # Meter
        if pos < len(all_remaining) and all_remaining[pos] in (METER_44, METER_34, METER_68, METER_24):
            meter_map = {METER_44: (4,4), METER_34: (3,4), METER_68: (6,8), METER_24: (2,4)}
            header["meter"] = meter_map[all_remaining[pos]]
            pos += 1

        # Composer conditioning token
        if pos < len(all_remaining) and COMPOSER_BASE <= all_remaining[pos] <= UNK_COMPOSER:
            header["composer_token"] = all_remaining[pos]
            pos += 1

        # Channel assignments (triplets: CH_tok, ROLE_tok, PATCH_tok)
        while pos + 2 < len(all_remaining):
            ch_tok   = all_remaining[pos]
            role_tok = all_remaining[pos + 1]
            patch_tok = all_remaining[pos + 2]

            is_ch = (
                (CH_FM_BASE <= ch_tok <= CH_FM_BASE + 5) or
                ch_tok == CH_DAC_TOK or
                (CH_PSG_BASE <= ch_tok <= CH_PSG_BASE + 2)
            )
            is_role  = ROLE_TOKEN_BASE <= role_tok < ROLE_TOKEN_BASE + 7
            is_patch = PATCH_BASE <= patch_tok < PATCH_BASE + NUM_PATCHES

            if not (is_ch and is_role and is_patch):
                break

            # Map ch_tok back to channel index
            if CH_FM_BASE <= ch_tok <= CH_FM_BASE + 5:
                ch = ch_tok - CH_FM_BASE
            elif ch_tok == CH_DAC_TOK:
                ch = CH_DAC
            else:
                ch = CH_PSG_0 + (ch_tok - CH_PSG_BASE)

            header["channel_roles"][ch]   = TOKEN_TO_ROLE.get(role_tok, ROLE_UNK)
            header["channel_patches"][ch] = patch_tok - PATCH_BASE
            pos += 3

        return header, iter(all_remaining[pos:])

    # ------------------------------------------------------------------
    # Transposition augmentation
    # ------------------------------------------------------------------

    def transpose(self, tokens: list[int], semitones: int) -> list[int]:
        """Shift all PITCH tokens by *semitones* (may be negative).

        Tokens outside the valid pitch range after transposition are replaced
        with the clamped boundary pitch.  The KEY header token is also updated.
        """
        if semitones == 0:
            return list(tokens)

        result = []
        for tok in tokens:
            if PITCH_BASE <= tok < PITCH_BASE + NUM_PITCHES:
                new_tok = tok + semitones
                new_tok = max(PITCH_BASE, min(PITCH_BASE + NUM_PITCHES - 1, new_tok))
                result.append(new_tok)
            elif KEY_BASE <= tok < KEY_BASE + 24:
                # Rotate key index by semitones
                offset   = tok - KEY_BASE
                is_minor = offset >= 12
                key_idx  = offset % 12
                new_key  = (key_idx + semitones) % 12
                result.append(KEY_BASE + (12 if is_minor else 0) + new_key)
            else:
                result.append(tok)
        return result


# ---------------------------------------------------------------------------
# CLI helper: build patch library
# ---------------------------------------------------------------------------

def _build_library_cli() -> None:
    import argparse
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build v4 patch library from VGM corpus")
    parser.add_argument("--vgm-dir",  default="data/vgm",  help="Directory of VGM/VGZ files")
    parser.add_argument("--out",      default="data/patch_library_v4.json")
    parser.add_argument("--max",      type=int, default=NUM_PATCHES, help="Max patches to keep")
    parser.add_argument("--max-files",type=int, default=None)
    args = parser.parse_args()

    vgm_dir = Path(args.vgm_dir)
    paths   = sorted(vgm_dir.glob("*.vg[mz]"))
    if args.max_files:
        paths = paths[:args.max_files]

    log.info("Scanning %d VGM files for patches …", len(paths))
    lib = PatchLibrary.build(paths, max_patches=args.max)
    lib.save(args.out)


if __name__ == "__main__":
    _build_library_cli()
