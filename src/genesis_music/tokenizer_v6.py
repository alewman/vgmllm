"""v6 tokenizer — lossless FM patch encoding + structured metadata header.

Identical to v4 except the per-channel header stores every FM operator
parameter directly as tokens instead of a 128-entry library lookup ID.
This eliminates the timbre quantisation error that caused "harpsichord"
artefacts in v4 roundtrips: the exact patch is preserved in the token
stream so synthesise_vgm() always has perfect register values.

Vocabulary: 794 tokens (660 original v6 + 134 new metadata tokens).

FM parameter token ranges (appended after v4 vocab):

  IDs 621–628  FM_ALG_BASE   algorithm 0–7
  IDs 629–636  FM_P8_BASE    8-value range (feedback, detune)
  IDs 637–640  FM_P4_BASE    4-value range (ams, ks)
  IDs 641–648  FM_FMS_BASE   fms 0–7
  IDs 649–776  FM_TL_BASE    total level 0–127
  IDs 777–808  FM_AR32_BASE  5-bit range (ar, dr, sr 0–31)
  IDs 809–824  FM_P16_BASE   4-bit range (rr, sl, mul 0–15)

New metadata token ranges:

  IDs 825–888  GAME_BASE     game title ID 0–63
  IDs 889      UNK_GAME
  IDs 890–897  CTX_* / LOOP  track context + loop indicator

Note: TEMPO_BINS expanded to 181 entries (60–240 BPM at 1 BPM steps) in v6.2,
shifting all IDs from KEY_BASE onwards by +165 vs v6.0/v6.1.
VOCAB_SIZE = 898.

Per-FM-channel header block (40 tokens):
  ALG, FB, AMS, FMS,
  [TL, AR, DR, SR, RR, SL, MUL, DT, KS] × 4 operators

Token stream order:
  BOS, TEMPO, KEY, METER, COMPOSER, GAME, CTX,
  [CH_tok, ROLE_tok, [40 FM params]] × active channels,
  note stream …, EOS

Usage::

    tok = TokenizerV6(composer_map=cmap, dac_slot_map=dac_slot_map)
    tokens = tok.encode(vgm_file)
    note_events, header = tok.decode(tokens)
    vgm_bytes = synthesise_vgm(note_events, total_samples,
                                header["channel_patches_direct"])
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import struct
from collections import Counter
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
    NoteEvent, Ym2612Patch, TL_SILENCE_THRESHOLD, _carrier_ops, decode_vgm,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary layout  (660 tokens total)
# ---------------------------------------------------------------------------
#
# IDs 0–3    Special
# IDs 4–184  Tempo (181 bins, 60–240 BPM at 1 BPM steps)
# IDs 185–208 Key (24: 12 major + 12 minor)
# IDs 209–212 Meter
# IDs 213    BAR
# IDs 214–229 BEAT_0 … BEAT_15 (16th-note positions within a bar)
# IDs 230    PHRASE_END
# IDs 231–236 CH_FM_0 … CH_FM_5
# IDs 237    CH_DAC_TOK
# IDs 238–240 CH_PSG_0_TOK … CH_PSG_2_TOK
# IDs 241–247 ROLE tokens (7)
# IDs 248–375 PATCH_0 … PATCH_127  (unused in encode; kept for compat)
# IDs 376–378 NOTE_ON / NOTE_OFF / NOTE_HOLD
# IDs 379–466 PITCH_0 … PITCH_87  (MIDI 24–111)
# IDs 467–482 VEL_0 … VEL_15
# IDs 483–490 DAC_HIT_0 … DAC_HIT_7
# IDs 491    PSG_NOISE_HIT
# IDs 492–619 COMPOSER_0 … COMPOSER_127
# IDs 620    UNK_COMPOSER
# IDs 621–628 FM_ALG_BASE   algorithm 0–7
# IDs 629–636 FM_P8_BASE    8-value range (feedback, detune)
# IDs 637–640 FM_P4_BASE    4-value range (ams, ks)
# IDs 641–648 FM_FMS_BASE   fms 0–7
# IDs 649–776 FM_TL_BASE    total level 0–127
# IDs 777–808 FM_AR32_BASE  5-bit range: ar, dr, sr 0–31
# IDs 809–824 FM_P16_BASE   4-bit range: rr, sl, mul 0–15
# IDs 825–888 GAME_BASE     game title 0–63  (top 64 games; ≥34 tracks)
# IDs 889    UNK_GAME
# IDs 890    CTX_LEVEL  (default: stage/level music)
# IDs 891    CTX_BOSS
# IDs 892    CTX_TITLE
# IDs 893    CTX_CREDITS
# IDs 894    CTX_GAMEOVER
# IDs 895    LOOP_PRESENT  (VGM loop_offset != 0)
# IDs 896    LOOP_ABSENT
# IDs 897    CTX_UNKNOWN
# IDs 732    CTX_UNKNOWN   (uninformative / missing track name)
# VOCAB_SIZE  733
# ---------------------------------------------------------------------------

PAD = 0
BOS = 1
EOS = 2
UNK = 3

TEMPO_BASE = 4
# TEMPO_BINS has 181 entries (60–240 at 1 BPM step) → IDs 4–184
KEY_BASE   = 185   # 4 + 181
METER_44   = 209   # KEY_BASE + 24
METER_34   = 210
METER_68   = 211
METER_24   = 212

BAR        = 213
BEAT_BASE  = 214
PHRASE_END = 230

CH_FM_BASE  = 231
CH_DAC_TOK  = 237
CH_PSG_BASE = 238

ROLE_TOKEN_BASE = 241
_ROLE_ORDER = [ROLE_BASS, ROLE_LEAD, ROLE_HARM,
               ROLE_COUNTER, ROLE_DRUMS, ROLE_PERC, ROLE_UNK]
ROLE_TO_TOKEN = {r: ROLE_TOKEN_BASE + i for i, r in enumerate(_ROLE_ORDER)}
TOKEN_TO_ROLE = {v: k for k, v in ROLE_TO_TOKEN.items()}

PATCH_BASE  = 248          # kept for backward compat / not used in encode
NUM_PATCHES = 128

NOTE_ON   = 376
NOTE_OFF  = 377
NOTE_HOLD = 378

PITCH_BASE     = 379
PITCH_MIN_MIDI = 24
PITCH_MAX_MIDI = 111
NUM_PITCHES    = PITCH_MAX_MIDI - PITCH_MIN_MIDI + 1   # 88

VEL_BASE = 467

NUM_DAC_SLOTS = 8
DAC_HIT_BASE  = 483
DAC_HIT_UNK   = DAC_HIT_BASE
DAC_HIT       = DAC_HIT_BASE

PSG_NOISE_HIT = 491

COMPOSER_BASE = 492
NUM_COMPOSERS = 128
UNK_COMPOSER  = COMPOSER_BASE + NUM_COMPOSERS  # 620

# ---------------------------------------------------------------------------
# New in v6: direct FM parameter tokens
# ---------------------------------------------------------------------------

FM_ALG_BASE  = 621   # algorithm 0–7        (8 tokens)
FM_P8_BASE   = 629   # 8-value range        (8 tokens)  feedback, detune
FM_P4_BASE   = 637   # 4-value range        (4 tokens)  ams, ks
FM_FMS_BASE  = 641   # fms 0–7              (8 tokens)
FM_TL_BASE   = 649   # total level 0–127    (128 tokens)
FM_AR32_BASE = 777   # 5-bit range 0–31     (32 tokens)  ar, dr, sr
FM_P16_BASE  = 809   # 4-bit range 0–15     (16 tokens)  rr, sl, mul

FM_PATCH_TOKENS = 40  # tokens per FM channel in header

# ---------------------------------------------------------------------------
# Game map and track-context tokens  (new in v6.1)
# ---------------------------------------------------------------------------

GAME_BASE    = 825   # game title ID 0–63  (top-64 games; ≥34 tracks each)
NUM_GAMES    = 64
UNK_GAME     = GAME_BASE + NUM_GAMES         # 889

CTX_BASE     = 890   # track context tokens (6 tokens)
CTX_LEVEL    = 890   # default: in-level / stage music
CTX_BOSS     = 891   # boss encounter
CTX_TITLE    = 892   # title screen / menu
CTX_CREDITS  = 893   # staff roll / ending / credits
CTX_GAMEOVER = 894   # game over / continue
CTX_UNKNOWN  = 897   # uninformative / missing / all-non-ASCII track name

LOOP_PRESENT = 895   # VGM loop_offset != 0  (track loops)
LOOP_ABSENT  = 896   # no loop

VOCAB_SIZE = 898


# ---------------------------------------------------------------------------
# Shared token helpers (identical to v4)
# ---------------------------------------------------------------------------

def _split_composers(raw: str) -> list[str]:
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
        pos = gd3_abs + 12

        def _read_utf16le(data: bytes, p: int) -> tuple[str, int]:
            chars = []
            while p + 1 < len(data):
                cp = struct.unpack_from('<H', data, p)[0]
                p += 2
                if cp == 0:
                    break
                chars.append(chr(cp))
            return ''.join(chars), p

        for _ in range(6):
            _, pos = _read_utf16le(raw, pos)
        author_en, _ = _read_utf16le(raw, pos)
        return author_en.strip()
    except Exception:
        return ''


_GD3_REGION_RE = re.compile(
    r'\s*\((japan|usa|us|europe|world|rev\s*[a-z0-9]+|[jue][^)]*)\)\s*$',
    re.IGNORECASE,
)


def _fast_read_gd3_fields(path: Path, *field_indices: int) -> list[str]:
    """Read GD3 text fields by 0-based index from a VGM/VGZ without full parse.

    Field order: 0=track_name_en, 1=track_name_jp, 2=game_name_en,
                 3=game_name_jp, 4=system_en, 5=system_jp, 6=author_en, …
    Returns a list (one entry per requested index); missing fields return ''.
    """
    result = [''] * len(field_indices)
    if not field_indices:
        return result
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == '.vgz':
            raw = gzip.decompress(raw)
        if len(raw) < 0x20 or raw[:4] != b'Vgm ':
            return result
        gd3_rel = struct.unpack_from('<I', raw, 0x14)[0]
        if gd3_rel == 0:
            return result
        gd3_abs = gd3_rel + 0x14
        if gd3_abs + 12 > len(raw) or raw[gd3_abs:gd3_abs + 4] != b'Gd3 ':
            return result
        pos = gd3_abs + 12
        idx_to_slot = {idx: slot for slot, idx in enumerate(field_indices)}
        max_field = max(field_indices)
        for field_i in range(max_field + 1):
            chars: list[str] = []
            while pos + 1 < len(raw):
                cp = struct.unpack_from('<H', raw, pos)[0]
                pos += 2
                if cp == 0:
                    break
                chars.append(chr(cp))
            if field_i in idx_to_slot:
                result[idx_to_slot[field_i]] = ''.join(chars).strip()
    except Exception:
        pass
    return result


def _normalize_game_name(raw: str) -> str:
    """Lowercase and strip trailing region suffixes for consistent matching."""
    return _GD3_REGION_RE.sub('', raw.strip()).lower().strip()


def infer_track_context(track_name: str) -> int:
    """Map a GD3 track_name_en → one of the CTX_* tokens via keyword heuristics."""
    stripped = track_name.strip()

    # Trivially uninformative: empty, all-numeric, or very short
    if len(stripped) < 3 or stripped.isdigit():
        return CTX_UNKNOWN

    # Japanese keyword patterns (before the all-non-ASCII gate)
    if re.search(r'\u30dc\u30b9|\u30d5\u30a1\u30a4\u30c8', stripped):      # ボス|ファイト
        return CTX_BOSS
    if re.search(r'\u30bf\u30a4\u30c8\u30eb|\u30aa\u30fc\u30d7\u30cb\u30f3\u30b0', stripped):  # タイトル|オープニング
        return CTX_TITLE
    if re.search(r'\u30a8\u30f3\u30c7\u30a3\u30f3\u30b0|\u30b9\u30bf\u30c3\u30d5', stripped):  # エンディング|スタッフ
        return CTX_CREDITS
    if re.search(r'\u30b2\u30fc\u30e0\u30aa\u30fc\u30d0\u30fc', stripped):  # ゲームオーバー
        return CTX_GAMEOVER

    # All-non-ASCII with no matched keywords: uninformative
    if all(ord(c) > 127 for c in stripped if not c.isspace()):
        return CTX_UNKNOWN

    name = stripped.lower()

    # English keyword patterns
    if re.search(r'\b(boss|battle|fight|combat|versus|vs\.?)\b', name):
        return CTX_BOSS
    if re.search(r'\b(title|opening|intro|name\s*entry|player\s*select|character\s*select)\b', name):
        return CTX_TITLE
    if re.search(r'\b(staff|ending|credit|outro|epilogue|fin(?:ale?)?)\b', name):
        return CTX_CREDITS
    if re.search(r'\b(game\s*over|continue|death)\b', name):
        return CTX_GAMEOVER
    return CTX_LEVEL


def tempo_to_token(bpm: float) -> int:
    diffs = [abs(bpm - b) for b in TEMPO_BINS]
    return TEMPO_BASE + int(min(range(len(diffs)), key=diffs.__getitem__))


def key_to_token(key_index: int, is_minor: bool) -> int:
    offset = 12 if is_minor else 0
    return KEY_BASE + offset + (key_index % 12)


def token_to_key(token: int) -> tuple[int, bool]:
    v = token - KEY_BASE
    return v % 12, v >= 12


def pitch_to_token(midi: int) -> int | None:
    if PITCH_MIN_MIDI <= midi <= PITCH_MAX_MIDI:
        return PITCH_BASE + (midi - PITCH_MIN_MIDI)
    return None


def token_to_pitch(token: int) -> int:
    return PITCH_MIN_MIDI + (token - PITCH_BASE)


def channel_to_token(ch: int) -> int | None:
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
# Direct FM patch encode / decode  (the v6 core improvement)
# ---------------------------------------------------------------------------

def encode_fm_patch(patch: Ym2612Patch) -> list[int]:
    """Encode one Ym2612Patch as exactly FM_PATCH_TOKENS (40) parameter tokens.

    Token order (fixed, no separator needed):
      ALG, FB, AMS, FMS,
      for each of 4 operators: TL, AR, DR, SR, RR, SL, MUL, DT, KS
    """
    toks = [
        FM_ALG_BASE + (patch.algorithm & 7),
        FM_P8_BASE  + (patch.feedback  & 7),
        FM_P4_BASE  + (patch.ams       & 3),
        FM_FMS_BASE + (patch.fms       & 7),
    ]
    for i in range(4):
        toks += [
            FM_TL_BASE   + (patch.tl[i]  & 127),
            FM_AR32_BASE + (patch.ar[i]  & 31),
            FM_AR32_BASE + (patch.dr[i]  & 31),
            FM_AR32_BASE + (patch.sr[i]  & 31),
            FM_P16_BASE  + (patch.rr[i]  & 15),
            FM_P16_BASE  + (patch.sl[i]  & 15),
            FM_P16_BASE  + (patch.mul[i] & 15),
            FM_P8_BASE   + (patch.dt[i]  & 7),
            FM_P4_BASE   + (patch.ks[i]  & 3),
        ]
    assert len(toks) == FM_PATCH_TOKENS
    return toks


def decode_fm_patch(tokens: list[int], pos: int) -> Ym2612Patch | None:
    """Decode FM_PATCH_TOKENS (40) direct parameter tokens → Ym2612Patch.

    Returns None if tokens at *pos* are not a valid patch block.
    """
    if pos + FM_PATCH_TOKENS > len(tokens):
        return None
    t = tokens[pos:]
    if not (FM_ALG_BASE <= t[0] < FM_ALG_BASE + 8):
        return None

    algorithm = t[0] - FM_ALG_BASE
    feedback  = t[1] - FM_P8_BASE
    ams       = t[2] - FM_P4_BASE
    fms       = t[3] - FM_FMS_BASE

    tl  = [0] * 4; ar  = [0] * 4; dr  = [0] * 4
    sr  = [0] * 4; rr  = [0] * 4; sl  = [0] * 4
    mul = [0] * 4; dt  = [0] * 4; ks  = [0] * 4

    for i in range(4):
        base = 4 + i * 9
        tl[i]  = t[base + 0] - FM_TL_BASE
        ar[i]  = t[base + 1] - FM_AR32_BASE
        dr[i]  = t[base + 2] - FM_AR32_BASE
        sr[i]  = t[base + 3] - FM_AR32_BASE
        rr[i]  = t[base + 4] - FM_P16_BASE
        sl[i]  = t[base + 5] - FM_P16_BASE
        mul[i] = t[base + 6] - FM_P16_BASE
        dt[i]  = t[base + 7] - FM_P8_BASE
        ks[i]  = t[base + 8] - FM_P4_BASE

    return Ym2612Patch(
        algorithm=algorithm, feedback=feedback,
        tl=tuple(tl), ar=tuple(ar), dr=tuple(dr),
        sr=tuple(sr), rr=tuple(rr), sl=tuple(sl),
        mul=tuple(mul), dt=tuple(dt), ks=tuple(ks),
        ams=ams, fms=fms,
    )


# ---------------------------------------------------------------------------
# ComposerMap  (identical to v4)
# ---------------------------------------------------------------------------

class ComposerMap:
    def __init__(self, composers: list[str]) -> None:
        self.composers: list[str] = composers[:NUM_COMPOSERS]
        self._lookup: dict[str, int] = {
            name.lower(): COMPOSER_BASE + i
            for i, name in enumerate(self.composers)
        }

    def lookup(self, raw_author: str) -> int:
        if not raw_author:
            return UNK_COMPOSER
        tok = self._lookup.get(raw_author.lower().strip())
        if tok is not None:
            return tok
        for name in _split_composers(raw_author):
            tok = self._lookup.get(name.lower())
            if tok is not None:
                return tok
        return UNK_COMPOSER

    def name(self, token_id: int) -> str:
        idx = token_id - COMPOSER_BASE
        if 0 <= idx < len(self.composers):
            return self.composers[idx]
        return "Unknown"

    def __len__(self) -> int:
        return len(self.composers)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"composers": self.composers}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ComposerMap":
        data = json.loads(Path(path).read_text())
        return cls(data["composers"])

    @classmethod
    def build(cls, vgm_files: Sequence, top_n: int = NUM_COMPOSERS) -> "ComposerMap":
        counter: Counter[str] = Counter()
        for item in vgm_files:
            if hasattr(item, 'gd3'):
                gd3 = getattr(item, 'gd3', None)
                raw = gd3.author_en.strip() if gd3 else ''
            else:
                raw = _fast_read_author(Path(str(item)))
            if raw and raw.lower() not in ('unknown', ''):
                for name in _split_composers(raw):
                    counter[name] += 1
        top = [name for name, _ in counter.most_common(top_n)]
        return cls(top)


# ---------------------------------------------------------------------------
# GameMap
# ---------------------------------------------------------------------------

class GameMap:
    """Maps game title strings → GAME_* token IDs (660–787) or UNK_GAME (788).

    Built once from the corpus GD3 game_name_en tags, keeping the top-N
    games by track count.  Region suffixes are normalised so
    "Sonic The Hedgehog (Japan)" and "Sonic The Hedgehog (US)" count as
    the same entry.  The stored names are the normalised (lowercased,
    region-stripped) forms used for lookup.
    """

    def __init__(self, games: list[str]) -> None:
        self.games: list[str] = games[:NUM_GAMES]
        self._lookup: dict[str, int] = {
            _normalize_game_name(g): GAME_BASE + i
            for i, g in enumerate(self.games)
        }

    def lookup(self, raw_game: str) -> int:
        if not raw_game:
            return UNK_GAME
        tok = self._lookup.get(_normalize_game_name(raw_game))
        return tok if tok is not None else UNK_GAME

    def name(self, token_id: int) -> str:
        idx = token_id - GAME_BASE
        if 0 <= idx < len(self.games):
            return self.games[idx]
        return "Unknown"

    def __len__(self) -> int:
        return len(self.games)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"games": self.games}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GameMap":
        data = json.loads(Path(path).read_text())
        return cls(data["games"])

    @classmethod
    def build(cls, vgm_files: Sequence, top_n: int = NUM_GAMES) -> "GameMap":
        """Build a GameMap by scanning GD3 game_name_en tags in the corpus."""
        counter: Counter[str] = Counter()
        for item in vgm_files:
            if hasattr(item, 'gd3'):
                gd3 = getattr(item, 'gd3', None)
                raw = gd3.game_name_en.strip() if gd3 else ''
            else:
                fields = _fast_read_gd3_fields(Path(str(item)), 2)
                raw = fields[0]
            norm = _normalize_game_name(raw)
            if norm and norm not in ('unknown', ''):
                counter[norm] += 1
        top = [name for name, _ in counter.most_common(top_n)]
        return cls(top)


# ---------------------------------------------------------------------------
# TokenizerV6
# ---------------------------------------------------------------------------

class TokenizerV6:
    """v6 tokenizer: lossless FM patch encoding via direct parameter tokens.

    PatchLibrary is not required.  Every FM channel's full patch parameters
    are stored directly in the token header (40 tokens per channel).

    Parameters
    ----------
    composer_map : ComposerMap | None
        Optional composer conditioning.  Can reuse existing v4 composer map
        (same token IDs 327–455).
    dac_slot_map : dict[int, int] | None
        Maps pcm_offset → DAC slot index 0–7.  Same format as v4.
    beats_per_bar : int
        Default 4.
    subdivisions : int
        Beat-slot grid size.  Always 16 (full BEAT token range, IDs 214–229).
        For 4/4 songs each slot is a 16th note; for 2/4 songs each slot is a
        32nd note (the bar spans the same 16 slots at finer per-slot granularity).
        Default 16.
    """

    def __init__(
        self,
        composer_map: "ComposerMap | None" = None,
        game_map: "GameMap | None" = None,
        beats_per_bar: int = 4,
        subdivisions: int = 16,
        dac_slot_map: dict[int, int] | None = None,
    ) -> None:
        self.composer_map  = composer_map
        self.game_map      = game_map
        self.beats_per_bar = beats_per_bar
        self.subdivisions  = subdivisions
        self._dac_slot_map: dict[int, int] = dac_slot_map or {}

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        vgm: VgmFile,
        *,
        skip_filter: bool = False,
    ) -> list[int] | None:
        """Encode VGM → token list, or None if filtered out."""
        note_events, _last_patches = decode_vgm(vgm)
        total_samples = vgm.header.total_samples or (
            max((e.sample_on for e in note_events), default=0) + SAMPLE_RATE
        )

        # Build patch_map: for each FM channel, pick the most-used patch
        # (by note count) that has an audible carrier TL. This handles songs
        # that change instruments mid-song or temporarily mute a channel with
        # TL=127 — we want the primary audible timbre, not the last-seen state.
        _patch_counter: dict[int, Counter] = {}
        _patch_by_fp: dict[int, dict] = {}
        for n in note_events:
            ch = n.channel
            if not (0 <= ch <= 5) or n.patch is None:
                continue
            fp = n.patch.to_fingerprint()
            _patch_counter.setdefault(ch, Counter())[fp] += 1
            _patch_by_fp.setdefault(ch, {})[fp] = n.patch
        patch_map: dict[int, Ym2612Patch] = dict(_last_patches)
        for ch in range(6):
            if ch not in _patch_counter:
                continue
            carriers = _carrier_ops(_last_patches[ch].algorithm if ch in _last_patches else 0)
            # Prefer most-common patch whose carrier TL is below the silence
            # threshold; fall back to overall most-common if all are muted.
            ordered = _patch_counter[ch].most_common()
            chosen = next(
                (fp for fp, _ in ordered
                 if all(_patch_by_fp[ch][fp].tl[c] < TL_SILENCE_THRESHOLD for c in carriers)),
                ordered[0][0] if ordered else None,
            )
            if chosen is not None:
                patch_map[ch] = _patch_by_fp[ch][chosen]

        if not skip_filter:
            discard, reason = should_discard(note_events, total_samples)
            if discard:
                log.debug("Filtered %s: %s", vgm.source_path, reason)
                return None

        analysis = analyse_vgm(note_events, total_samples)

        tokens: list[int] = [BOS]

        # ---- Global header ----
        tokens.append(tempo_to_token(analysis.tempo_bpm))
        tokens.append(key_to_token(analysis.key_index, analysis.is_minor))
        # Meter: use detected time signature rather than hardcoding 4/4
        _meter_tok_map = {
            (4, 4): METER_44, (3, 4): METER_34,
            (6, 8): METER_68, (2, 4): METER_24,
        }
        tokens.append(_meter_tok_map.get(
            (analysis.meter_numerator, analysis.meter_denominator), METER_44))

        # ---- Composer ----
        gd3 = getattr(vgm, 'gd3', None)
        if self.composer_map is not None:
            raw_author = gd3.author_en.strip() if gd3 else ''
            tokens.append(self.composer_map.lookup(raw_author))
        else:
            tokens.append(UNK_COMPOSER)

        # ---- Game ----
        if self.game_map is not None:
            raw_game = gd3.game_name_en.strip() if gd3 else ''
            tokens.append(self.game_map.lookup(raw_game))
        else:
            tokens.append(UNK_GAME)

        # ---- Track context (rule-based from GD3 track name) ----
        raw_track = gd3.track_name_en.strip() if gd3 else ''
        tokens.append(infer_track_context(raw_track))

        # ---- Loop indicator (from VGM header loop_offset field) ----
        has_loop = getattr(vgm.header, 'loop_offset', 0) != 0
        tokens.append(LOOP_PRESENT if has_loop else LOOP_ABSENT)

        # ---- Channel header: CH_tok, ROLE_tok, [40 patch tokens for FM] ----
        active_channels = sorted(analysis.channel_roles.keys())
        for ch in active_channels:
            ch_tok = channel_to_token(ch)
            if ch_tok is None:
                continue
            role = analysis.channel_roles[ch]
            role_tok = ROLE_TO_TOKEN.get(role, ROLE_TO_TOKEN[ROLE_UNK])
            tokens.append(ch_tok)
            tokens.append(role_tok)
            # FM channels: emit 40 direct parameter tokens
            if 0 <= ch <= 5:
                patch = patch_map.get(ch)
                if patch is not None:
                    tokens.extend(encode_fm_patch(patch))
                else:
                    # Fallback: silence patch (all zeros, algo 0)
                    tokens.extend(encode_fm_patch(Ym2612Patch(
                        algorithm=0, feedback=0,
                        tl=(127,)*4, ar=(0,)*4, dr=(0,)*4, sr=(0,)*4,
                        rr=(0,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
                    )))
            # DAC and PSG channels: no patch tokens

        # ---- Note stream (identical to v4) ----
        tokens.extend(self._encode_note_stream(note_events, analysis))
        tokens.append(EOS)
        return tokens

    def _encode_note_stream(self, note_events: list[NoteEvent], analysis) -> list[int]:
        if not note_events:
            return []

        # Use the quantized BPM that will be stored in the TEMPO token so that
        # bar/beat slot numbers round-trip to the same sample positions in decode().
        # Using analysis.tempo_bpm (actual) here while decode() uses the quantized
        # value caused systematic linear timing drift of ~24 ms/bar.
        bpm       = float(TEMPO_BINS[tempo_to_token(analysis.tempo_bpm) - TEMPO_BASE])
        meter_num = analysis.meter_numerator
        meter_den = analysis.meter_denominator

        # Grid resolution: always 16 slots per bar (the full BEAT token range).
        # beat_samples adapts to the meter so 2/4 gets 32nd-note granularity
        # and 4/4 keeps 16th-note granularity — no vocab changes required.
        #
        # 2/4 example (113 BPM): bar = 8 × sixteenth = 46 839 samp
        #   beat_samples = 46 839 / 16 = 2 927 samp = 66 ms  (32nd note)
        # 4/4 example (113 BPM): bar = 16 × sixteenth = 93 678 samp
        #   beat_samples = 93 678 / 16 = 5 855 samp = 133 ms (16th note)
        sixteenth            = SAMPLE_RATE * 60.0 / bpm / 4.0
        bar_sixteenth_count  = meter_num * (4 if meter_den == 4 else 2)  # 16th notes per bar
        slots_per_bar        = 16                                          # always use all beat slots
        bar_samples          = sixteenth * bar_sixteenth_count             # bar length unchanged
        beat_samples         = bar_samples / slots_per_bar                 # adapts to meter

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
            # Round to the nearest 16th-note slot instead of flooring.  Floor
            # quantization caused a systematic ~65 ms early-bias: notes near the
            # END of a slot were placed at the START, up to 133 ms early.  With
            # round, the max error is ±66 ms and the mean bias is ~0, which also
            # eliminates the audible "stutter" at section boundaries where
            # previously the timing would snap 130 ms forward.
            beat_frac = (sample_pos % bar_samples) / beat_samples
            # Use round-half-up (int(x+0.5)) instead of Python's banker's round()
            # so that notes exactly at a half-slot boundary round UP to the later
            # slot rather than the nearer-even slot.  Without this, a note spaced
            # exactly 2.5 slots from the previous one would round to 2 (banker's)
            # instead of 3, creating a visible clustering in the Synthesia piano-
            # roll every 3-4 notes.
            beat = int(beat_frac + 0.5)
            if beat >= slots_per_bar:
                beat = 0
                bar += 1   # note rounds into the next bar

            if bar != current_bar:
                # Emit one BAR token per bar advanced so that multi-bar silence
                # gaps (inter-section breaks) are faithfully preserved.  The old
                # code always emitted a single BAR token, which caused the decoder
                # to lose N-1 bars for every N-bar gap — up to 8.6 s of drift on
                # a 2-minute song.
                n_bars = (bar - current_bar) if current_bar >= 0 else (bar + 1)
                for _ in range(n_bars):
                    tokens.append(BAR)
                current_bar  = bar
                current_beat = -1

            if beat != current_beat:
                tokens.append(BEAT_BASE + beat)
                current_beat = beat

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
                    tokens.extend([ch_tok, NOTE_ON, pitch_tok, vel_to_token(event.velocity)])
            else:
                if ch in (CH_DAC, CH_PSG_NOISE):
                    continue
                ch_tok = channel_to_token(ch)
                if ch_tok is None:
                    continue
                tokens.extend([ch_tok, NOTE_OFF])

        return tokens

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        tokens: list[int],
        tempo_bpm: float | None = None,
    ) -> tuple[list[NoteEvent], dict]:
        """Decode token list → (NoteEvents, header dict).

        header dict keys:
          tempo_bpm, key_index, is_minor, meter, composer_token,
          channel_roles, channel_patches_direct  ← dict[int, Ym2612Patch]
        """
        all_tokens = list(tokens)
        pos = 0

        header: dict = {
            "tempo_bpm": 120.0,
            "key_index": 0,
            "is_minor": False,
            "meter": (4, 4),
            "composer_token": UNK_COMPOSER,
            "game_token": UNK_GAME,
            "context_token": CTX_LEVEL,
            "loop_present": False,
            "channel_roles": {},
            "channel_patches_direct": {},   # ch → Ym2612Patch (lossless)
        }

        if pos < len(all_tokens) and all_tokens[pos] == BOS:
            pos += 1

        # Tempo
        if pos < len(all_tokens) and TEMPO_BASE <= all_tokens[pos] < TEMPO_BASE + len(TEMPO_BINS):
            header["tempo_bpm"] = float(TEMPO_BINS[all_tokens[pos] - TEMPO_BASE])
            pos += 1

        # Key
        if pos < len(all_tokens) and KEY_BASE <= all_tokens[pos] < KEY_BASE + 24:
            key_idx, is_minor = token_to_key(all_tokens[pos])
            header["key_index"] = key_idx
            header["is_minor"]  = is_minor
            pos += 1

        # Meter
        if pos < len(all_tokens) and all_tokens[pos] in (METER_44, METER_34, METER_68, METER_24):
            meter_map = {METER_44: (4,4), METER_34: (3,4), METER_68: (6,8), METER_24: (2,4)}
            header["meter"] = meter_map[all_tokens[pos]]
            pos += 1

        # Composer
        if pos < len(all_tokens) and COMPOSER_BASE <= all_tokens[pos] <= UNK_COMPOSER:
            header["composer_token"] = all_tokens[pos]
            pos += 1

        # Game
        if pos < len(all_tokens) and GAME_BASE <= all_tokens[pos] <= UNK_GAME:
            header["game_token"] = all_tokens[pos]
            pos += 1

        # Track context
        _ctx_toks = (CTX_LEVEL, CTX_BOSS, CTX_TITLE, CTX_CREDITS, CTX_GAMEOVER, CTX_UNKNOWN)
        if pos < len(all_tokens) and all_tokens[pos] in _ctx_toks:
            header["context_token"] = all_tokens[pos]
            pos += 1

        # Loop indicator
        if pos < len(all_tokens) and all_tokens[pos] in (LOOP_PRESENT, LOOP_ABSENT):
            header["loop_present"] = (all_tokens[pos] == LOOP_PRESENT)
            pos += 1

        # Channel header entries: CH_tok, ROLE_tok, [40 patch tokens if FM]
        while pos < len(all_tokens):
            ch_tok = all_tokens[pos]

            is_fm_ch  = CH_FM_BASE <= ch_tok <= CH_FM_BASE + 5
            is_dac_ch = ch_tok == CH_DAC_TOK
            is_psg_ch = CH_PSG_BASE <= ch_tok <= CH_PSG_BASE + 2

            if not (is_fm_ch or is_dac_ch or is_psg_ch):
                break  # reached note stream

            # Need at least ch_tok + role_tok
            if pos + 1 >= len(all_tokens):
                break
            role_tok = all_tokens[pos + 1]
            if not (ROLE_TOKEN_BASE <= role_tok < ROLE_TOKEN_BASE + 7):
                break

            # Map to channel index
            if is_fm_ch:
                ch = ch_tok - CH_FM_BASE
            elif is_dac_ch:
                ch = CH_DAC
            else:
                ch = CH_PSG_0 + (ch_tok - CH_PSG_BASE)

            header["channel_roles"][ch] = TOKEN_TO_ROLE.get(role_tok, ROLE_UNK)
            pos += 2  # consumed ch_tok + role_tok

            # FM channels: consume 40 direct patch tokens
            if is_fm_ch:
                patch = decode_fm_patch(all_tokens, pos)
                if patch is not None:
                    header["channel_patches_direct"][ch] = patch
                    pos += FM_PATCH_TOKENS
                # If invalid (e.g. truncated model output), skip gracefully

        if tempo_bpm is not None:
            header["tempo_bpm"] = tempo_bpm

        bpm = header["tempo_bpm"]
        meter_num, meter_den = header["meter"]
        sixteenth            = SAMPLE_RATE * 60.0 / bpm / 4.0
        bar_sixteenth_count  = meter_num * (4 if meter_den == 4 else 2)  # 16th notes per bar
        slots_per_bar        = 16                                          # always use all beat slots
        bar_samples          = max(1, int(sixteenth * bar_sixteenth_count))  # bar length unchanged
        beat_samples         = max(1, int(bar_samples / slots_per_bar))      # adapts to meter

        note_events: list[NoteEvent] = []
        open_notes: dict[int, NoteEvent] = {}

        current_bar  = -1
        current_beat = 0

        def current_sample() -> int:
            return current_bar * bar_samples + current_beat * beat_samples

        it = iter(all_tokens[pos:])
        for tok in it:
            if tok in (EOS, PAD):
                break

            if tok == BAR:
                current_bar  += 1
                current_beat  = 0

            elif BEAT_BASE <= tok < BEAT_BASE + self.subdivisions:
                current_beat = tok - BEAT_BASE

            elif tok == PHRASE_END:
                pass

            elif DAC_HIT_BASE <= tok < DAC_HIT_BASE + NUM_DAC_SLOTS:
                slot = tok - DAC_HIT_BASE
                note_events.append(NoteEvent(
                    channel=CH_DAC, pitch=-1, velocity=15,
                    sample_on=current_sample(),
                    sample_off=current_sample() + beat_samples,
                    dac_sample_id=slot,
                ))

            elif tok == PSG_NOISE_HIT:
                note_events.append(NoteEvent(
                    channel=CH_PSG_NOISE, pitch=-1, velocity=12,
                    sample_on=current_sample(),
                    sample_off=current_sample() + beat_samples,
                ))

            elif CH_FM_BASE <= tok <= CH_FM_BASE + 5:
                ch = tok - CH_FM_BASE
                self._decode_ch_event(it, ch, current_sample(),
                                      open_notes, note_events, header)

            elif tok == CH_DAC_TOK:
                pass

            elif CH_PSG_BASE <= tok < CH_PSG_BASE + 3:
                ch = CH_PSG_0 + (tok - CH_PSG_BASE)
                self._decode_ch_event(it, ch, current_sample(),
                                      open_notes, note_events, header)

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

            # Use the lossless direct patch from header
            patch = header["channel_patches_direct"].get(ch)

            if ch in open_notes:
                old = open_notes[ch]
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

    # ------------------------------------------------------------------
    # Transposition augmentation (identical to v4)
    # ------------------------------------------------------------------

    def transpose(self, tokens: list[int], semitones: int) -> list[int]:
        if semitones == 0:
            return list(tokens)
        result = []
        for tok in tokens:
            if PITCH_BASE <= tok < PITCH_BASE + NUM_PITCHES:
                new_tok = tok + semitones
                new_tok = max(PITCH_BASE, min(PITCH_BASE + NUM_PITCHES - 1, new_tok))
                result.append(new_tok)
            elif KEY_BASE <= tok < KEY_BASE + 24:
                offset   = tok - KEY_BASE
                is_minor = offset >= 12
                key_idx  = offset % 12
                new_key  = (key_idx + semitones) % 12
                result.append(KEY_BASE + (12 if is_minor else 0) + new_key)
            else:
                result.append(tok)
        return result
