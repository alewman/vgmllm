"""v7 tokenizer — SSG-EG, hardware state tokens, envelope note-splitting, FM6 fix.

Extends v6 with:
  - SSG-EG operator parameters in FM patch header (44 tokens/channel, up from 40)
  - Hardware state tokens (PAN, LFO, CH3 mode, DAC enable, loop point) inline
  - INSTRUMENT_CHANGE: mid-song patch re-encoding
  - PSG vol_envelope note-splitting at 8-level attenuation bucket boundaries
  - FM tl_envelope note-splitting at carrier-TL bucket boundaries
  - FM6 reclassification fix (CH_DAC notes with FM patch → CH_FM_5)
  - DOWNBEAT / HALFBEAT structural tokens
  - SEP token for cross-song sequence packing

Vocabulary extends v6 (VOCAB_SIZE=898) to VOCAB_SIZE=1024 (sparse upper range).

New token ranges (898–959):
  898–901  PAN_OFF / PAN_LEFT / PAN_RIGHT / PAN_CENTER
  902      LFO_OFF
  903–910  LFO_ON_RATE_0 … LFO_ON_RATE_7
  911–912  CH3_NORMAL_MODE / CH3_SPECIAL_MODE
  913–914  DAC_DISABLE / DAC_ENABLE
  915      LOOP_POINT
  916      INSTRUMENT_CHANGE
  917      DOWNBEAT
  918      HALFBEAT
  919      SEP
  920–935  SSG_EG_BASE … +15   (used in 44-token FM patch)
  936–943  TL_BUCKET_BASE … +7  (reserved for future use)
  944–959  PITCH_WP_BASE … +15  (reserved for future use)

VOCAB_SIZE_V7 = 1024  (960–1023 reserved)
FM_PATCH_TOKENS_V7 = 44  (40 v6 params + 4 SSG-EG, one per operator)
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .music_analysis import (
    ROLE_BASS, ROLE_COUNTER, ROLE_DRUMS, ROLE_HARM,
    ROLE_LEAD, ROLE_PERC, ROLE_UNK,
    SAMPLE_RATE, TEMPO_BINS,
    analyse_vgm, should_discard,
)
from .vgm_parser import EventType, VgmFile, load_vgm
from .ym2612 import (
    CH_DAC, CH_FM_0, CH_FM_5, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    NoteEvent, Ym2612Patch, TL_SILENCE_THRESHOLD, _carrier_ops, decode_vgm,
)

# Re-export all v6 token constants so callers can import from one place
from .tokenizer_v6 import (
    PAD, BOS, EOS, UNK,
    TEMPO_BASE, KEY_BASE,
    METER_44, METER_34, METER_68, METER_24,
    BAR, BEAT_BASE, PHRASE_END,
    CH_FM_BASE, CH_DAC_TOK, CH_PSG_BASE,
    ROLE_TOKEN_BASE, ROLE_TO_TOKEN, TOKEN_TO_ROLE,
    PATCH_BASE, NUM_PATCHES,
    NOTE_ON, NOTE_OFF, NOTE_HOLD,
    PITCH_BASE, PITCH_MIN_MIDI, PITCH_MAX_MIDI, NUM_PITCHES,
    VEL_BASE,
    NUM_DAC_SLOTS, DAC_HIT_BASE, DAC_HIT_UNK, DAC_HIT,
    PSG_NOISE_HIT,
    COMPOSER_BASE, NUM_COMPOSERS, UNK_COMPOSER,
    FM_ALG_BASE, FM_P8_BASE, FM_P4_BASE, FM_FMS_BASE,
    FM_TL_BASE, FM_AR32_BASE, FM_P16_BASE,
    FM_PATCH_TOKENS,
    GAME_BASE, NUM_GAMES, UNK_GAME,
    CTX_BASE, CTX_LEVEL, CTX_BOSS, CTX_TITLE,
    CTX_CREDITS, CTX_GAMEOVER, CTX_UNKNOWN,
    LOOP_PRESENT, LOOP_ABSENT,
    ComposerMap, GameMap,
    infer_track_context, tempo_to_token, key_to_token, token_to_key,
    pitch_to_token, token_to_pitch, channel_to_token, vel_to_token,
    encode_fm_patch, decode_fm_patch,
    _split_composers, _fast_read_author, _fast_read_gd3_fields,
    _normalize_game_name,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v7 token constants  (IDs 898–959; 960–1023 reserved)
# ---------------------------------------------------------------------------

# Pan state tokens (per YM2612 channel; bits 7:6 of reg 0xB4-0xB6)
PAN_OFF    = 898   # 00 → no output
PAN_RIGHT  = 899   # 01 → right only
PAN_LEFT   = 900   # 10 → left only
PAN_CENTER = 901   # 11 → both (default)

_PAN_BITS_TO_TOK = {0: PAN_OFF, 1: PAN_RIGHT, 2: PAN_LEFT, 3: PAN_CENTER}
_PAN_TOK_TO_BITS = {v: k for k, v in _PAN_BITS_TO_TOK.items()}
_ALL_PAN_TOKS = frozenset(_PAN_BITS_TO_TOK.values())

# LFO control (reg 0x22, port 0)
LFO_OFF      = 902
LFO_ON_BASE  = 903   # + rate (0–7) → 903–910

# CH3 special mode (reg 0x27 bit 6, port 0)
CH3_NORMAL_MODE  = 911
CH3_SPECIAL_MODE = 912

# DAC enable (reg 0x2B bit 7, port 0)
DAC_DISABLE = 913
DAC_ENABLE  = 914

# Structural / inline markers
LOOP_POINT        = 915   # inline loop-start marker
INSTRUMENT_CHANGE = 916   # followed by 44-token FM patch block
DOWNBEAT          = 917   # after BAR token (model beat-emphasis hint)
HALFBEAT          = 918   # at beat slot 8 (mid-bar emphasis)
SEP               = 919   # cross-song packing separator

# SSG-EG per-operator (4 bits, 0–15; 0=off, 8–15=looping envelopes)
SSG_EG_BASE = 920   # + val (0–15) → 920–935

# Reserved for future envelope tokens
TL_BUCKET_BASE  = 936   # 8 buckets (936–943)
PITCH_WP_BASE   = 944   # 16 waypoints (944–959)

VOCAB_SIZE_V7     = 1024
FM_PATCH_TOKENS_V7 = 44    # 40 original + 4 SSG-EG (one per op)

# Rare hardware-state token IDs for loss weighting
RARE_TOKEN_IDS: frozenset[int] = frozenset(range(898, 920))

# ---------------------------------------------------------------------------
# v7 FM patch encode / decode  (44 tokens: 40 v6 + 4 SSG-EG)
# ---------------------------------------------------------------------------

def encode_fm_patch_v7(patch: Ym2612Patch) -> list[int]:
    """Encode Ym2612Patch → exactly FM_PATCH_TOKENS_V7 (44) tokens.

    Token order:
      ALG, FB, AMS, FMS,
      for each of 4 operators: TL, AR, DR, SR, RR, SL, MUL, DT, KS, SSG_EG
    """
    toks = [
        FM_ALG_BASE + (patch.algorithm & 7),
        FM_P8_BASE  + (patch.feedback  & 7),
        FM_P4_BASE  + (patch.ams       & 3),
        FM_FMS_BASE + (patch.fms       & 7),
    ]
    ssg_eg = patch.ssg_eg if patch.ssg_eg else (0, 0, 0, 0)
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
            SSG_EG_BASE  + (ssg_eg[i]    & 15),
        ]
    assert len(toks) == FM_PATCH_TOKENS_V7
    return toks


def decode_fm_patch_v7(tokens: list[int], pos: int) -> Ym2612Patch | None:
    """Decode FM_PATCH_TOKENS_V7 (44) tokens → Ym2612Patch.

    Returns None if tokens at *pos* are not a valid v7 patch block.
    """
    if pos + FM_PATCH_TOKENS_V7 > len(tokens):
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
    ssg = [0] * 4

    for i in range(4):
        base = 4 + i * 10   # 10 tokens per op in v7 (was 9 in v6)
        tl[i]  = t[base + 0] - FM_TL_BASE
        ar[i]  = t[base + 1] - FM_AR32_BASE
        dr[i]  = t[base + 2] - FM_AR32_BASE
        sr[i]  = t[base + 3] - FM_AR32_BASE
        rr[i]  = t[base + 4] - FM_P16_BASE
        sl[i]  = t[base + 5] - FM_P16_BASE
        mul[i] = t[base + 6] - FM_P16_BASE
        dt[i]  = t[base + 7] - FM_P8_BASE
        ks[i]  = t[base + 8] - FM_P4_BASE
        ssg[i] = t[base + 9] - SSG_EG_BASE

    return Ym2612Patch(
        algorithm=algorithm, feedback=feedback,
        tl=tuple(tl), ar=tuple(ar), dr=tuple(dr),
        sr=tuple(sr), rr=tuple(rr), sl=tuple(sl),
        mul=tuple(mul), dt=tuple(dt), ks=tuple(ks),
        ams=ams, fms=fms, ssg_eg=tuple(ssg),
    )


# ---------------------------------------------------------------------------
# Hardware event collection
# ---------------------------------------------------------------------------

@dataclass
class _HwEvent:
    """Zero-time hardware state change to be inserted in the note stream."""
    sample_pos: int
    ch: int         # -1 = global, 0-5 = per FM channel
    tokens: list[int] = field(default_factory=list)


def _collect_hw_events(vgm: VgmFile) -> list[_HwEvent]:
    """Scan VGM event list for hardware state changes → sorted list of _HwEvent."""
    events: list[_HwEvent] = []

    # Track prior state (-1 = unknown → emit on first encounter)
    pan_state: list[int] = [-1] * 6   # per FM channel
    lfo_state:  int = -1
    ch3_state:  int = -1
    dac_state:  int = -1

    # Loop point (from header)
    if getattr(vgm.header, 'loop_samples', 0) > 0:
        total = getattr(vgm.header, 'total_samples', 0)
        loop_samps = vgm.header.loop_samples
        if total > loop_samps:
            events.append(_HwEvent(
                sample_pos=total - loop_samps,
                ch=-1, tokens=[LOOP_POINT],
            ))

    for e in vgm.events:
        if e.type not in (EventType.YM2612_PORT0, EventType.YM2612_PORT1):
            continue
        reg = e.register
        val = e.value
        sp  = e.sample_pos

        # Pan  (regs 0xB4–0xB6, both ports)
        if 0xB4 <= reg <= 0xB6:
            ch_off = reg - 0xB4
            ch_idx = ch_off + (3 if e.type == EventType.YM2612_PORT1 else 0)
            if 0 <= ch_idx <= 5:
                pan_bits = (val >> 6) & 0x03
                tok = _PAN_BITS_TO_TOK.get(pan_bits, PAN_CENTER)
                if pan_state[ch_idx] != tok:
                    pan_state[ch_idx] = tok
                    events.append(_HwEvent(
                        sample_pos=sp, ch=ch_idx,
                        tokens=[CH_FM_BASE + ch_idx, tok],
                    ))

        # LFO  (reg 0x22, port 0 only)
        elif e.type == EventType.YM2612_PORT0 and reg == 0x22:
            if val & 0x08:
                tok = LFO_ON_BASE + (val & 0x07)
            else:
                tok = LFO_OFF
            if lfo_state != tok:
                lfo_state = tok
                events.append(_HwEvent(sample_pos=sp, ch=-1, tokens=[tok]))

        # CH3 mode  (reg 0x27, port 0 only)
        elif e.type == EventType.YM2612_PORT0 and reg == 0x27:
            tok = CH3_SPECIAL_MODE if (val & 0x40) else CH3_NORMAL_MODE
            if ch3_state != tok:
                ch3_state = tok
                events.append(_HwEvent(sample_pos=sp, ch=-1, tokens=[tok]))

        # DAC enable  (reg 0x2B, port 0 only)
        elif e.type == EventType.YM2612_PORT0 and reg == 0x2B:
            tok = DAC_ENABLE if (val & 0x80) else DAC_DISABLE
            if dac_state != tok:
                dac_state = tok
                events.append(_HwEvent(sample_pos=sp, ch=-1, tokens=[tok]))

    events.sort(key=lambda ev: ev.sample_pos)
    return events


def _collect_patch_changes(
    note_events: list[NoteEvent],
    header_patch_map: dict[int, Ym2612Patch],
) -> list[_HwEvent]:
    """Detect mid-song FM patch changes and encode as INSTRUMENT_CHANGE events."""
    result: list[_HwEvent] = []
    current: dict[int, Ym2612Patch] = dict(header_patch_map)

    by_ch: dict[int, list[NoteEvent]] = {}
    for n in note_events:
        if 0 <= n.channel <= 5 and n.patch is not None:
            by_ch.setdefault(n.channel, []).append(n)

    for ch, ch_notes in by_ch.items():
        ch_notes.sort(key=lambda n: n.sample_on)
        for note in ch_notes:
            cur = current.get(ch)
            if cur is not None and note.patch.to_fingerprint() != cur.to_fingerprint():
                toks = [CH_FM_BASE + ch, INSTRUMENT_CHANGE] + encode_fm_patch_v7(note.patch)
                result.append(_HwEvent(sample_pos=note.sample_on, ch=ch, tokens=toks))
            current[ch] = note.patch

    return result


# ---------------------------------------------------------------------------
# Note splitting helpers
# ---------------------------------------------------------------------------

def _psg_att_to_vel(raw_att: int) -> int:
    """SN76489 raw attenuation (0=loudest, 15=silent) → 0-15 velocity scale."""
    if raw_att >= 15:
        return 0
    return max(1, 15 - round(raw_att * 14 / 14))


def _vel_8bucket(vel: int) -> int:
    """Map 0-15 velocity to 8-level coarse bucket (0=silent, 7=loudest)."""
    return vel * 7 // 15


def _split_psg_note_by_vol_env(note: NoteEvent) -> list[NoteEvent]:
    """Split PSG note at vol_envelope 8-level bucket boundaries (max 8 segments)."""
    if not note.vol_envelope or note.sample_off < 0:
        return [note]

    # Only consider envelope points within the note window
    vol_changes = sorted(
        (s, a) for s, a in note.vol_envelope
        if note.sample_on <= s < note.sample_off
    )
    if not vol_changes:
        return [note]

    # Build segments by walking bucket changes
    segments: list[tuple[int, int]] = []   # (start_sample, velocity)
    curr_bucket = _vel_8bucket(note.velocity)
    curr_start  = note.sample_on
    curr_vel    = note.velocity

    for sample, raw_att in vol_changes:
        vel    = _psg_att_to_vel(raw_att)
        bucket = _vel_8bucket(vel)
        if bucket != curr_bucket:
            segments.append((curr_start, curr_vel))
            curr_start  = sample
            curr_vel    = vel
            curr_bucket = bucket

    segments.append((curr_start, curr_vel))

    if len(segments) == 1:
        return [note]

    # Cap at 8 segments
    if len(segments) > 8:
        segments = segments[:8]

    result: list[NoteEvent] = []
    for i, (start, vel) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else note.sample_off
        result.append(NoteEvent(
            channel=note.channel,
            pitch=note.pitch,
            velocity=max(0, min(15, vel)),
            sample_on=start,
            sample_off=end,
            noise_mode=note.noise_mode,
        ))
    return result


def _tl_to_vel(avg_tl: float) -> int:
    """Average carrier TL (0=loudest, 127=silent) → 0-15 velocity."""
    bucket = min(7, int(avg_tl * 7 / 127.0))
    return max(1, (7 - bucket) * 15 // 7)


def _tl_8bucket(avg_tl: float) -> int:
    return min(7, int(avg_tl * 7 / 127.0))


def _split_fm_note_by_tl_env(note: NoteEvent) -> list[NoteEvent]:
    """Split FM note at carrier-TL 8-level bucket boundaries (max 8 segments)."""
    if not note.tl_envelope or note.sample_off < 0 or note.patch is None:
        return [note]

    carriers = _carrier_ops(note.patch.algorithm)

    def _avg_tl(tl_vals: list) -> float:
        if len(tl_vals) < 4:
            return 127.0
        return sum(tl_vals[i] for i in carriers) / len(carriers)

    tl_changes = sorted(
        (s, _avg_tl(tls)) for s, tls in note.tl_envelope
        if note.sample_on <= s < note.sample_off
    )
    if not tl_changes:
        return [note]

    init_avg  = _avg_tl(list(note.patch.tl))
    segments: list[tuple[int, int]] = []
    curr_bucket = _tl_8bucket(init_avg)
    curr_start  = note.sample_on
    curr_vel    = note.velocity

    for sample, avg_tl in tl_changes:
        bucket = _tl_8bucket(avg_tl)
        if bucket != curr_bucket:
            segments.append((curr_start, curr_vel))
            curr_start  = sample
            curr_vel    = _tl_to_vel(avg_tl)
            curr_bucket = bucket

    segments.append((curr_start, curr_vel))

    if len(segments) == 1:
        return [note]

    if len(segments) > 8:
        segments = segments[:8]

    result: list[NoteEvent] = []
    for i, (start, vel) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else note.sample_off
        result.append(NoteEvent(
            channel=note.channel,
            pitch=note.pitch,
            velocity=max(0, min(15, vel)),
            sample_on=start,
            sample_off=end,
            patch=note.patch,
            dac_sample_id=note.dac_sample_id,
        ))
    return result


# ---------------------------------------------------------------------------
# FM6 DAC detection fix
# ---------------------------------------------------------------------------

def _reclassify_fm6_notes(note_events: list[NoteEvent]) -> list[NoteEvent]:
    """Reclassify CH_DAC notes that carry an FM patch (not a DAC sample)
    as CH_FM_5 (FM channel 6 used in synth mode).
    """
    result: list[NoteEvent] = []
    for n in note_events:
        if n.channel == CH_DAC and n.dac_sample_id == -1 and n.patch is not None:
            result.append(NoteEvent(
                channel=CH_FM_5,
                pitch=n.pitch,
                velocity=n.velocity,
                sample_on=n.sample_on,
                sample_off=n.sample_off,
                patch=n.patch,
                tl_envelope=list(n.tl_envelope),
                pitch_envelope=list(n.pitch_envelope),
                pan_envelope=list(n.pan_envelope),
            ))
        else:
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# Curated GameMap builder
# ---------------------------------------------------------------------------

def build_curated_game_map(json_path: "str | Path") -> "GameMap":
    """Build a GameMap from a curated games JSON file (e.g. curated_games_v7.json).

    The JSON must contain a ``driver_groups`` list, each group having a
    ``games`` list whose entries contain a ``gd3_name_hint`` string.
    Games are assigned token IDs in the order they appear across all groups.

    Unlike ``GameMap.build()`` (which picks the top-N corpus titles by track
    count), this preserves intentional curation: rare drivers and specific
    hardware archetypes are guaranteed a dedicated token regardless of their
    corpus frequency.

    Usage::

        game_map = build_curated_game_map("data/curated_games_v7.json")
        game_map.save("data/game_map_v7.json")
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    hints: list[str] = []
    seen_normalized: set[str] = set()
    for group in data["driver_groups"]:
        for g in group["games"]:
            hint = g["gd3_name_hint"]
            norm = _normalize_game_name(hint)
            if norm not in seen_normalized:
                hints.append(hint)
                seen_normalized.add(norm)
    return GameMap(hints)


# ---------------------------------------------------------------------------
# TokenizerV7
# ---------------------------------------------------------------------------

class TokenizerV7:
    """v7 tokenizer: SSG-EG, hardware state tokens, envelope note-splitting.

    Parameters
    ----------
    composer_map  : ComposerMap | None
    game_map      : GameMap | None
    dac_slot_map  : dict[int, int] | None  — maps pcm_offset → slot 0-7
    beats_per_bar : int  (default 4)
    subdivisions  : int  (default 16)
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

        # FM6 DAC fix: reclassify notes that are FM but decoded as DAC
        note_events = _reclassify_fm6_notes(note_events)

        # Apply envelope note-splitting
        expanded: list[NoteEvent] = []
        for n in note_events:
            if n.channel == CH_PSG_NOISE or n.channel == CH_DAC:
                expanded.append(n)
            elif CH_PSG_0 <= n.channel <= CH_PSG_2:
                expanded.extend(_split_psg_note_by_vol_env(n))
            elif 0 <= n.channel <= 5:
                expanded.extend(_split_fm_note_by_tl_env(n))
            else:
                expanded.append(n)
        note_events = expanded

        # Build patch_map: most-common audible patch per FM channel
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
            carriers = _carrier_ops(
                _last_patches[ch].algorithm if ch in _last_patches else 0
            )
            ordered = _patch_counter[ch].most_common()
            chosen = next(
                (fp for fp, _ in ordered
                 if all(_patch_by_fp[ch][fp].tl[c] < TL_SILENCE_THRESHOLD
                        for c in carriers)),
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

        # ---- Global header (identical to v6) ----
        tokens.append(tempo_to_token(analysis.tempo_bpm))
        tokens.append(key_to_token(analysis.key_index, analysis.is_minor))
        _meter_tok_map = {
            (4, 4): METER_44, (3, 4): METER_34,
            (6, 8): METER_68, (2, 4): METER_24,
        }
        tokens.append(_meter_tok_map.get(
            (analysis.meter_numerator, analysis.meter_denominator), METER_44))

        gd3 = getattr(vgm, 'gd3', None)
        if self.composer_map is not None:
            raw_author = gd3.author_en.strip() if gd3 else ''
            tokens.append(self.composer_map.lookup(raw_author))
        else:
            tokens.append(UNK_COMPOSER)

        if self.game_map is not None:
            raw_game = gd3.game_name_en.strip() if gd3 else ''
            tokens.append(self.game_map.lookup(raw_game))
        else:
            tokens.append(UNK_GAME)

        raw_track = gd3.track_name_en.strip() if gd3 else ''
        tokens.append(infer_track_context(raw_track))

        has_loop = getattr(vgm.header, 'loop_offset', 0) != 0
        tokens.append(LOOP_PRESENT if has_loop else LOOP_ABSENT)

        # ---- Channel header: CH_tok, ROLE_tok, [44 patch tokens for FM] ----
        active_channels = sorted(analysis.channel_roles.keys())
        for ch in active_channels:
            ch_tok = channel_to_token(ch)
            if ch_tok is None:
                continue
            role = analysis.channel_roles[ch]
            role_tok = ROLE_TO_TOKEN.get(role, ROLE_TO_TOKEN[ROLE_UNK])
            tokens.append(ch_tok)
            tokens.append(role_tok)
            if 0 <= ch <= 5:
                patch = patch_map.get(ch)
                if patch is None:
                    patch = Ym2612Patch(
                        algorithm=0, feedback=0,
                        tl=(127,)*4, ar=(0,)*4, dr=(0,)*4, sr=(0,)*4,
                        rr=(0,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
                    )
                tokens.extend(encode_fm_patch_v7(patch))

        # ---- Hardware state events ----
        hw_events = _collect_hw_events(vgm)
        patch_change_events = _collect_patch_changes(note_events, patch_map)
        all_hw = sorted(hw_events + patch_change_events, key=lambda e: e.sample_pos)

        # ---- Note stream with interleaved HW events ----
        tokens.extend(
            self._encode_note_stream_v7(note_events, analysis, all_hw)
        )
        tokens.append(EOS)
        return tokens

    def _encode_note_stream_v7(
        self,
        note_events: list[NoteEvent],
        analysis,
        hw_events: list[_HwEvent],
    ) -> list[int]:
        if not note_events and not hw_events:
            return []

        bpm       = float(TEMPO_BINS[tempo_to_token(analysis.tempo_bpm) - TEMPO_BASE])
        meter_num = analysis.meter_numerator
        meter_den = analysis.meter_denominator

        sixteenth           = SAMPLE_RATE * 60.0 / bpm / 4.0
        bar_sixteenth_count = meter_num * (4 if meter_den == 4 else 2)
        slots_per_bar       = 16
        bar_samples         = sixteenth * bar_sixteenth_count
        beat_samples        = bar_samples / slots_per_bar

        # Build unified event list: (sample_pos, priority, payload)
        # priority: 0=hw, 1=note_off, 2=note_on
        all_events: list[tuple[int, int, object]] = []
        for e in note_events:
            if e.sample_on >= 0:
                all_events.append((e.sample_on, 2, ("on", e)))
            if e.sample_off >= 0:
                all_events.append((e.sample_off, 1, ("off", e)))
        for hw in hw_events:
            all_events.append((hw.sample_pos, 0, ("hw", hw)))

        all_events.sort(key=lambda x: (x[0], x[1]))

        if not all_events:
            return []

        tokens: list[int] = []
        current_bar  = -1
        current_beat = -1

        def _grid_position(sample_pos: int) -> tuple[int, int]:
            bar       = int(sample_pos / bar_samples)
            beat_frac = (sample_pos % bar_samples) / beat_samples
            beat      = int(beat_frac + 0.5)
            if beat >= slots_per_bar:
                beat = 0
                bar += 1
            return bar, beat

        def _advance_to(bar: int, beat: int) -> None:
            nonlocal current_bar, current_beat
            if bar != current_bar:
                n_bars = (bar - current_bar) if current_bar >= 0 else (bar + 1)
                for _ in range(n_bars):
                    tokens.append(BAR)
                    tokens.append(DOWNBEAT)
                current_bar  = bar
                current_beat = -1
            if beat != current_beat:
                if beat == 8 and (current_beat < 0 or current_beat < 8):
                    tokens.append(HALFBEAT)
                tokens.append(BEAT_BASE + beat)
                current_beat = beat

        for _sp, _pri, payload in all_events:
            kind = payload[0]
            bar, beat = _grid_position(_sp)

            if kind == "hw":
                hw: _HwEvent = payload[1]
                _advance_to(bar, beat)
                tokens.extend(hw.tokens)

            elif kind == "on":
                event: NoteEvent = payload[1]
                _advance_to(bar, beat)
                ch = event.channel
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
                    tokens.extend([ch_tok, NOTE_ON, pitch_tok,
                                   vel_to_token(event.velocity)])

            else:  # "off"
                event = payload[1]
                ch = event.channel
                if ch in (CH_DAC, CH_PSG_NOISE):
                    continue
                ch_tok = channel_to_token(ch)
                if ch_tok is None:
                    continue
                _advance_to(bar, beat)
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
        """Decode v7 token list → (NoteEvents, header dict).

        header keys:
          tempo_bpm, key_index, is_minor, meter, composer_token,
          game_token, context_token, loop_present,
          channel_roles, channel_patches_direct,
          channel_pans, lfo_state, ch3_mode, dac_enabled
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
            "channel_patches_direct": {},
            "channel_pans": {},
            "lfo_state": LFO_OFF,
            "ch3_mode": CH3_NORMAL_MODE,
            "dac_enabled": False,
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

        # Channel header entries: CH_tok, ROLE_tok, [44 patch tokens if FM]
        while pos < len(all_tokens):
            ch_tok = all_tokens[pos]
            is_fm_ch  = CH_FM_BASE <= ch_tok <= CH_FM_BASE + 5
            is_dac_ch = ch_tok == CH_DAC_TOK
            is_psg_ch = CH_PSG_BASE <= ch_tok <= CH_PSG_BASE + 2
            if not (is_fm_ch or is_dac_ch or is_psg_ch):
                break
            if pos + 1 >= len(all_tokens):
                break
            role_tok = all_tokens[pos + 1]
            if not (ROLE_TOKEN_BASE <= role_tok < ROLE_TOKEN_BASE + 7):
                break

            if is_fm_ch:
                ch = ch_tok - CH_FM_BASE
            elif is_dac_ch:
                ch = CH_DAC
            else:
                ch = CH_PSG_0 + (ch_tok - CH_PSG_BASE)

            header["channel_roles"][ch] = TOKEN_TO_ROLE.get(role_tok, ROLE_UNK)
            pos += 2

            if is_fm_ch:
                patch = decode_fm_patch_v7(all_tokens, pos)
                if patch is not None:
                    header["channel_patches_direct"][ch] = patch
                    pos += FM_PATCH_TOKENS_V7
                else:
                    # Graceful fallback: try v6 40-token patch
                    patch_v6 = decode_fm_patch(all_tokens, pos)
                    if patch_v6 is not None:
                        header["channel_patches_direct"][ch] = patch_v6
                        pos += FM_PATCH_TOKENS

        if tempo_bpm is not None:
            header["tempo_bpm"] = tempo_bpm

        bpm = header["tempo_bpm"]
        meter_num, meter_den = header["meter"]
        sixteenth           = SAMPLE_RATE * 60.0 / bpm / 4.0
        bar_sixteenth_count = meter_num * (4 if meter_den == 4 else 2)
        slots_per_bar       = 16
        bar_samples         = max(1, int(sixteenth * bar_sixteenth_count))
        beat_samples        = max(1, int(bar_samples / slots_per_bar))

        note_events: list[NoteEvent] = []
        open_notes:  dict[int, NoteEvent] = {}
        current_bar  = -1
        current_beat = 0

        def current_sample() -> int:
            return current_bar * bar_samples + current_beat * beat_samples

        it = iter(all_tokens[pos:])
        for tok in it:
            if tok in (EOS, PAD):
                break

            # --- Beat grid ---
            if tok == BAR:
                current_bar  += 1
                current_beat  = 0
            elif tok == DOWNBEAT or tok == HALFBEAT:
                pass  # zero-time structural hints
            elif BEAT_BASE <= tok < BEAT_BASE + self.subdivisions:
                current_beat = tok - BEAT_BASE

            # --- Global HW state tokens ---
            elif tok == LFO_OFF:
                header["lfo_state"] = LFO_OFF
            elif LFO_ON_BASE <= tok <= LFO_ON_BASE + 7:
                header["lfo_state"] = tok
            elif tok == CH3_NORMAL_MODE:
                header["ch3_mode"] = CH3_NORMAL_MODE
            elif tok == CH3_SPECIAL_MODE:
                header["ch3_mode"] = CH3_SPECIAL_MODE
            elif tok == DAC_DISABLE:
                header["dac_enabled"] = False
            elif tok == DAC_ENABLE:
                header["dac_enabled"] = True
            elif tok == LOOP_POINT:
                pass  # informational

            # --- DAC / PSG noise hits ---
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

            # --- FM channel events ---
            elif CH_FM_BASE <= tok <= CH_FM_BASE + 5:
                ch = tok - CH_FM_BASE
                self._decode_ch_event_v7(it, ch, current_sample(),
                                         open_notes, note_events, header)

            elif tok == CH_DAC_TOK:
                pass

            # --- PSG channel events ---
            elif CH_PSG_BASE <= tok < CH_PSG_BASE + 3:
                ch = CH_PSG_0 + (tok - CH_PSG_BASE)
                self._decode_ch_event_v7(it, ch, current_sample(),
                                         open_notes, note_events, header)

        final_sample = current_sample()
        for ch, note in open_notes.items():
            note.sample_off = final_sample
            note_events.append(note)

        note_events.sort(key=lambda e: (e.sample_on, e.channel))
        return note_events, header

    def _decode_ch_event_v7(
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

        elif action in _ALL_PAN_TOKS:
            # Zero-time pan change for this channel
            header["channel_pans"][ch] = action

        elif action == INSTRUMENT_CHANGE:
            # Read exactly FM_PATCH_TOKENS_V7 tokens from the iterator
            patch_toks: list[int] = []
            for _ in range(FM_PATCH_TOKENS_V7):
                try:
                    patch_toks.append(next(it))
                except StopIteration:
                    break
            if len(patch_toks) == FM_PATCH_TOKENS_V7:
                patch = decode_fm_patch_v7(patch_toks, 0)
                if patch is not None:
                    header["channel_patches_direct"][ch] = patch

    # ------------------------------------------------------------------
    # Transposition augmentation
    # ------------------------------------------------------------------

    def transpose(self, tokens: list[int], semitones: int) -> list[int]:
        """Shift all PITCH and KEY tokens by `semitones`."""
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
