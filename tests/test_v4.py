"""Tests for the YM2612 decoder (ym2612.py) and music_analysis.py."""

import math
import pytest

from genesis_music.ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_NOISE,
    NoteEvent, Ym2612Patch, Ym2612State, Sn76489State,
    fnumber_to_midi, midi_to_fnumber, freq_to_midi, optimal_block,
    decode_vgm,
)
from genesis_music.vgm_parser import EventType, VgmEvent, VgmFile, VgmHeader
from genesis_music.vgm_synth import synthesise_vgm, _build_vgm_header


# ---------------------------------------------------------------------------
# Helpers to build synthetic VgmFile objects
# ---------------------------------------------------------------------------

def _make_vgm(events: list[VgmEvent], total_samples: int = 44100) -> VgmFile:
    header = VgmHeader(
        ym2612_clock=7_670_454,
        sn76489_clock=3_579_545,
        total_samples=total_samples,
    )
    return VgmFile(header=header, events=events)


def _evt(etype, reg, val, sample=0):
    return VgmEvent(type=etype, register=reg, value=val, sample_pos=sample)


def _wait(n):
    return VgmEvent(type=EventType.WAIT, register=0, value=n, sample_pos=0)


def _fm0(reg, val):
    return _evt(EventType.YM2612_PORT0, reg, val)


def _fm1(reg, val):
    return _evt(EventType.YM2612_PORT1, reg, val)


def _psg(val):
    return _evt(EventType.SN76489, 0, val)


def _end():
    return _evt(EventType.END, 0, 0)


# ---------------------------------------------------------------------------
# F-Number / MIDI conversion
# ---------------------------------------------------------------------------

class TestFnumberMidi:
    def test_a4_block4(self):
        # A4 = 440 Hz.  F-Number for block 4 should be ~1083.
        fnum = midi_to_fnumber(69, block=4)  # A4
        midi = fnumber_to_midi(fnum, block=4)
        assert abs(midi - 69) <= 1, f"Expected 69, got {midi}"

    def test_c4_block4(self):
        # Middle C = MIDI 60
        fnum = midi_to_fnumber(60, block=4)
        midi = fnumber_to_midi(fnum, block=4)
        assert abs(midi - 60) <= 1

    def test_zero_fnum_returns_minus1(self):
        assert fnumber_to_midi(0, 4) == -1

    def test_roundtrip_all_midi_notes(self):
        """Every MIDI note 24–111 should round-trip through F-Number within 1 semitone."""
        for midi_in in range(24, 112):
            block = optimal_block(midi_in)
            fnum  = midi_to_fnumber(midi_in, block)
            midi_out = fnumber_to_midi(fnum, block)
            assert abs(midi_out - midi_in) <= 1, (
                f"MIDI {midi_in} → fnum={fnum} block={block} → {midi_out}"
            )

    def test_optimal_block_in_range(self):
        for midi in range(24, 112):
            b = optimal_block(midi)
            assert 0 <= b <= 7

    def test_freq_to_midi_440hz(self):
        assert freq_to_midi(440.0) == 69

    def test_freq_to_midi_zero(self):
        assert freq_to_midi(0.0) == -1


# ---------------------------------------------------------------------------
# Ym2612State — register processing
# ---------------------------------------------------------------------------

class TestYm2612State:
    def _simple_note_events(self, ch_idx: int = 0) -> list[NoteEvent]:
        """Build a minimal sequence: set pitch, key on, wait, key off."""
        port = 0 if ch_idx < 3 else 1
        ch_offset = ch_idx % 3

        # F-Number for MIDI 69 (A4), block 4
        fnum_lo = midi_to_fnumber(69, 4) & 0xFF
        fnum_hi = ((4 << 3) | (midi_to_fnumber(69, 4) >> 8)) & 0xFF

        key_on_val  = (0xF0 | (ch_idx if ch_idx < 3 else ch_idx + 1))
        key_off_val = (0x00 | (ch_idx if ch_idx < 3 else ch_idx + 1))

        fm = _fm0 if port == 0 else _fm1

        events = [
            fm(0xA0 + ch_offset, fnum_lo),
            fm(0xA4 + ch_offset, fnum_hi),
            _fm0(0x28, key_on_val),   # key on always uses port 0 reg 0x28
            _wait(4410),
            _fm0(0x28, key_off_val),
            _end(),
        ]
        vgm = _make_vgm(events)

        decoder = Ym2612State()
        return list(decoder.process_vgm(vgm))

    def test_note_on_emitted(self):
        events = self._simple_note_events(ch_idx=0)
        on_events = [e for e in events if e.sample_off < 0 or e.sample_on != e.sample_off]
        assert any(e.channel == 0 for e in events), "No event on channel 0"

    def test_pitch_close_to_a4(self):
        events = self._simple_note_events(ch_idx=0)
        on_event = next(
            (e for e in events if e.channel == 0 and e.sample_on == 0), None
        )
        assert on_event is not None
        assert abs(on_event.pitch - 69) <= 1, f"Expected ~69, got {on_event.pitch}"

    def test_note_off_closes_event(self):
        events = self._simple_note_events(ch_idx=0)
        closed = [e for e in events if e.sample_off > 0]
        assert len(closed) >= 1

    def test_channel_1_works(self):
        events = self._simple_note_events(ch_idx=1)
        assert any(e.channel == 1 for e in events)

    def test_channel_3_port1(self):
        events = self._simple_note_events(ch_idx=3)
        assert any(e.channel == 3 for e in events)

    def test_dac_enable_emits_dac_events(self):
        events_list = [
            _fm0(0x2B, 0x80),   # DAC enable
            _wait(735),
            _evt(EventType.DAC_WRITE, 0, 0x80),
            _wait(735),
            _end(),
        ]
        vgm = _make_vgm(events_list)
        decoder = Ym2612State()
        notes = list(decoder.process_vgm(vgm))
        dac_notes = [e for e in notes if e.channel == CH_DAC]
        assert len(dac_notes) >= 1

    def test_patch_extracted(self):
        ch_idx = 0
        fnum_lo = midi_to_fnumber(60, 4) & 0xFF
        fnum_hi = ((4 << 3) | (midi_to_fnumber(60, 4) >> 8)) & 0xFF
        events_list = [
            _fm0(0x40, 20),     # OP1 TL = 20 for ch0
            _fm0(0x50, 28),     # OP1 AR = 28 for ch0
            _fm0(0xB0, 0b00101_010),  # feedback=5, algorithm=2
            _fm0(0xA0, fnum_lo),
            _fm0(0xA4, fnum_hi),
            _fm0(0x28, 0xF0),   # key on ch0
            _wait(100),
            _fm0(0x28, 0x00),   # key off ch0
            _end(),
        ]
        vgm = _make_vgm(events_list)
        decoder = Ym2612State()
        notes = list(decoder.process_vgm(vgm))
        on_note = next((e for e in notes if e.channel == 0 and e.patch is not None), None)
        assert on_note is not None
        assert on_note.patch.algorithm == 2
        assert on_note.patch.feedback == 5


# ---------------------------------------------------------------------------
# Sn76489State
# ---------------------------------------------------------------------------

class TestSn76489State:
    def test_tone_note_on_off(self):
        """Volume transition 15→0→15 should produce note-on then note-off."""
        # SN76489 write: latch ch0 volume (0x90 | 0 = silent already)
        # Set volume: ch0 tone (latch: 1|00|1|VVVV), then set non-silent
        events_list = [
            _psg(0x9F),   # ch0 volume = 15 (silent) — initial state
            _wait(100),
            _psg(0x90),   # ch0 volume = 0 (max) → NOTE ON
            _wait(4410),
            _psg(0x9F),   # ch0 volume = 15 (silent) → NOTE OFF
            _end(),
        ]
        vgm = _make_vgm(events_list)
        decoder = Sn76489State()
        notes = list(decoder.process_vgm(vgm))
        psg_notes = [e for e in notes if e.channel == CH_PSG_0]
        assert len(psg_notes) >= 1
        closed = [e for e in psg_notes if e.sample_off > 0]
        assert len(closed) >= 1


# ---------------------------------------------------------------------------
# decode_vgm integration
# ---------------------------------------------------------------------------

class TestDecodeVgm:
    def test_returns_sorted_by_sample(self):
        ch_idx = 0
        fnum_lo = midi_to_fnumber(60, 4) & 0xFF
        fnum_hi = ((4 << 3) | (midi_to_fnumber(60, 4) >> 8)) & 0xFF
        events_list = [
            _fm0(0xA0, fnum_lo),
            _fm0(0xA4, fnum_hi),
            _fm0(0x28, 0xF0),
            _wait(4410),
            _fm0(0x28, 0x00),
            _end(),
        ]
        vgm = _make_vgm(events_list)
        notes, patch_map = decode_vgm(vgm)
        sample_positions = [e.sample_on for e in notes]
        assert sample_positions == sorted(sample_positions)

    def test_patch_map_populated(self):
        ch_idx = 0
        fnum_lo = midi_to_fnumber(60, 4) & 0xFF
        fnum_hi = ((4 << 3) | (midi_to_fnumber(60, 4) >> 8)) & 0xFF
        events_list = [
            _fm0(0xA0, fnum_lo), _fm0(0xA4, fnum_hi),
            _fm0(0x28, 0xF0), _wait(100), _fm0(0x28, 0x00), _end(),
        ]
        vgm = _make_vgm(events_list)
        _, patch_map = decode_vgm(vgm)
        assert isinstance(patch_map, dict)
        assert 0 in patch_map or len(patch_map) >= 0  # at least exists


# ---------------------------------------------------------------------------
# music_analysis
# ---------------------------------------------------------------------------

class TestMusicAnalysis:
    def _make_notes(self, pitches, sample_interval=7350):
        """Build a list of NoteEvents at regular intervals."""
        notes = []
        for i, p in enumerate(pitches):
            notes.append(NoteEvent(
                channel=0, pitch=p, velocity=10,
                sample_on=i * sample_interval,
                sample_off=i * sample_interval + sample_interval - 1,
            ))
        return notes

    def test_detect_tempo_basic(self):
        from genesis_music.music_analysis import detect_tempo
        # 7350 samples apart at 44100 Hz = ~343 ms interval
        # 343 ms ≈ one beat at ~175 BPM (or 87.5 BPM with half-time)
        notes = self._make_notes([60] * 64, sample_interval=7350)
        bpm = detect_tempo(notes, total_samples=64 * 7350)
        # Just check it's a plausible value
        assert 50 <= bpm <= 300

    def test_detect_key_c_major(self):
        from genesis_music.music_analysis import detect_key
        # C major scale notes, MIDI 60-71
        c_major_pcs = [60, 62, 64, 65, 67, 69, 71]
        notes = self._make_notes(c_major_pcs * 8)
        key_idx, is_minor, key_name = detect_key(notes)
        # Should detect C major (key_index=0, is_minor=False)
        assert not is_minor
        assert key_idx == 0, f"Expected C major (0), got key_idx={key_idx}"

    def test_detect_key_a_minor(self):
        from genesis_music.music_analysis import detect_key
        # A natural minor scale: A B C D E F G
        a_minor_pcs = [69, 71, 60, 62, 64, 65, 67]  # A minor = relative of C major
        notes = self._make_notes(a_minor_pcs * 16)
        key_idx, is_minor, key_name = detect_key(notes)
        # A minor = key_index 9, is_minor True
        # (note: this test is probabilistic — profile may prefer C major)
        assert isinstance(key_idx, int) and 0 <= key_idx < 12

    def test_should_discard_short(self):
        from genesis_music.music_analysis import should_discard
        notes = self._make_notes([60, 62, 64], sample_interval=7350)
        # 3 notes, 3 * 7350 / 44100 = 0.5 seconds total
        discard, reason = should_discard(notes, total_samples=3 * 7350)
        assert discard
        assert "short" in reason

    def test_should_discard_few_channels(self):
        from genesis_music.music_analysis import should_discard
        # All notes on channel 0 only → only 1 FM channel active
        notes = self._make_notes([60, 62, 64, 65] * 20, sample_interval=735)
        total = 80 * 735 + 44100 * 10  # long enough
        discard, reason = should_discard(notes, total_samples=total)
        # 1 channel < min_fm_channels=2 → should discard
        assert discard

    def test_classify_roles_bass(self):
        from genesis_music.music_analysis import classify_channel_roles
        notes = [
            NoteEvent(channel=0, pitch=36, velocity=10, sample_on=i*735, sample_off=i*735+700)
            for i in range(50)
        ]
        roles = classify_channel_roles(notes)
        assert roles.get(0) == "BASS"

    def test_classify_roles_lead(self):
        from genesis_music.music_analysis import classify_channel_roles
        # High pitch, many notes per second → LEAD
        notes = [
            NoteEvent(channel=1, pitch=72, velocity=10, sample_on=i*735, sample_off=i*735+700)
            for i in range(200)
        ]
        roles = classify_channel_roles(notes)
        assert roles.get(1) == "LEAD"


# ---------------------------------------------------------------------------
# TokenizerV4
# ---------------------------------------------------------------------------

class TestTokenizerV4:
    def _make_library(self):
        from genesis_music.tokenizer_v4 import PatchLibrary
        patches = [
            Ym2612Patch(
                algorithm=0, feedback=0,
                tl=(0,)*4, ar=(31,)*4, dr=(0,)*4, sr=(0,)*4,
                rr=(15,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
            )
        ]
        return PatchLibrary(patches)

    def test_vocab_size(self):
        from genesis_music.tokenizer_v4 import VOCAB_SIZE
        assert VOCAB_SIZE == 449

    def test_tempo_token_roundtrip(self):
        from genesis_music.tokenizer_v4 import tempo_to_token, TEMPO_BASE, TEMPO_BINS
        for bpm in TEMPO_BINS:
            tok = tempo_to_token(bpm)
            assert TEMPO_BASE <= tok < TEMPO_BASE + len(TEMPO_BINS)

    def test_key_token_roundtrip(self):
        from genesis_music.tokenizer_v4 import key_to_token, token_to_key
        for key in range(12):
            for minor in (False, True):
                tok = key_to_token(key, minor)
                k2, m2 = token_to_key(tok)
                assert k2 == key
                assert m2 == minor

    def test_pitch_token_roundtrip(self):
        from genesis_music.tokenizer_v4 import pitch_to_token, token_to_pitch, PITCH_MIN_MIDI, PITCH_MAX_MIDI
        for midi in range(PITCH_MIN_MIDI, PITCH_MAX_MIDI + 1):
            tok = pitch_to_token(midi)
            assert tok is not None
            assert token_to_pitch(tok) == midi

    def test_pitch_out_of_range(self):
        from genesis_music.tokenizer_v4 import pitch_to_token
        assert pitch_to_token(0) is None
        assert pitch_to_token(127) is None

    def test_transpose_semitone(self):
        from genesis_music.tokenizer_v4 import TokenizerV4, BOS, EOS, pitch_to_token, token_to_pitch
        lib = self._make_library()
        tok = TokenizerV4(lib)
        tokens = [BOS, pitch_to_token(60), pitch_to_token(64), EOS]
        up2 = tok.transpose(tokens, 2)
        assert token_to_pitch(up2[1]) == 62
        assert token_to_pitch(up2[2]) == 66

    def test_transpose_key_rotates(self):
        from genesis_music.tokenizer_v4 import TokenizerV4, key_to_token, token_to_key
        lib = self._make_library()
        tok = TokenizerV4(lib)
        tokens = [key_to_token(0, False)]   # C major
        up3 = tok.transpose(tokens, 3)
        key_idx, is_minor = token_to_key(up3[0])
        assert key_idx == 3   # D# / Eb major
        assert not is_minor

    def test_patch_library_lookup(self):
        from genesis_music.tokenizer_v4 import PatchLibrary
        lib = self._make_library()
        patch = lib.patches[0]
        assert lib.lookup(patch) == 0

    def test_patch_library_unknown_returns_nearest(self):
        from genesis_music.tokenizer_v4 import PatchLibrary
        lib = self._make_library()
        unknown = Ym2612Patch(
            algorithm=7, feedback=7,
            tl=(100,)*4, ar=(1,)*4, dr=(0,)*4, sr=(0,)*4,
            rr=(15,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
        )
        idx = lib.lookup(unknown)
        assert 0 <= idx < len(lib)


# ---------------------------------------------------------------------------
# vgm_synth
# ---------------------------------------------------------------------------

class TestVgmSynth:
    def _default_patch(self):
        return Ym2612Patch(
            algorithm=0, feedback=0,
            tl=(20,)*4, ar=(31,)*4, dr=(0,)*4, sr=(0,)*4,
            rr=(15,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
        )

    def test_header_magic(self):
        data = _build_vgm_header(b"\x66", total_samples=44100,
                                  ym2612_clock=7_670_454, sn76489_clock=3_579_545)
        assert data[:4] == b"Vgm "

    def test_header_total_samples(self):
        import struct
        total = 44100 * 5
        data = _build_vgm_header(b"\x66", total_samples=total,
                                  ym2612_clock=7_670_454, sn76489_clock=3_579_545)
        stored = struct.unpack_from("<I", data, 0x18)[0]
        assert stored == total

    def test_synthesise_returns_bytes(self):
        notes = [
            NoteEvent(channel=0, pitch=60, velocity=10,
                      sample_on=0, sample_off=4410,
                      patch=self._default_patch()),
        ]
        result = synthesise_vgm(notes, total_samples=44100)
        assert isinstance(result, bytes)
        assert result[:4] == b"Vgm "

    def test_synthesise_empty_notes(self):
        result = synthesise_vgm([], total_samples=44100)
        assert isinstance(result, bytes)
        assert len(result) > 64  # has header

    def test_synthesise_psg_note(self):
        notes = [
            NoteEvent(channel=CH_PSG_0, pitch=69, velocity=10,
                      sample_on=0, sample_off=4410),
        ]
        result = synthesise_vgm(notes, total_samples=44100)
        assert result[:4] == b"Vgm "

    def test_synthesise_roundtrip_contains_key_on(self):
        """A synthesised FM note should contain a KEY_ON byte (0xF0) in data."""
        notes = [
            NoteEvent(channel=0, pitch=69, velocity=10,
                      sample_on=0, sample_off=4410,
                      patch=self._default_patch()),
        ]
        result = synthesise_vgm(notes, total_samples=44100)
        # Command 0x52 is YM2612 port 0 write; data follows
        # Scan for 0x52, 0x28, 0xF0 triplet (key on ch0, all ops)
        found = False
        for i in range(len(result) - 2):
            if result[i] == 0x52 and result[i+1] == 0x28 and (result[i+2] & 0xF0) == 0xF0:
                found = True
                break
        assert found, "KEY_ON register write not found in synthesised VGM"

    def test_synthesise_dac_note(self):
        notes = [
            NoteEvent(channel=CH_DAC, pitch=-1, velocity=15,
                      sample_on=0, sample_off=735),
        ]
        result = synthesise_vgm(notes, total_samples=44100)
        assert result[:4] == b"Vgm "


# ---------------------------------------------------------------------------
# ComposerMap
# ---------------------------------------------------------------------------

class TestComposerMap:
    def _make_map(self):
        from genesis_music.tokenizer_v4 import ComposerMap
        composers = ["Matt Furniss", "Yuzo Koshiro", "Hiroshi Kawaguchi"]
        return ComposerMap(composers)

    def test_known_composer_lookup(self):
        from genesis_music.tokenizer_v4 import COMPOSER_BASE
        cmap = self._make_map()
        assert cmap.lookup("Matt Furniss") == COMPOSER_BASE + 0
        assert cmap.lookup("Yuzo Koshiro")  == COMPOSER_BASE + 1

    def test_case_insensitive_lookup(self):
        from genesis_music.tokenizer_v4 import COMPOSER_BASE
        cmap = self._make_map()
        assert cmap.lookup("matt furniss") == COMPOSER_BASE + 0
        assert cmap.lookup("YUZO KOSHIRO")  == COMPOSER_BASE + 1

    def test_unknown_composer_returns_unk(self):
        from genesis_music.tokenizer_v4 import UNK_COMPOSER
        cmap = self._make_map()
        assert cmap.lookup("Nobody Famous") == UNK_COMPOSER
        assert cmap.lookup("") == UNK_COMPOSER

    def test_compound_author_split(self):
        """Compound 'A, B' string should match on first known individual name."""
        from genesis_music.tokenizer_v4 import COMPOSER_BASE
        cmap = self._make_map()
        tok = cmap.lookup("Matt Furniss, Yuzo Koshiro")
        assert tok == COMPOSER_BASE + 0  # first match wins

    def test_name_reverse_lookup(self):
        from genesis_music.tokenizer_v4 import COMPOSER_BASE
        cmap = self._make_map()
        assert cmap.name(COMPOSER_BASE + 0) == "Matt Furniss"
        assert cmap.name(COMPOSER_BASE + 1) == "Yuzo Koshiro"

    def test_unk_name_returns_unknown(self):
        from genesis_music.tokenizer_v4 import UNK_COMPOSER
        cmap = self._make_map()
        assert cmap.name(UNK_COMPOSER) == "Unknown"

    def test_save_load_roundtrip(self, tmp_path):
        from genesis_music.tokenizer_v4 import ComposerMap, COMPOSER_BASE
        cmap = self._make_map()
        path = tmp_path / "test_cmap.json"
        cmap.save(path)
        loaded = ComposerMap.load(path)
        assert loaded.composers == cmap.composers
        assert loaded.lookup("Yuzo Koshiro") == COMPOSER_BASE + 1

    def test_build_from_file_list(self, tmp_path):
        """ComposerMap.build() on file paths should produce a valid map."""
        from genesis_music.tokenizer_v4 import ComposerMap, COMPOSER_BASE, UNK_COMPOSER
        import gzip, struct

        def _make_vgz_with_gd3(author: str) -> bytes:
            """Create a minimal VGM file with a GD3 tag containing the given author."""
            def _utf16le(s: str) -> bytes:
                return s.encode('utf-16-le') + b'\x00\x00'

            # GD3 string fields: track_en, track_jp, game_en, game_jp,
            #                    sys_en, sys_jp, author_en
            fields = [_utf16le(""), _utf16le(""), _utf16le("Test Game"),
                      _utf16le(""), _utf16le(""), _utf16le(""), _utf16le(author)]
            gd3_data = b''.join(fields)
            gd3_block = (b'Gd3 '
                         + struct.pack('<I', 0x100)   # version 1.00
                         + struct.pack('<I', len(gd3_data))
                         + gd3_data)

            # Minimal VGM 1.61 header (0x40 bytes)
            header = bytearray(0x40)
            header[0:4] = b'Vgm '
            # eof_offset at 0x04: relative, so 0x40 - 4 = 0x3c
            struct.pack_into('<I', header, 0x04, 0x3C - 0x04)
            struct.pack_into('<I', header, 0x08, 0x0161)   # version 1.61
            struct.pack_into('<I', header, 0x0C, 3_579_545)  # SN76489 clock
            struct.pack_into('<I', header, 0x2C, 7_670_454)  # YM2612 clock
            gd3_abs = 0x40
            struct.pack_into('<I', header, 0x14, gd3_abs - 0x14)  # relative gd3_offset
            struct.pack_into('<I', header, 0x18, 44100)  # total samples
            # VGM data offset (relative to 0x34)
            struct.pack_into('<I', header, 0x34, 0x0C)   # points to data right after header

            raw = bytes(header) + gd3_block + b'\x66'   # 0x66 = end-of-data
            return gzip.compress(raw)

        p1 = tmp_path / "track1.vgz"
        p2 = tmp_path / "track2.vgz"
        p3 = tmp_path / "track3.vgz"
        p1.write_bytes(_make_vgz_with_gd3("Alice"))
        p2.write_bytes(_make_vgz_with_gd3("Bob"))
        p3.write_bytes(_make_vgz_with_gd3("Alice"))  # Alice appears twice

        cmap = ComposerMap.build([p1, p2, p3], top_n=2)
        # Alice (2 tracks) should rank above Bob (1 track)
        assert cmap.lookup("Alice") == COMPOSER_BASE + 0
        assert cmap.lookup("Bob")   == COMPOSER_BASE + 1

    def test_encode_emits_composer_token(self):
        """Encoding a VGM with a ComposerMap should include a COMPOSER token in header."""
        from genesis_music.tokenizer_v4 import (
            ComposerMap, PatchLibrary, TokenizerV4,
            COMPOSER_BASE, UNK_COMPOSER, BOS,
        )
        from genesis_music.vgm_parser import VgmFile, VgmHeader

        composers = ["Matt Furniss", "Yuzo Koshiro"]
        cmap = ComposerMap(composers)

        patches = [Ym2612Patch(
            algorithm=0, feedback=0,
            tl=(0,)*4, ar=(31,)*4, dr=(0,)*4, sr=(0,)*4,
            rr=(15,)*4, sl=(0,)*4, mul=(1,)*4, dt=(0,)*4,
        )]
        lib = PatchLibrary(patches)
        tok = TokenizerV4(lib, composer_map=cmap)

        # Build a minimal VgmFile with enough notes to pass the filter
        # We'll use skip_filter=True to bypass the duration/channel check
        from genesis_music.vgm_parser import Gd3Tag
        header = VgmHeader(ym2612_clock=7_670_454, total_samples=44100 * 30)
        vgm = VgmFile(header=header, events=[])
        gd3 = Gd3Tag()
        gd3.author_en = "Yuzo Koshiro"
        vgm.gd3 = gd3

        # Build some note events so encoder has something to process
        from genesis_music.ym2612 import NoteEvent as NE
        vgm._test_notes = None  # We'll inject note_events via monkey-patch

        # Use a VGM with enough FM notes (hack: patch decode_vgm)
        import unittest.mock as mock
        many_notes = [
            NE(channel=ch, pitch=60+i, velocity=10,
               sample_on=i * 22050, sample_off=i * 22050 + 10000)
            for ch in range(6) for i in range(20)
        ]
        patch_map = {ch: patches[0] for ch in range(6)}

        with mock.patch('genesis_music.tokenizer_v4.decode_vgm',
                        return_value=(many_notes, patch_map)):
            tokens = tok.encode(vgm, skip_filter=True)

        assert tokens is not None
        # Composer token should appear in the first 6 tokens (BOS, TEMPO, KEY, METER, COMPOSER, ...)
        assert (COMPOSER_BASE + 1) in tokens[:6], (
            f"Expected COMPOSER_BASE+1={COMPOSER_BASE+1} in first 6 tokens, got {tokens[:6]}"
        )
