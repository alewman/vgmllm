# VgmGPT v4 — Design Document

## Why v4

v3 proved the model can learn chip-level behaviour from raw register data: it
discovered FM channels, proper Key On/Off lifecycle, stereo panning, and
eventually began activating percussion and multiple voices simultaneously.

What v3 cannot learn from raw registers:
- Which channels are related (a PSG tone doubling an FM voice)
- When to be silent (arrangement / rests)
- Beat-level coordination between channels (shared clock invisible in the raw stream)
- Musical pitch identity (the same note C4 has different F-Number values at
  different blocks — the model must reverse-engineer logarithms)

The fundamental problem: ~44% of v3's 8192-token context is wait tokens that
carry no musical information.  The model's "working memory" of what it played
4 bars ago is pushed out by a river of timing bookkeeping.

v4 fixes all of this by moving from register-write tokens to note-event tokens.

---

## 1. Representation

### 1.1 Token vocabulary (320 tokens total)

```
IDs 0–3    Special:  PAD BOS EOS UNK
IDs 4–19   Tempo:    16 BPM bins (60,70,80,90,100,110,120,130,140,150,160,170,180,200,220,240)
IDs 20–43  Key:      24 tokens — C-major … B-major, C-minor … B-minor
IDs 44–47  Meter:    METER_44 METER_34 METER_68 METER_24
IDs 48–48  BAR:      bar-boundary marker
IDs 49–64  BEAT_0…BEAT_15: 16th-note positions within a bar (0=beat-1, 4=beat-2, …)
IDs 65–65  PHRASE_END: phrase boundary hint
IDs 66–71  CH_0…CH_5: FM channels 0–5 (select token)
IDs 72–72  CH_DAC: DAC channel select
IDs 73–75  CH_PSG0…CH_PSG2: PSG tone channels 0–2
IDs 76–82  ROLE_BASS ROLE_LEAD ROLE_HARM ROLE_COUNTER ROLE_DRUMS ROLE_PERC ROLE_UNK
IDs 83–210 PATCH_0…PATCH_127: FM patch library references (top-128 by corpus frequency)
IDs 211–213 NOTE_ON NOTE_OFF NOTE_HOLD
IDs 214–301 PITCH_0…PITCH_87: MIDI notes 24–111  (C1 – B7, covers full Genesis range)
IDs 302–317 VEL_0…VEL_15: velocity (0=quietest, 15=loudest)
IDs 318–318 DAC_HIT: DAC drum/sample event
IDs 319–319 PSG_NOISE_HIT: SN76489 noise channel hit
```

Compare to v3: 30,283 tokens → **320 tokens** (100× smaller vocabulary).
Every token is a human-understandable musical concept.

### 1.2 File token structure

```
BOS
<TEMPO_n> <KEY_n> <METER_44>
<CH_0> <ROLE_BASS>   <PATCH_12>
<CH_1> <ROLE_LEAD>   <PATCH_3>
<CH_2> <ROLE_HARM>   <PATCH_7>
<CH_3> <ROLE_HARM>   <PATCH_7>
<CH_DAC> <ROLE_DRUMS> <PATCH_0>
<CH_PSG0> <ROLE_PERC> <PATCH_0>
<BAR>
  <BEAT_0>  <CH_0> <NOTE_ON> <PITCH_C2> <VEL_9>
            <CH_DAC> <DAC_HIT>
  <BEAT_4>  <CH_0> <NOTE_OFF>
            <CH_1> <NOTE_ON> <PITCH_G4> <VEL_11>
  <BEAT_8>  <CH_0> <NOTE_ON> <PITCH_C2> <VEL_9>
            <PSG_NOISE_HIT>
  <BEAT_12> <CH_1> <NOTE_OFF>
<BAR>
  ...
EOS
```

### 1.3 Wait token elimination

v3: ~44% of context is WAIT tokens (no musical information).
v4: ZERO wait tokens.  Time is implicit in BEAT_n position within BAR blocks.
Effective musical information per token: **~8× higher** than v3.

### 1.4 Sub-beat timing

Arpeggios, LFO sweep, and driver micro-timing run faster than a 16th note.
These are **timbre effects**, not compositional events.  They belong in the FM
patch definition, not the composition token stream.  The vgm_synth layer
handles them deterministically given the patch.

---

## 2. Processing Pipeline

### 2.1 VGM → NoteEvents  (`ym2612.py`)

1. Parse raw VGM events (existing `vgm_parser.py`)
2. Replay YM2612 register state machine:
   - Track F-Number + Block per channel → convert to MIDI pitch via:
     `freq = F_number × YM2612_clock / (144 × 2^(21-block))`
     `midi = round(69 + 12 × log2(freq / 440))`
   - Track KEY_ON (register 0x28) → emit NoteEvent(on)
   - Track KEY_OFF → close open NoteEvent(off)
   - Extract FM patch (algorithm, feedback, Total Level per operator) on KEY_ON
   - Track DAC enable (register 0x2B bit 7) → CH6 is drums when set
3. Replay SN76489 register state machine:
   - Decode tone channels (3) → pitch + volume
   - Decode noise channel → DAC_HIT / PSG_NOISE_HIT
4. Output: `list[NoteEvent]`, `dict[ch → Ym2612Patch]`

### 2.2 Musical analysis  (`music_analysis.py`)

- **Tempo detection**: autocorrelation of note-onset histogram (60–300 BPM)
- **Key detection**: pitch-class histogram vs Krumhansl-Kessler profiles
  (24 rotations: 12 major + 12 minor)
- **Channel role classification**:
  - CH_DAC active (register 0x2B bit 7) → DRUMS
  - PSG noise channel → PERC
  - Mean pitch < MIDI 48 (C3) → BASS
  - Mean pitch ≥ MIDI 60 (C5) AND high note density → LEAD
  - Mean pitch ≥ MIDI 60 AND lower density → COUNTER
  - Otherwise → HARM

### 2.3 Patch library  (`tokenizer_v4.py` — `PatchLibrary`)

First pass over corpus:
1. Extract all FM patches (ALG, FB, TL×4, AR×4)
2. Fingerprint each patch as a frozen tuple
3. Frequency-rank, keep top 128
4. Save to `data/patch_library_v4.json`

Subsequent tokenization: lookup patch → ID; unseen patches → nearest by L1
distance on normalised parameter vector.

### 2.4 Encoding  (`tokenizer_v4.py` — `encode()`)

1. Run ym2612 decoder → note events + patch map
2. Run music analysis → tempo, key, roles
3. Emit header tokens
4. Assign each NoteEvent to (bar, beat_16th) from `sample_pos`
5. Walk time forward, emitting BAR / BEAT_n / note tokens in order
6. Emit EOS

### 2.5 Decoding  (`tokenizer_v4.py` — `decode()`)

1. Parse header → tempo, key, patch assignments
2. Walk token stream: reconstruct NoteEvent list with absolute sample positions
   `sample = bar × bar_samples + beat × beat_samples`
3. Lookup FM patches from patch library
4. Pass to vgm_synth to generate register writes → VGM file

### 2.6 VGM Synthesis  (`vgm_synth.py`)

Given NoteEvent list + FM patches:
1. Sort events by sample_pos
2. For FM channels:
   - On NOTE_ON: write patch registers (ALG/FB/TL/AR/DR/SR/RR/SL), then
     write F-Number+Block registers, then write KEY_ON (0x28)
   - On NOTE_OFF: write KEY_OFF (0x28 with op bits = 0)
3. For DAC: write PCM data or substitute FM6 kick patch
4. For PSG: write SN76489 tone/noise registers
5. Insert WAIT tokens between events (from sample position differences)
6. Wrap in valid VGM header (copy clock fields from prompt file or use defaults)

---

## 3. Data Augmentation

### 3.1 Transposition (12× dataset size)

- Transpose every training file to all 12 keys in token space
- Rotate PITCH tokens by ±n semitones; clamp to PITCH_0–PITCH_87
- Update KEY header token to match
- Cost: zero additional disk I/O (done on-the-fly in dataset loader)

### 3.2 Tempo augmentation (5× additional)

- Stretch tempo ±15% in 5% steps: ×0.85, ×0.90, ×1.00, ×1.05, ×1.10
- In token space: adjust TEMPO header token; BAR/BEAT positions unchanged
  (they're relative, so tempo stretch is just a header change)
- Combined with transposition: up to 60× effective dataset size

### 3.3 Corpus filtering (quality improvement)

Current corpus has ~11,309 files including jingles, SFX, 1-second fanfares.
Filter criteria:
- Duration < 8 seconds → discard
- Active FM channels < 2 for > 80% of duration → discard (SFX / mono beeps)
- Unique pitch count < 5 → discard (single-note drones)
- No KEY_ON events → discard (noise/atmosphere only)
Expected remaining: ~9,000–10,000 files of genuine music.

---

## 4. Architecture Changes

### 4.1 Vocabulary size

v3: 30,283 → v4: 320.
Embedding table: 320 × 768 = 245,760 params (< 0.5 MB).
The model can now dedicate far more capacity to musical relationships.

### 4.2 Context length

With 320-token vocabulary and 8× better information density per token, an
8192-token context covers ~65,000 samples worth of music — roughly 90 seconds
at 150 BPM.  For full 3-minute songs, target 16K context (achievable with
FlashAttention-2 on RTX 3090 without gradient checkpointing).

### 4.3 Model size

v3 had to learn chip-level timing, register encoding, and music simultaneously
with 136.5M params.  v4's model only needs to learn music.  A 136.5M model on
this vocabulary and context is almost certainly overparameterised — consider
starting with config_medium (~55M) and scaling up if needed.

---

## 5. Implementation Phases

### Phase 1: Core modules (build first)
- [x] `ym2612.py` — YM2612 + SN76489 state decoder → NoteEvents
- [x] `music_analysis.py` — tempo, key, channel role detection
- [x] `tokenizer_v4.py` — vocabulary, encode(), decode(), PatchLibrary
- [ ] `vgm_synth.py` — NoteEvents + patches → VGM binary

### Phase 2: Data pipeline
- [ ] Update `data_pipeline.py` with filtering (duration, channel count, pitch variety)
- [ ] `dataset_v4.py` — build_patch_library(), prepare_v4(), VgmDatasetV4
  - Transposition and tempo augmentation in __getitem__

### Phase 3: Training
- [ ] Update `train.py` to accept `--tokenizer v4`
  - vocab_size=320, context_len=16384 (or start with 8192)
  - Remove `--no-compile` requirement (v4 has smaller vocab, may compile cleanly)

### Phase 4: Generation
- [ ] Update `generate.py` for v4 tokenizer
  - Header conditioning: `--tempo 150 --key A_minor --meter 4/4`
  - Channel role specification: `--ch0-role bass --ch1-role lead`
  - No prompt VGM required (can generate from header tokens alone)

---

## 6. Open Questions

1. **Phrase boundary detection**: autocorrelation of inter-onset gaps + drop in
   note density?  Or skip for v4.0?
2. **CH3 special mode** (YM2612 can give CH3 operators independent pitches for
   complex timbres): treat as a single channel with the primary pitch, ignore
   per-operator pitch modulation for now.
3. **DAC samples**: v3 corpus has PCM drum kits embedded in VGM.  For v4
   synthesis, we can: (a) substitute a FM percussion patch, or (b) embed the
   most common PCM samples in the patch library.  Option (a) for v4.0.
4. **Loop handling**: Most VGM files loop.  Should we unroll 1 loop iteration
   for training to give the model full phrase context?  Or discard looped
   sections?  Likely: encode the loop body once, add loop-unroll as augmentation.

---

## 7. Expected Quality Gains

| Problem in v3 | v4 fix | Expected impact |
|---|---|---|
| ~44% context wasted on waits | No wait tokens | Much longer musical memory |
| Model doesn't know channel roles | Role header tokens | Arrangement learning |
| PSG/FM coupling invisible | Same-frame note events are adjacent | Coupling learnable |
| All channels active at once | Role + arrangement tokens | Selective silence |
| No beat clock | BAR / BEAT_n tokens | Rhythmic coordination |
| Key/pitch ambiguous | Explicit PITCH tokens + KEY header | Melodic structure |
| Patch = interleaved register writes | PATCH_n header + library | Timbre as first-class concept |
| Dataset: SFX and jingles contaminate | Filtering step | Cleaner training signal |
| Fixed key bias | Transposition augmentation (12×) | Key-agnostic composition |
