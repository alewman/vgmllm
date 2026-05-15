# Run v2: Conditional Multi-Chip Transformer

## Goal

Generate "fusion" tracks that never existed on original hardware by prompting
with metadata like composer, system, and game tags. Move from the v1
unconditional baseline to conditional multi-chip generation.

Example prompt:
  COMPOSER:Yuzo_Koshiro SYSTEM:Master_System_FM SYSTEM:Genesis

---

## 1. Metadata Injection (GD3 Tags)

### 1.1 Token Layout (new)

    ID 0-3:        PAD, BOS, EOS, UNK                           (unchanged)
    ID 4-9:        META_COMPOSER, META_GAME, META_SYSTEM,        (new)
                   META_DATE, META_REGION, META_CHIP
    ID 10-73:      64 log-spaced wait bins                       (shifted from 4-67)
    ID 74+:        Event tokens (chip-prefixed, see section 2)   (shifted from 68+)

The 6 meta-field tokens are "key" tokens. Each is followed by a "value" token
from the metadata vocabulary (see 1.3). A full prefix looks like:

    BOS META_COMPOSER composer:yuzo_koshiro META_GAME game:thunder_force_iii
        META_SYSTEM system:genesis META_SYSTEM system:master_system_fm
        META_CHIP chip:ym2612 META_CHIP chip:ym2413 META_CHIP chip:sn76489
        ... event tokens ...

Multiple META_SYSTEM and META_CHIP tags are allowed for fusion tracks.

### 1.2 String Normalization

All GD3 string values are normalized before tokenization:

    def normalize_gd3(raw: str) -> str:
        s = raw.strip().lower()
        s = unicodedata.normalize("NFKD", s)        # decompose unicode
        s = re.sub(r"[^\w\s-]", "", s)               # strip punctuation
        s = re.sub(r"\s+", "_", s)                    # spaces to underscores
        return s

    # "Yuzo Koshiro"     -> "yuzo_koshiro"
    # "Y. Koshiro"       -> "y_koshiro"         (different token — see 1.3)
    # "Streets of Rage"  -> "streets_of_rage"
    # "ベア・ナックル"      -> "beanaxtukuru"      (NFKD + strip)

### 1.3 Alias Resolution

Many composers/games have inconsistent naming across VGM rips. We build an
alias table from the corpus:

    COMPOSER_ALIASES = {
        "y_koshiro":           "yuzo_koshiro",
        "y._koshiro":          "yuzo_koshiro",
        "koshiro_yuzo":        "yuzo_koshiro",
        "motoi_sakuraba":      "motoi_sakuraba",
        "sakuraba_motoi":      "motoi_sakuraba",
        ...
    }

Steps to build:
  1. Extract all GD3 author fields from the 19K corpus
  2. Normalize with normalize_gd3()
  3. Cluster by edit distance (threshold: 0.3 Levenshtein ratio)
  4. Manual review pass for ambiguous clusters
  5. Save as aliases.json

### 1.4 Metadata Vocab Size Estimate

    Unique composers (after aliasing):    ~200-400
    Unique games:                         ~1200 (one per VGM pack roughly)
    Unique systems:                       ~5-8  (genesis, master_system, game_gear, etc.)
    Unique chips:                         ~4    (ym2612, ym2413, sn76489, dual_ym2612)
    Date buckets (by year):               ~15   (1985-2000)
    Region tags:                          ~3    (japan, usa, europe)

    Total metadata value tokens:          ~1,400-1,650

Impact on embedding table: +1,650 rows x 768 dims = +1.27M params = ~2.4 MB in BF16.
Negligible VRAM impact.

---

## 2. Multi-Chip Vocabulary

### 2.1 Chip-Prefixed Event Tokens

v1 token format:  YM2612_PORT0:2A:7F  (no chip disambiguation for multi-system)
v2 token format:  Prefix every event token with explicit chip namespace:

    GENESIS_FM0:2A:7F       (YM2612 port 0 register 0x2A = 0x7F)
    GENESIS_FM1:B4:C0       (YM2612 port 1 register 0xB4 = 0xC0)
    GENESIS_PSG:00:9F       (SN76489 on Genesis)
    SMS_FM:09:00            (YM2413 register 0x09 = 0x00)
    SMS_PSG:00:BF           (SN76489 on Master System)
    DUAL_FM0:2A:7F          (Virtual second YM2612, port 0)
    DUAL_FM1:B4:C0          (Virtual second YM2612, port 1)

### 2.2 Chip Namespace Enumeration

    class ChipNamespace(Enum):
        # --- Genesis / Mega Drive ---
        GENESIS_FM0  = "GENESIS_FM0"    # YM2612 port 0 (channels 1-3)   VGM: 0x52
        GENESIS_FM1  = "GENESIS_FM1"    # YM2612 port 1 (channels 4-6)   VGM: 0x53
        GENESIS_PSG  = "GENESIS_PSG"    # SN76489 via Genesis             VGM: 0x50
        DUAL_FM0     = "DUAL_FM0"       # 2nd YM2612 port 0              VGM: 0xA2
        DUAL_FM1     = "DUAL_FM1"       # 2nd YM2612 port 1              VGM: 0xA3
        # --- Master System ---
        SMS_FM       = "SMS_FM"         # YM2413 (OPLL)                  VGM: 0x51
        SMS_PSG      = "SMS_PSG"        # SN76489 via Master System       VGM: 0x50
        # --- OPN Family (NEW — VGMPlay-playable) ---
        ARCADE_FM    = "ARCADE_FM"      # YM2151 (OPM), 8 FM ch           VGM: 0x54
        OPN_FM       = "OPN_FM"         # YM2203 (OPN), 3 FM ch           VGM: 0x55
        OPNA_FM0     = "OPNA_FM0"       # YM2608 (OPNA) port 0            VGM: 0x56
        OPNA_FM1     = "OPNA_FM1"       # YM2608 (OPNA) port 1            VGM: 0x57
        NEOGEO_FM0   = "NEOGEO_FM0"     # YM2610 (OPNB) port 0            VGM: 0x58
        NEOGEO_FM1   = "NEOGEO_FM1"     # YM2610 (OPNB) port 1            VGM: 0x59

### 2.3 Vocab Size Estimate

    v1 event tokens:         ~31,976  (YM2612 + SN76489 combined)
    v2 chip-prefixed tokens: ~55,000-65,000  (estimated with OPN family)

    Breakdown:
      GENESIS_FM0:   256 regs x up to 256 vals = ~15,000 seen in corpus
      GENESIS_FM1:   similar range              = ~12,000
      GENESIS_PSG:   single-byte commands       = ~256
      SMS_FM:        64 regs x 256 vals         = ~3,000 (smaller register space)
      SMS_PSG:       single-byte commands        = ~256
      DUAL_FM0/FM1:  mapped from GENESIS_FM     = 0 new (alias tokens, see 2.4)
      ARCADE_FM:     256 regs x ~200 vals       = ~4,000 (YM2151, 1,667 tracks)
      OPN_FM:        256 regs x ~150 vals       = ~2,500 (YM2203, 472 tracks)
      NEOGEO_FM0:    256 regs x ~200 vals       = ~3,500 (YM2610 port 0, 596 tracks)
      NEOGEO_FM1:    similar range              = ~2,500 (YM2610 port 1)
      OPNA_FM0/FM1:  smaller corpus             = ~1,000 (YM2608, 18 tracks)

    Total vocab:  4 special + 6 meta-key + ~1,800 meta-value + 64 wait + ~60,000 event
                = ~61,874

    Embedding table: 61,874 x 768 = 47.5M params = ~90.9 MB in BF16
    (v1 was 32,041 x 768 = 24.6M params = ~47.1 MB)
    Delta: +43.8 MB — still well within 24GB VRAM

### 2.4 Dual-Chip Token Mapping

For "impossible hardware" (two YM2612s), we don't double the event vocab.
Instead, DUAL_FM0/DUAL_FM1 tokens share embeddings with GENESIS_FM0/GENESIS_FM1
via a learned offset vector:

    class DualChipEmbedding(nn.Module):
        def __init__(self, base_embed, dual_offset_dim):
            self.base = base_embed
            self.dual_offset = nn.Parameter(torch.zeros(1, dual_offset_dim))

        def forward(self, token_ids):
            embeds = self.base(token_ids)
            is_dual = (token_ids >= DUAL_FM0_START) & (token_ids <= DUAL_FM1_END)
            embeds[is_dual] += self.dual_offset
            return embeds

This way the model understands DUAL_FM writes are "the same kind of thing" as
GENESIS_FM writes but on a different virtual chip. Adds only 768 parameters.

Alternative (simpler): just add DUAL tokens as normal vocab entries. The
shared-embedding approach is an optimization we can try if quality is poor.

---

## 3. Cross-Platform Training

### 3.1 System Detection by Clock Values

    def classify_system(header: VgmHeader) -> str:
        """Classify primary system. A file may match multiple."""
        if header.ym2612_clock > 0:
            return "genesis"
        elif header.ym2610_clock > 0:
            return "neo_geo"
        elif header.ym2608_clock > 0:
            return "opna"
        elif header.ym2203_clock > 0:
            return "opn"
        elif header.ym2151_clock > 0:
            return "arcade"
        elif header.ym2413_clock > 0:
            return "master_system_fm"
        elif header.sn76489_clock > 0:
            return "master_system"    # PSG-only SMS
        else:
            return "unknown"

### 3.2 Delta-Time Normalization

All chips must share a global clock for WAIT tokens. VGM already normalizes
this: all sample counts are at 44100 Hz regardless of the source system.

    Genesis native FM rate:     7670453 Hz (/ 144 = ~53 kHz sample rate)
    SMS YM2413 rate:            3579545 Hz
    VGM standard:               44100 Hz (all waits expressed in this)

No conversion needed — VGM format already normalizes timing. Our existing 64
log-spaced wait bins work identically across systems. A WAIT_735 token means
exactly 1/60th second whether it came from a Genesis or Master System track.

### 3.3 Dataset Composition

    Current corpus:  ~14,000 Genesis tracks (YM2612 + SN76489)
                     ~2,000-3,000 SMS FM tracks (YM2413 + SN76489, subset of 19K)
                     ~2,000-3,000 other/arcade (filtered or kept based on chips)

    v2 filtering:
      - Keep: genesis (ym2612_clock > 0)
      - Keep: master_system_fm (ym2413_clock > 0)
      - Keep: neo_geo (ym2610_clock > 0) — OPN-family, closest to YM2612
      - Keep: opn (ym2203_clock > 0) — OPN ancestor, 3 FM channels
      - Keep: opna (ym2608_clock > 0) — OPN superset, 6 FM + SSG + ADPCM
      - Consider: arcade (ym2151_clock > 0) — different register map, v3?
      - Drop: OKIM6295-only (pure ADPCM, no FM synthesis)
      - Drop: OPL2/OPL3 (different FM engine, incompatible register layout)
      - Drop: NES, POKEY, HuC6280, AY-3-8910 only (too different)
      - Drop: files with no GD3 tags (can't condition without metadata)

    Estimated v2 corpus: ~15,000-16,000 tracks (all Yamaha OPN-family + OPLL)

### 3.4 Cross-System Sequence Format

A Genesis track becomes:
    BOS META_COMPOSER composer:yuzo_koshiro META_GAME game:streets_of_rage
    META_SYSTEM system:genesis META_CHIP chip:ym2612 META_CHIP chip:sn76489
    GENESIS_FM0:28:F0 WAIT_735 GENESIS_PSG:00:9F GENESIS_FM1:B4:C0 ...
    EOS

A Master System FM track becomes:
    BOS META_COMPOSER composer:tokuhiko_uwabo META_GAME game:phantasy_star
    META_SYSTEM system:master_system_fm META_CHIP chip:ym2413 META_CHIP chip:sn76489
    SMS_FM:00:16 WAIT_735 SMS_PSG:00:BF SMS_FM:10:01 ...
    EOS

A fusion prompt at generation time:
    BOS META_COMPOSER composer:yuzo_koshiro
    META_SYSTEM system:genesis META_SYSTEM system:master_system_fm
    META_CHIP chip:ym2612 META_CHIP chip:ym2413 META_CHIP chip:sn76489

A Neo Geo track becomes:
    BOS META_COMPOSER composer:snk_sound_team META_GAME game:metal_slug
    META_SYSTEM system:neo_geo META_CHIP chip:ym2610
    NEOGEO_FM0:28:F0 WAIT_735 NEOGEO_FM1:01:3C ...
    EOS

A "Genesis Neo" fusion prompt (impossible hardware, playable in VGMPlay):
    BOS META_SYSTEM system:genesis META_SYSTEM system:neo_geo
    META_CHIP chip:ym2612 META_CHIP chip:ym2610 META_CHIP chip:sn76489
    (Model generates interleaved GENESIS_FM and NEOGEO_FM events)

The model has never seen this combo in training, but has learned what each chip
sounds like and what Yuzo Koshiro's compositional patterns are. The hypothesis
is that it will generate interleaved GENESIS_FM and SMS_FM events.

---

## 4. Hardware Efficiency (RTX 3090 24GB Budget)

### 4.1 VRAM Budget

    Component                     v1 (MB)     v2 (MB)     Delta
    ---------------------------------------------------------------
    Embedding (vocab x 768)        47.1        90.9       +43.8
    Transformer blocks             504.0       504.0        0.0
    LM head (tied w/ embed)          0.0         0.0        0.0
    Optimizer (AdamW 2x FP32)    1,102.0     1,188.0      +86.0
    Gradients (FP32)               551.0       594.0      +43.0
    Activations (BF16, bs=4)    ~19,000     ~19,000        0.0
    CUDA overhead                ~1,500      ~1,500        0.0
    ---------------------------------------------------------------
    Total                       ~22,704     ~22,877     +173 MB

    Headroom: 24,576 - 22,877 = 1,699 MB — safe.

The expanded OPN-family vocab adds ~173 MB total. Context window stays at 4096.
Metadata prefix consumes ~10-25 tokens (more chips = more META_CHIP tags).

### 4.2 Batch Size

batch_size=4 with grad_accum=8 remains the optimal config. No change needed.

### 4.3 Sequence Length

Keep 4096. The metadata prefix (10-20 tokens) is < 0.5% of the context window.
Average Genesis track is ~170K tokens — many context windows per track, and only
the first window per file gets the metadata prefix.

Wait — this is a design decision:

    Option A: Prefix only first chunk per file
      - Pro: 99.5% of context is music
      - Con: Most training windows have no metadata → model barely learns conditioning

    Option B: Prefix EVERY chunk with metadata
      - Pro: Model always sees conditioning context → strong metadata association
      - Con: Wastes ~15 tokens per 4096-token window (0.4% overhead)
      - RECOMMENDED — 0.4% overhead is trivial, conditioning quality is critical

### 4.4 Training Speed

Same model size (138M params), same hardware optimizations. Marginal slowdown
from larger vocab (softmax over ~62K vs 32K) — estimated < 5% impact on tok/s.
Training time: ~84 hours (vs ~80 hours for v1).

---

## 5. Implementation Roadmap

### Phase 1: Metadata Pipeline (est. 2-3 hours)

    1. Add normalize_gd3() to tokenizer.py
    2. Scan corpus for all unique GD3 values, build alias table
    3. Add classify_system() to vgm_parser.py
    4. Generate aliases.json (manual review step)
    5. Add metadata token definitions to tokenizer

### Phase 2: Multi-Chip Tokenizer (est. 3-4 hours)

    1. Add ChipNamespace enum to tokenizer
    2. Modify encode_vgm() to prefix events with chip namespace based on
       classify_system() result
    3. Modify build_vocab() to use chip-prefixed event keys
    4. Add DUAL_FM0/DUAL_FM1 namespace support
    5. Update decode_tokens() for new format
    6. Update vgm_writer to handle multi-chip output
    7. Update tests

### Phase 3: Dataset Rebuild (est. 1-2 hours compute)

    1. Rebuild vocab from full corpus with new tokenizer
    2. Rebuild dataset with metadata prefixes on every chunk
    3. Verify train/val split, token counts, meta.json

### Phase 4: Model Update (est. 1 hour)

    1. Update model config with new vocab_size
    2. Optional: DualChipEmbedding wrapper
    3. Verify forward pass, VRAM usage with dummy batch

### Phase 5: Training (est. 80-85 hours)

    1. Launch training: same hyperparameters as v1
    2. Monitor loss curve — compare to v1 baseline
    3. Evaluate at checkpoints using conditional prompts

### Phase 6: Generation & Evaluation (est. 2-3 hours)

    1. Update generate.py to accept metadata prompt args
    2. Generate test tracks:
       a. Unconditional (no metadata) — compare to v1
       b. Single-system conditioned (Genesis only, SMS only)
       c. Single-composer conditioned
       d. Fusion: cross-system, cross-composer
    3. Convert to VGM, play in VGM player, evaluate quality

---

## 6. VGM Writer Updates for Multi-Chip Output

The v1 writer only emits YM2612 (0x52/0x53) and SN76489 (0x50) commands.
v2 must also handle the full OPN family:

    YM2413 (OPLL/SMS FM):   VGM command 0x51 (reg, val)
    YM2151 (OPM/Arcade):    VGM command 0x54 (reg, val)
    YM2203 (OPN):           VGM command 0x55 (reg, val)
    YM2608 (OPNA) port 0/1: VGM command 0x56/0x57 (reg, val)
    YM2610 (OPNB) port 0/1: VGM command 0x58/0x59 (reg, val)
    Dual YM2612 port 0/1:   VGM command 0xA2/0xA3 (reg, val)

    def events_to_vgm_v2(events, chips_used):
        header = build_header_v2(chips_used)   # set correct clock fields
        for event in events:
            match event.chip:
                case ChipNamespace.GENESIS_FM0:  emit(0x52, reg, val)
                case ChipNamespace.GENESIS_FM1:  emit(0x53, reg, val)
                case ChipNamespace.SMS_FM:       emit(0x51, reg, val)
                case ChipNamespace.GENESIS_PSG | ChipNamespace.SMS_PSG:
                                                 emit(0x50, val)
                case ChipNamespace.DUAL_FM0:     emit(0xA2, reg, val)
                case ChipNamespace.DUAL_FM1:     emit(0xA3, reg, val)
                case ChipNamespace.ARCADE_FM:    emit(0x54, reg, val)
                case ChipNamespace.OPN_FM:       emit(0x55, reg, val)
                case ChipNamespace.OPNA_FM0:     emit(0x56, reg, val)
                case ChipNamespace.OPNA_FM1:     emit(0x57, reg, val)
                case ChipNamespace.NEOGEO_FM0:   emit(0x58, reg, val)
                case ChipNamespace.NEOGEO_FM1:   emit(0x59, reg, val)

The VGM header must set clock fields for all chips present:
    - offset 0x0C: sn76489_clock (3579545 if PSG used)
    - offset 0x10: ym2413_clock  (3579545 if SMS FM used)
    - offset 0x2C: ym2612_clock  (7670453 if Genesis FM used)
    - offset 0x30: ym2151_clock  (3579545 if Arcade FM used)
    - offset 0x44: ym2203_clock  (3579545 if OPN used)
    - offset 0x48: ym2608_clock  (7987200 if OPNA used)
    - offset 0x4C: ym2610_clock  (8000000 if Neo Geo used)

For dual chips, set bit 30 of the clock field. VGM version must be >= 1.61
for the extended header fields (0x44+).

---

## 7. VGMPlay Compatibility

VGMPlay (and its successor vgmplay-libvgm) is the reference VGM player that
emulates every chip independently. This means ANY combination of chips in a
single VGM file is playable — including "impossible hardware" combos that
never existed on any real console.

### 7.1 VGM Command → Chip Mapping (used by writer)

    Cmd     Chip           Ports  Header Clock Offset
    0x50    SN76489        -      0x0C
    0x51    YM2413 (OPLL)  -      0x10
    0x52    YM2612 port 0  0      0x2C
    0x53    YM2612 port 1  1      0x2C
    0x54    YM2151 (OPM)   -      0x30
    0x55    YM2203 (OPN)   -      0x44
    0x56    YM2608 port 0  0      0x48
    0x57    YM2608 port 1  1      0x48
    0x58    YM2610 port 0  0      0x4C
    0x59    YM2610 port 1  1      0x4C
    0xA2    YM2612 #2 p0   0      0x2C (bit 30 set = dual chip)
    0xA3    YM2612 #2 p1   1      0x2C (bit 30 set = dual chip)

### 7.2 Fantasy Hardware Examples

All playable in VGMPlay out of the box:

    "Super Genesis"  — YM2612 + 2nd YM2612 + SN76489
                       12 FM channels + 3 PSG = 15 voices

    "Genesis Neo"    — YM2612 + YM2610 + SN76489
                       6 FM (OPN2) + 4 FM (OPNB) + 3 PSG = 13 voices

    "Mega Arcade"    — YM2612 + YM2151 + SN76489
                       6 FM (OPN2) + 8 FM (OPM) + 3 PSG = 17 voices

    "All Yamaha"     — YM2612 + YM2413 + YM2151 + YM2610 + SN76489
                       6 + 9 + 8 + 4 FM + 3 PSG = 30 voices

    "Kitchen Sink"   — Every supported chip active simultaneously

### 7.3 Writer Constraints

To ensure VGMPlay compatibility:
  - VGM version must be >= 1.51 for YM2151, >= 1.61 for YM2203/YM2608/YM2610
  - Each chip's clock field must be non-zero in the header for VGMPlay to
    instantiate its emulator
  - Dual-chip flag: set bit 30 of the clock field (e.g., 0x2C for 2nd YM2612)
  - Data offset (0x34) must point past the header (>= 0xCC for extended fields)
  - All wait commands are in 44100 Hz samples regardless of chip clocks

---

## 8. Open Questions (renumbered)

1. **Should we fine-tune from v1 checkpoint or train from scratch?**
   (Note: with the expanded OPN-family chip vocabulary, training from scratch
   is even more strongly recommended — too many new token embeddings to init)
   - Fine-tuning is tricky because vocab changed (embedding matrix resized)
   - Could initialize shared event tokens from v1 weights, zero-init new ones
   - Recommendation: train from scratch — cleaner, and 80h is acceptable

2. **Metadata dropout for robustness?**
   - During training, randomly drop some metadata fields (e.g., 10% chance to
     omit composer, 5% chance to omit game)
   - Forces model to not over-rely on any single field
   - Allows generation with partial metadata at inference time
   - RECOMMENDED: yes, implement metadata dropout

3. **How to handle unknown composers at inference?**
   - Option A: Omit META_COMPOSER entirely (works with dropout training)
   - Option B: Use a special UNKNOWN_COMPOSER token
   - Recommendation: Option A (simpler, dropout handles it)

4. **Master System FM dataset size?** (ANSWERED)
   - Corpus breakdown (19,174 total files):
       Genesis (YM2612):     13,044 tracks
       SMS FM (YM2413):       1,228 tracks
       SMS PSG-only:            355 tracks
       Arcade YM2151:         1,667 tracks  (NEW — usable via VGMPlay)
       Neo Geo YM2610:          596 tracks  (NEW — OPN-family, closest to YM2612)
       YM2203-based:             472 tracks  (NEW — OPN ancestor)
       YM2608-based:              18 tracks  (NEW — OPN superset)
       OKIM6295-only:            419 tracks  (SKIP — ADPCM samples only)
       Other:                  ~1,375 tracks  (SKIP — PSG-only, OPL, NES, etc.)
   - OPN-family total: 596 + 472 + 18 = 1,086 new FM synthesis tracks
   - Combined v2 corpus: 13,044 + 1,228 + 1,086 = ~15,358 FM tracks
   - Consider oversampling SMS FM + OPN-family tracks 2-3x during dataset prep
   - Arcade YM2151 (1,667 tracks) is a stretch — different register layout
     from OPN family, but could be added as a v3 expansion

5. **PCM/DAC handling in fusion tracks?**
   - Genesis DAC (channel 6) has no equivalent on SMS
   - For fusion: keep include_dac=False (same as v1) for now
   - Future: dedicated DAC model or sample-bank conditioning
