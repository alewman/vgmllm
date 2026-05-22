# Genesis Music ML — Roadmap to Production-Quality AI Music Generation

## Stage 1: Representation Design ← v2 focus
Raw audio bytes or raw register dumps don't encode musical structure. Tokenization must make musical primitives atomic:
- **Pitch as named notes** (C4, D#5) not frequency register pairs
- **Duration as musical time** (quarter note, eighth note) not sample counts
- **Velocity/dynamics** as discrete levels
- **Instrument/timbre** as patch IDs or learned embeddings
- **Positional structure** — bar lines, beat positions, time signatures baked into the token stream
- Representation must be **losslessly invertible** back to audio
- For chip music: patch programming (FM algorithm, operator settings) as structured header, not interleaved with notes

**Status**: v1 used raw register writes. v2 design in V2_DESIGN.md targets note-level tokenization.

## Stage 2: Data Pipeline & Corpus Quality
- **Transcription/alignment** — raw VGM → structured note events, quantized to a musical grid
- **Metadata extraction** — key signature, tempo, time signature, genre, game title, composer
- **Filtering** — remove SFX, jingles, broken files, duplicates, sub-10s tracks
- **Augmentation** — transpose to all 12 keys, tempo stretch, pitch shift (prevents key/tempo overfitting, massively increases effective dataset size)
- **Balancing** — ensure representation across genres/games/compositional styles

**Status**: v1 has basic pipeline (19K VGM files, parallel tokenization). Needs filtering and augmentation.

## Stage 3: Architecture
- **Context length** — a 3-minute piece is 10K-50K tokens. Need efficient attention (sliding window, sparse, state-space hybrids like Mamba) for full song context
- **Hierarchical modeling** — note-level, bar-level, section-level. Flat autoregressive struggles with AABA structure over thousands of tokens. Approaches: hierarchical transformers, latent plan → detail (MusicLM-style), or explicit section-level planning tokens
- **Multi-track awareness** — interleave tracks with track-ID tokens, parallel decoding heads per voice, or piano-roll 2D structure

**Status**: v1 is 137.9M GPT-style, 4096 seq_len. Works for texture, insufficient for song structure.

## Stage 4: Training Regime
- **Pre-training** on full corpus (unconditional next-token prediction) — done in v1
- **Curriculum learning** — start with short sequences (single phrases), progressively increase to full songs. Critical for learning structure at multiple scales
- **Objective design** — beyond cross-entropy: contrastive losses for timbral consistency, auxiliary losses for predicting bar boundaries or chord roots, RL from music-theoretic reward models
- **Scale** — 100M params learns texture. For consistent musical structure, likely need 500M-1B+ with proportionally more data and compute

**Status**: v1 trained 50K steps, val_loss=0.1963. Learned texture well, not structure.

## Stage 5: Conditioning & Controllability
This is where "matching user prompts" comes in:
- **Text conditioning** — CLAP-style text-audio embedding alignment, or cross-attention from text encoder to music decoder
- **Style conditioning** — genre embeddings, composer embeddings, game/era tags (learned from metadata or contrastive pre-training)
- **Musical conditioning** — key, tempo, time signature, chord progression as explicit control tokens
- **Reference conditioning** — "sound like this track" via audio embeddings (timbre transfer)
- **Classifier-free guidance** — train with conditioning randomly dropped; at inference, interpolate between conditional and unconditional to control adherence strength

**Status**: Not started. v1 prompting experiment showed prefix conditioning works for timbre/channel usage.

## Stage 6: Long-Range Coherence
The hardest problem — separates "sounds nice for 10 seconds" from "this is a song":
- **Form/structure planning** — generate high-level plan (intro-verse-chorus-verse-chorus-bridge-chorus-outro) before notes. Explicit tokens or latent planning stage
- **Repetition with variation** — model must repeat earlier phrases with modifications. Requires very long effective context or explicit memory/retrieval
- **Harmonic arc** — tension and release over 2-4 minute timescales. Cadences, modulations, dramatic buildups
- **Self-similarity constraints** — losses or sampling strategies encouraging structural self-similarity at bar, phrase, and section scales

**Status**: Not addressed. v1 produces texture without structure.

## Stage 7: Evaluation & Iteration
- **Automated metrics** — Fréchet Audio Distance (FAD), pitch class distribution matching, rhythmic consistency, structural self-similarity matrices, harmonic analysis
- **Music-theoretic analysis** — automated key detection, chord progression extraction, phrase boundary detection on generated output
- **Human evaluation** — A/B tests, MOS (Mean Opinion Score) on musicality, stylistic accuracy, enjoyment
- **Reward modeling / RLHF** — train preference model from human ratings, fine-tune via PPO or DPO. This goes from "usually decent" to "consistently good"

**Status**: v1 evaluation is manual listening only.

## Stage 8: Inference & Post-Processing
- **Constrained decoding** — key-aware masking (only allow notes in current key/scale), rhythmic quantization constraints, voice-leading rules as soft/hard logit masks
- **Rejection sampling / best-of-N** — generate multiple candidates, score with reward model, keep best
- **Post-processing** — velocity smoothing, humanization (micro-timing), mixing/panning normalization
- **For chip music** — convert structured intermediate representation back to valid register writes, playable on real hardware or emulators

**Status**: v1 has basic repetition penalty and temperature/top-k/top-p sampling. No music-aware constraints.

---

## Progress Summary

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Representation Design | v1: raw registers. v2: note-level (planned) |
| 2 | Data Pipeline & Corpus | v1: basic (19K files). Needs filtering + augmentation |
| 3 | Architecture | v1: 137.9M GPT, 4096 ctx. Needs scaling + efficient attention |
| 4 | Training Regime | v1: 50K steps pre-training. Needs curriculum learning |
| 5 | Conditioning | Not started. Prefix prompting proof of concept done |
| 6 | Long-Range Coherence | Not started |
| 7 | Evaluation | Manual listening only |
| 8 | Inference Constraints | Basic sampling only |

## Key v1 Findings
- Model excels at FM synthesis timbres — authentic Genesis sounds
- Proper note lifecycle (Key On/Off)
- Good stereo panning
- Some pitch proximity (scale-like melodies)
- Only uses 1-2 of 6 FM channels without prompting
- Prefix prompting dramatically improves channel usage and timbre richness
- Repetition penalty rp=1.2/w64 is the sweet spot
- Dense music (TFIV: 0.5ms/tok) vs sparse (Sonic: 3.3ms/tok) affects token budget significantly
- Quality degrades beyond training context length (4096 tokens)

---

## v6 — Current Run (as of May 2026)

**Model**: 16 layers, 12 heads, d=768, 113.8M params, bfloat16, seq_len=16384  
**Run**: `v6_medium` — 50,000 steps, batch=16 (4×4 grad accum), lr=3e-4  
**Hardware**: RTX 3090, 25.8GB VRAM, WSL, ~13s/step  
**Tokenizer**: v6 — 794 tokens, lossless FM patch encoding (40 params per FM channel in header), 64 conditioned games (GAME_BASE 660–723)

**Val loss progression**: 250→2.07, 500→1.43, 1000→1.04, 2000→0.62, 4000→0.40, 6000→0.31, 7500→0.299, 7750→0.291 (ppl≈1.3)  
**Training loss at step ~11k**: ~0.11–0.13, grad_norm ~0.09–0.10 (very stable)  
**ETA to 50k**: ~6 more days from step 11k

### Key v6 Findings (step 11k — early, still training)
- **Best model yet by a wide margin** — already surpasses previous runs at 45k steps
- Confidently uses up to 7 instruments simultaneously (FM×6 + PSG/DAC)
- Instruments sound authentic — proper FM timbres carrying through
- Mildly coordinated multi-channel writing — channels are aware of each other
- Still "crawling" — melodic/harmonic development is slow, structure is not there yet
- Working through scales rather than assertive melodic statements
- Game conditioning works: SFII Ken's Stage prompt produces SFII-flavored output
- Lossless FM patch encoding (v6 key feature) is clearly helping timbral consistency

### Known v6 Limitations (targets for v7)
1. **Static channel role labeling** — roles (BASS, LEAD, HARM, COUNTER, DRUMS, PERC, UNK) are assigned *once per track* at encode time using whole-track mean pitch + note density. If a channel modulates from bass to lead mid-song, it gets a single stale ROLE token for the entire sequence. The model cannot represent or learn instrument role changes.
2. **Konami sound driver underrepresentation** — the 64 conditioned games are simply the top 64 by raw track count in the VGM corpus. Konami custom driver games (Castlevania: Bloodlines, Contra: Hard Corps, Rocket Knight Adventures, Sparkster, TMNT: Hyperstone Heist) are *in* the corpus but contributed only ~20–30 tracks each — far below the RPG-heavy titles dominating the top 64. Only Animaniacs (a licensed game) represents Konami in the conditioned set. This means the model has seen Konami-driver VGMs but can't be conditioned on them; they dilute the UNK_GAME bucket.
3. **No dynamic structure** — same as previous versions: no verse/chorus/bridge awareness, no long-range repetition with variation. Song form is not modeled.
4. **Roles computed from notes, not register writes** — DAC sample instrument changes mid-track are invisible to the role classifier since it operates on decoded NoteEvents, not raw register state.

---

## v7 — Planned Improvements

### Priority 1: Dynamic Channel Role Conditioning
Instead of a single ROLE token per channel in the track header, emit ROLE re-labeling tokens at segment boundaries (e.g., every N bars or at detected instrument change events). This allows the model to learn that a channel's musical function can shift mid-track, which is extremely common in Genesis music (bass doubles as a pad, lead trades with counter-melody, etc.).

Implementation options:
- **Segment-level ROLE tokens**: divide track into fixed segments (e.g., 4-bar chunks), re-emit `CH_n ROLE_x` at each segment boundary — simple, backward-compatible with v6 tokenizer structure
- **Change-detect ROLE tokens**: only emit a new ROLE token when `classify_channel_roles()` over a sliding window disagrees with the current label — more compact but requires windowed analysis in the encoder
- Either approach requires updating `tokenizer_v6.py` encode() and decode() and retraining from scratch

### Priority 2: Driver-Aware Corpus Curation
Replace raw track-count ranking with **stratified sampling by sound driver family**. Major Genesis driver families to ensure representation:
- **SMPS** (Sonic/Sega in-house) — already dominant in corpus
- **Konami custom** — Castlevania: Bloodlines, Contra: Hard Corps, Rocket Knight Adventures, Sparkster, TMNT: Hyperstone Heist
- **GEMS** (Electronic Arts) — NBA Live, FIFA series, many EA Sports titles
- **Squaresoft custom** — Final Fantasy ports if any exist
- **Capcom custom** — Street Fighter II variants (already represented), Mega Man: Wily Wars
- **Other in-house** — Treasure (Gunstar Heroes, Dynamite Headdy already in set), Technosoft (TFIV already in set)

Target: ~8–12 driver families × ~8 representative games each = ~64–96 conditioned games, balanced by driver rather than raw count.

### Priority 3: Structural Tokens / Song Form
Add explicit section boundary tokens (INTRO, VERSE, CHORUS, BRIDGE, OUTRO) either:
- Detected via self-similarity analysis (repeated phrase detection) at encode time
- Or as a learned planning stage (two-pass: first generate section plan, then generate notes conditioned on plan)

This is the hardest problem and may require a larger model (≥300M params) and longer sequences.

### Lower Priority
- **Evaluation metrics**: FAD, pitch class entropy, structural self-similarity matrix, automated harmonic analysis — move beyond manual listening
- **Key-aware constrained decoding**: mask logits to only permit in-key pitches at inference; softened version via logit bias rather than hard mask
- **Tempo/groove conditioning**: separate groove templates (swing, straight, shuffle) as explicit conditioning tokens beyond just BPM

---

## Updated Progress Summary (May 2026)

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Representation Design | **v6: lossless FM encoding, 7 channel roles, game/composer conditioning** |
| 2 | Data Pipeline & Corpus | v6: 64 conditioned games (track-count ranked). **v7 target: driver-stratified** |
| 3 | Architecture | v6: 113.8M GPT, seq_len=16384. Sufficient for current quality level |
| 4 | Training Regime | v6: 50K steps, stable loss ~0.11 at step 11k. No curriculum yet |
| 5 | Conditioning | v6: game, composer, key, tempo, meter, channel roles in header |
| 6 | Long-Range Coherence | **Not addressed.** v7 target: structural tokens |
| 7 | Evaluation | Manual listening only |
| 8 | Inference Constraints | Basic sampling (temp, top-k, top-p, rep-pen). No music-aware constraints |
