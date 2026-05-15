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
