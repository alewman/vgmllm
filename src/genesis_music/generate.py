"""Generate VGM music from a trained VgmGPT checkpoint.

Loads a trained model, generates token sequences via sampling,
decodes them back to VGM events, and writes playable .vgm files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .model import VgmGPT, ModelConfig
from .tokenizer import Vocab, decode_tokens, BOS, EOS
from .tokenizer_v2 import VocabV2, decode_ids_v2, encode_vgm_v2
from .tokenizer_v2 import BOS as BOS_V2, EOS as EOS_V2
from .tokenizer_v4 import PatchLibrary, TokenizerV4
from .tokenizer_v4 import BOS as BOS_V4, EOS as EOS_V4
from .vgm_parser import load_vgm
from .vgm_synth import synthesise_vgm
from .vgm_writer import save_vgm

log = logging.getLogger(__name__)


def load_model(
    checkpoint_path: Path | str,
    model_config_path: Path | str,
    device: str = "cuda",
) -> tuple[VgmGPT, torch.device]:
    """Load a trained VgmGPT from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_config_path: Path to model_config.json.
        device: Target device ('cuda' or 'cpu').

    Returns:
        (model, device) tuple.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model config
    cfg_dict = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
    cfg = ModelConfig(**cfg_dict)

    # Build model
    model = VgmGPT(cfg)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state_dict = ckpt["model"]
    # Handle torch.compile prefix
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(dev).eval()

    step = ckpt.get("step", "?")
    log.info("Loaded checkpoint (step %s) → %s", step, dev)

    return model, dev


def generate_vgm(
    model: VgmGPT,
    vocab: Vocab,
    device: torch.device,
    max_tokens: int = 8192,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    output_path: Path | str | None = None,
    prompt_tokens: list[int] | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
) -> Path | None:
    """Generate a VGM file from the model.

    Args:
        model: Trained VgmGPT model.
        vocab: Vocabulary for decoding.
        device: Torch device.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        output_path: Where to save the .vgm file. None = don't save.
        prompt_tokens: Optional list of token IDs to start with.
            Defaults to [BOS].
        repetition_penalty: Penalize repeated tokens (1.0 = off).
        repetition_window: Window of recent tokens for penalty.

    Returns:
        Path to saved file, or None if output_path is None.
    """
    # Build prompt
    if prompt_tokens is None:
        prompt_tokens = [BOS]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generate
    log.info(
        "Generating (max_tokens=%d, temp=%.2f, top_k=%d, top_p=%.2f, rep_pen=%.2f)...",
        max_tokens, temperature, top_k, top_p, repetition_penalty,
    )

    tokens = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token=EOS,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    token_list = tokens[0].tolist()
    log.info("Generated %d tokens", len(token_list))

    # Decode to events
    events = decode_tokens(token_list, vocab)
    log.info("Decoded to %d events", len(events))

    # Compute duration
    from .vgm_parser import EventType
    total_samples = sum(e.value for e in events if e.type == EventType.WAIT)
    duration = total_samples / 44100.0
    log.info("Duration: %.1f seconds", duration)

    # Convert to VGM and save
    if output_path is not None:
        output_path = Path(output_path)
        save_vgm(events, output_path)
        # Get file size
        file_size = output_path.stat().st_size
        log.info("Saved: %s (%.1f KB)", output_path, file_size / 1024)
        return output_path

    return None


def generate_vgm_v2(
    model: VgmGPT,
    vocab: VocabV2,
    device: torch.device,
    max_tokens: int = 8192,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    output_path: Path | str | None = None,
    prompt_tokens: list[int] | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
) -> Path | None:
    """Generate a VGM file using the v2 tokenizer."""
    if prompt_tokens is None:
        prompt_tokens = [BOS_V2]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    log.info(
        "Generating v2 (max_tokens=%d, temp=%.2f, top_k=%d, top_p=%.2f, "
        "rep_pen=%.2f, prompt_len=%d)...",
        max_tokens, temperature, top_k, top_p, repetition_penalty,
        len(prompt_tokens),
    )

    tokens = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token=EOS_V2,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    token_list = tokens[0].tolist()
    log.info("Generated %d tokens", len(token_list))

    # Decode v2 tokens to VGM events
    events = decode_ids_v2(token_list, vocab)
    log.info("Decoded to %d events", len(events))

    # Compute duration
    from .vgm_parser import EventType
    total_samples = sum(e.value for e in events if e.type == EventType.WAIT)
    duration = total_samples / 44100.0
    log.info("Duration: %.1f seconds", duration)

    if output_path is not None:
        output_path = Path(output_path)
        save_vgm(events, output_path)
        file_size = output_path.stat().st_size
        log.info("Saved: %s (%.1f KB)", output_path, file_size / 1024)
        return output_path

    return None


def generate_vgm_v4(
    model: VgmGPT,
    tokenizer: TokenizerV4,
    device: torch.device,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    output_path: Path | str | None = None,
    prompt_tokens: list[int] | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
    drum_kit: dict[int, bytes] | None = None,
) -> Path | None:
    """Generate a VGM file using the v4 tokenizer (NoteEvent-based)."""
    if prompt_tokens is None:
        prompt_tokens = [BOS_V4]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    log.info(
        "Generating v4 (max_tokens=%d, temp=%.2f, top_k=%d, top_p=%.2f, "
        "rep_pen=%.2f, prompt_len=%d)...",
        max_tokens, temperature, top_k, top_p, repetition_penalty,
        len(prompt_tokens),
    )

    tokens = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token=EOS_V4,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    token_list = tokens[0].tolist()
    log.info("Generated %d tokens", len(token_list))

    # Decode v4 tokens → NoteEvents + patch map
    note_events, patch_map = tokenizer.decode(token_list)
    log.info("Decoded to %d NoteEvents", len(note_events))

    # Estimate total playback length from note events
    if note_events:
        total_samples = max(
            (e.sample_off if e.sample_off >= 0 else e.sample_on)
            for e in note_events
        )
        total_samples = max(total_samples + 44100, 44100)  # at least 1s tail
    else:
        total_samples = 44100 * 10  # 10s silence fallback

    duration = total_samples / 44100.0
    log.info("Duration: %.1f seconds", duration)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vgm_bytes = synthesise_vgm(note_events, total_samples, patch_map,
                                    drum_kit=drum_kit)
        output_path.write_bytes(vgm_bytes)
        log.info("Saved: %s (%.1f KB)", output_path, len(vgm_bytes) / 1024)
        return output_path

    return None


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

    parser = argparse.ArgumentParser(description="Generate VGM music from trained model")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="Path to model_config.json (default: same dir as checkpoint)")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("output/generated.vgm"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Penalize repeated tokens (1.0=off, 1.2-1.5 recommended)")
    parser.add_argument("--repetition-window", type=int, default=64,
                        help="Recent token window for repetition penalty")
    parser.add_argument("--n", type=int, default=1, help="Number of files to generate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-version", choices=["v1", "v2", "v4"], default="v1",
                        help="Tokenizer version (v1=raw registers, v2=note-level, v4=new)")
    parser.add_argument("--patch-lib", type=Path, default=Path("data/patch_library_v4.json"),
                        help="Patch library for v4 tokenizer")
    parser.add_argument("--dac-slot-map", type=Path, default=None,
                        help="DAC slot map JSON for v4 tokenizer (defaults to "
                             "data/prepared_v4/dac_slot_map_v4.json if not set)")
    parser.add_argument("--prompt-vgm", type=Path, default=None,
                        help="VGM/VGZ file to use as prompt prefix")
    parser.add_argument("--prompt-tokens", type=int, default=512,
                        help="Number of tokens from prompt-vgm to use as prefix")

    args = parser.parse_args()

    # Resolve model config
    if args.model_config is None:
        args.model_config = args.checkpoint.parent / "model_config.json"

    # Load model and vocab / tokenizer
    model, device = load_model(args.checkpoint, args.model_config, args.device)

    is_v4 = args.vocab_version == "v4"
    is_v2 = args.vocab_version == "v2"

    if is_v4:
        patch_lib = PatchLibrary.load(args.patch_lib)
        dac_slot_map_path = args.dac_slot_map or Path("data/prepared_v4/dac_slot_map_v4.json")
        dac_slot_map: dict[int, int] = {}
        if Path(dac_slot_map_path).exists():
            import json as _json
            raw = _json.loads(Path(dac_slot_map_path).read_text())
            dac_slot_map = {int(k): int(v) for k, v in raw.items()}
        tokenizer_v4 = TokenizerV4(patch_lib, dac_slot_map=dac_slot_map)

        # Load drum kit: slot → raw PCM bytes (for VGM synthesis)
        drum_kit: dict[int, bytes] | None = None
        drum_kit_path = Path(dac_slot_map_path).parent / "dac_drum_kit_v4.json"
        if drum_kit_path.exists():
            import json as _json2
            raw_kit = _json2.loads(drum_kit_path.read_text())
            drum_kit = {int(k): bytes.fromhex(v) for k, v in raw_kit.items()}
            log.info("Loaded drum kit: %d slots", len(drum_kit))
    elif is_v2:
        vocab = VocabV2.load(args.vocab)
    else:
        vocab = Vocab.load(args.vocab)

    # Build prompt from VGM file if requested
    prompt_ids = None
    if args.prompt_vgm is not None:
        vgm = load_vgm(args.prompt_vgm)
        if is_v4:
            all_ids = tokenizer_v4.encode(vgm) or [BOS_V4]
        elif is_v2:
            all_ids = encode_vgm_v2(vgm, vocab)
        else:
            from .tokenizer import encode_vgm
            all_ids = encode_vgm(vgm, vocab)
        prompt_ids = all_ids[: args.prompt_tokens]
        log.info(
            "Prompt: %s → %d tokens (of %d total)",
            args.prompt_vgm.name, len(prompt_ids), len(all_ids),
        )

    # Generate
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        if args.n > 1:
            stem = args.output.stem
            out = output_dir / f"{stem}_{i+1:03d}.vgm"
        else:
            out = args.output

        if is_v4:
            generate_vgm_v4(
                model=model, tokenizer=tokenizer_v4, device=device,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                output_path=out, prompt_tokens=prompt_ids,
                drum_kit=drum_kit,
            )
        else:
            gen_fn = generate_vgm_v2 if is_v2 else generate_vgm
            gen_fn(
                model=model, vocab=vocab, device=device,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                output_path=out, prompt_tokens=prompt_ids,
            )


if __name__ == "__main__":
    main()
