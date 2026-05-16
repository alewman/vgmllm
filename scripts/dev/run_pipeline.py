"""Automated pipeline: wait for vocab build, prepare dataset, start training.

Run this after starting the vocab build in another terminal.
It will poll for vocab.json to update, then chain dataset prep and training.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
VOCAB = ROOT / "data" / "vocab.json"
VGM_DIR = ROOT / "data" / "vgm"
PREPARED_DIR = ROOT / "data" / "prepared"
OUTPUT_DIR = ROOT / "runs" / "v1"

# Record the initial state of vocab.json
initial_mtime = VOCAB.stat().st_mtime if VOCAB.exists() else 0
initial_size = VOCAB.stat().st_size if VOCAB.exists() else 0


def wait_for_vocab():
    """Poll until vocab.json is updated (different mtime or size)."""
    print(f"Waiting for vocab.json to update (current: {initial_size} bytes, mtime={initial_mtime:.0f})...")
    while True:
        if VOCAB.exists():
            st = VOCAB.stat()
            if st.st_mtime != initial_mtime or st.st_size != initial_size:
                print(f"vocab.json updated! New size: {st.st_size} bytes")
                return
        time.sleep(10)


def run_step(desc: str, cmd: list[str]):
    """Run a command, streaming output, and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"FAILED: {desc} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\nDONE: {desc}\n")


def main():
    # Step 1: Wait for vocab build to finish
    wait_for_vocab()

    # Step 2: Inspect vocab
    run_step(
        "Inspect vocabulary",
        [sys.executable, "-m", "genesis_music.tokenizer", "inspect",
         "--vocab", str(VOCAB)],
    )

    # Step 3: Prepare dataset
    run_step(
        "Prepare dataset (tokenize all VGMs → .npy)",
        [sys.executable, "-m", "genesis_music.dataset", "prepare",
         "--vgm-dir", str(VGM_DIR),
         "--vocab", str(VOCAB),
         "--output", str(PREPARED_DIR),
         "--seq-len", "4096"],
    )

    # Step 4: Start training
    run_step(
        "Train model (medium, 50K steps)",
        [sys.executable, "-m", "genesis_music.train",
         "--data-dir", str(PREPARED_DIR),
         "--output-dir", str(OUTPUT_DIR),
         "--model-size", "medium",
         "--seq-len", "4096",
         "--batch-size", "4",
         "--grad-accum", "8",
         "--max-steps", "50000",
         "--no-compile"],
    )

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
