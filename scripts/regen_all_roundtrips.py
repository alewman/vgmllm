"""Batch-regenerate roundtrip VGMs for all original test songs.

Finds every *_original.vgz and standalone .vgz in output/roundtrip/,
strips '_original' from the stem to produce the slug, and calls
regen_song.run() for each.

Usage:
    python scripts/regen_all_roundtrips.py [--dry-run]
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from pathlib import Path
import importlib.util, types

# ── load regen_song without executing its __main__ block ──────────────────────
_spec = importlib.util.spec_from_file_location(
    "regen_song",
    Path(__file__).parent / "regen_song.py",
)
regen_song: types.ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regen_song)

# ── collect originals ─────────────────────────────────────────────────────────
ROUNDTRIP_DIR = Path("output/roundtrip")

# Explicit standalone .vgz files that don't carry the _original suffix
STANDALONE = [
    "IceCap_Zone_Act_1.vgz",
    "My_Lover.vgz",
    "Alien_Power.vgz",
    "Truth_Haides.vgz",
]

candidates: list[tuple[Path, str]] = []

# 1. All *_original.vgz files
for p in sorted(ROUNDTRIP_DIR.glob("*_original.vgz")):
    slug = p.stem.replace("_original", "")
    candidates.append((p, slug))

# 2. Standalone .vgz files listed above
for name in STANDALONE:
    p = ROUNDTRIP_DIR / name
    if p.exists():
        slug = p.stem          # e.g. IceCap_Zone_Act_1
        candidates.append((p, slug))

# 3. go_straight original .vgm
go_orig = ROUNDTRIP_DIR / "go_straight_ORIGINAL.vgm"
if go_orig.exists():
    candidates.append((go_orig, "Go_Straight"))

dry_run = "--dry-run" in sys.argv

print(f"Found {len(candidates)} original(s) to regenerate:\n")
for p, slug in candidates:
    print(f"  {slug:55s}  <- {p.name}")

if dry_run:
    print("\n[dry-run] No files written.")
    sys.exit(0)

print()
errors: list[tuple[str, Exception]] = []
for i, (p, slug) in enumerate(candidates, 1):
    print(f"[{i}/{len(candidates)}] {slug}")
    try:
        regen_song.run(p, slug)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        errors.append((slug, exc))
    print()

print("=" * 60)
if errors:
    print(f"DONE with {len(errors)} error(s):")
    for slug, exc in errors:
        print(f"  {slug}: {exc}")
else:
    print(f"DONE — all {len(candidates)} roundtrips regenerated successfully.")
