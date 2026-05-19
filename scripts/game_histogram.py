"""Quick histogram of VGM corpus: games by track count, to tune NUM_GAMES."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from genesis_music.tokenizer_v6 import _fast_read_gd3_fields, _normalize_game_name

vgm_dir = Path(__file__).parent.parent / "data" / "vgm"
files = sorted(list(vgm_dir.rglob("*.vgm")) + list(vgm_dir.rglob("*.vgz")))
print(f"Files: {len(files)}", flush=True)

counts: Counter = Counter()
for i, p in enumerate(files, 1):
    try:
        fields = _fast_read_gd3_fields(p, 2)
        name = _normalize_game_name(fields[0]) if fields else ""
        counts[name if name else "__unknown__"] += 1
    except Exception:
        counts["__error__"] += 1
    if i % 5000 == 0:
        print(f"  {i}/{len(files)}…", flush=True)

sorted_games = counts.most_common()
total = sum(counts.values())
print(f"\nUnique games: {len(counts)}\n")

print("Coverage at each N:")
for n in [20, 30, 40, 50, 64, 80, 96, 128]:
    if n > len(sorted_games):
        break
    c2 = sum(v for _, v in sorted_games[:n])
    last_n = sorted_games[n - 1][1]
    print(f"  Top {n:3d}: min_tracks={last_n:3d}  cumulative={c2}/{total} ({100*c2/total:.1f}%)")

print("\nTrack count distribution (how many games have <= N tracks):")
for b in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
    n_g = sum(1 for _, c in counts.items() if c <= b)
    print(f"  <= {b:2d} tracks: {n_g:4d} games")

print("\nElbow zone (ranks 55-90):")
for rank, (name, n) in enumerate(sorted_games, 1):
    if 55 <= rank <= 90:
        print(f"  {rank:3d}. {name[:50]:<50} {n:3d}")
