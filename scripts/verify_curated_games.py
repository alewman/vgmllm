"""Verify curated_games_v7.json against the actual corpus GD3 game_name_en tags.

For each game in the curated list, shows:
  - How many VGM files matched (by normalized GD3 tag)
  - The actual GD3 strings that matched (to check for variations)
  - Games with ZERO matches (need to fix gd3_name_hint)

Also shows top-20 unmatched corpus games (in case we want to swap anything in).

Usage:
    python scripts/verify_curated_games.py --vgm-dir data/vgm
    python scripts/verify_curated_games.py --vgm-dir data/vgm --json data/curated_games_v7.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Add parent src to path for direct script execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis_music.tokenizer_v6 import _normalize_game_name, _fast_read_gd3_fields


def main():
    parser = argparse.ArgumentParser(description="Verify curated game list against corpus GD3 tags")
    parser.add_argument("--vgm-dir", required=True, type=Path, help="Directory of VGM files")
    parser.add_argument("--json", type=Path, default=Path("data/curated_games_v7.json"),
                        help="Path to curated_games_v7.json")
    parser.add_argument("--top-unmatched", type=int, default=25,
                        help="Show this many top unmatched corpus games")
    args = parser.parse_args()

    # Load curated list
    data = json.loads(args.json.read_text())
    driver_groups = data["driver_groups"]

    # Build curated hint → (short_name, driver_name) map
    hint_to_meta: dict[str, tuple[str, str]] = {}
    all_hints_normalized: dict[str, str] = {}  # normalized → original hint
    for grp in driver_groups:
        for g in grp["games"]:
            hint = g["gd3_name_hint"]
            norm = _normalize_game_name(hint)
            hint_to_meta[norm] = (g["name"], grp["driver_name"])
            all_hints_normalized[norm] = hint

    print(f"Loaded {len(hint_to_meta)} curated game hints from {args.json}")

    # Scan corpus
    vgm_files = sorted(
        list(args.vgm_dir.rglob("*.vgm")) + list(args.vgm_dir.rglob("*.vgz"))
    )
    print(f"Scanning {len(vgm_files)} VGM files…\n")

    # For each VGM: get raw GD3 game name
    corpus_raw_counter:  Counter[str] = Counter()  # raw name → track count
    curated_raw_matches: dict[str, list[str]] = defaultdict(list)  # norm_hint → [raw names seen]
    curated_count:       Counter[str] = Counter()   # norm_hint → file count

    for i, path in enumerate(vgm_files):
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(vgm_files)} files scanned…")
        try:
            fields = _fast_read_gd3_fields(Path(str(path)), 2)
            raw = fields[0].strip() if fields else ""
        except Exception:
            raw = ""
        if not raw:
            continue

        corpus_raw_counter[raw] += 1
        norm = _normalize_game_name(raw)
        if norm in hint_to_meta:
            curated_count[norm] += 1
            if raw not in curated_raw_matches[norm]:
                curated_raw_matches[norm].append(raw)

    # -----------------------------------------------------------------------
    # Report: curated games with matches
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("CURATED GAMES — CORPUS MATCH REPORT")
    print("=" * 70)

    matched   = []
    unmatched = []

    for grp in driver_groups:
        for g in grp["games"]:
            hint = g["gd3_name_hint"]
            norm = _normalize_game_name(hint)
            count = curated_count.get(norm, 0)
            raw_seen = curated_raw_matches.get(norm, [])
            entry = {
                "name": g["name"],
                "driver": grp["driver_name"],
                "hint": hint,
                "norm": norm,
                "count": count,
                "raw_seen": raw_seen,
            }
            if count > 0:
                matched.append(entry)
            else:
                unmatched.append(entry)

    print(f"\n[OK] MATCHED ({len(matched)}/{len(matched)+len(unmatched)}):\n")
    for e in sorted(matched, key=lambda x: -x["count"]):
        raw_str = ", ".join(f'"{r}"' for r in e["raw_seen"][:3])
        print(f"  [{e['count']:4d} tracks]  {e['name']}")
        print(f"              hint: {e['hint']!r}")
        print(f"              GD3 matches: {raw_str}")
        print()

    if unmatched:
        print(f"\n[MISS] NO MATCH ({len(unmatched)}) - need to fix gd3_name_hint:\n")
        for e in unmatched:
            print(f"  {e['name']}")
            print(f"    driver: {e['driver']}")
            print(f"    hint:   {e['hint']!r}  (normalized: {e['norm']!r})")
            print()
    else:
        print("\n[PERFECT] All curated games matched at least one corpus file.\n")

    # -----------------------------------------------------------------------
    # Report: top unmatched corpus games
    # -----------------------------------------------------------------------
    matched_norms = set(curated_count.keys())
    print("=" * 70)
    print(f"TOP {args.top_unmatched} CORPUS GAMES NOT IN CURATED LIST:")
    print("=" * 70)
    print()
    unmatched_corpus = [
        (raw, cnt)
        for raw, cnt in corpus_raw_counter.most_common()
        if _normalize_game_name(raw) not in hint_to_meta
    ]
    for raw, cnt in unmatched_corpus[:args.top_unmatched]:
        norm = _normalize_game_name(raw)
        print(f"  [{cnt:4d} tracks]  {raw!r}  (normalized: {norm!r})")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_corpus_tracks = sum(corpus_raw_counter.values())
    curated_tracks = sum(curated_count.values())
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total corpus tracks:          {total_corpus_tracks:6d}")
    print(f"  Curated games matched tracks: {curated_tracks:6d}  ({100*curated_tracks/max(1,total_corpus_tracks):.1f}%)")
    print(f"  Curated games matched:        {len(matched):6d} / {len(matched)+len(unmatched)}")
    if unmatched:
        print(f"\n  [!] Fix {len(unmatched)} unmatched hint(s) above, then re-run.")
    print()


if __name__ == "__main__":
    main()
