import json
from pathlib import Path

base = Path(__file__).parents[2]
d = json.loads((base / "data/prepared_v6/game_map_v6.json").read_text())
games = d["games"]
print(f"Total games in map: {len(games)}")
print("\nTop 64 (conditioned games):")
for i, g in enumerate(games[:64]):
    print(f"  token {660+i}: {g}")
