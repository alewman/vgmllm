"""Quick helper: search game_map_v6 for a keyword and show tokenizer prompts."""
import json, sys
from pathlib import Path

keyword = sys.argv[1].lower() if len(sys.argv) > 1 else "column"
base = Path(__file__).parents[2]

gmap = json.loads((base / "data/prepared_v6/game_map_v6.json").read_text())
games = gmap.get("games", gmap) if isinstance(gmap, dict) else gmap
if isinstance(games, list):
    matches = [g for g in games if keyword in g.lower()]
    print("Games matching:", matches)
elif isinstance(games, dict):
    matches = {k: v for k, v in games.items() if keyword in k.lower()}
    print("Games matching:", json.dumps(matches, indent=2))

# Also check tokenizer v6 for game tokens
try:
    tok_path = base / "data/prepared_v6/meta.json"
    meta = json.loads(tok_path.read_text())
    print("\nmeta keys:", list(meta.keys())[:10])
except Exception as e:
    print("meta.json error:", e)
