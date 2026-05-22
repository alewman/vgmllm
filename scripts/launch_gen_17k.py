"""Launch gen_v6_ken_17k.sh as a fully detached subprocess."""
import subprocess
from pathlib import Path

base = Path(__file__).parents[1]
script   = str(base / "scripts" / "gen_v6_ken_17k.sh")
log_path = str(base / "logs"    / "gen_v6_ken_17k.log")

(base / "logs").mkdir(exist_ok=True)

with open(log_path, "w") as log:
    p = subprocess.Popen(
        ["bash", script],
        stdout=log, stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
print(f"Launched PID {p.pid} → {log_path}")
