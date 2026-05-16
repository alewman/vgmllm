import time
import traceback
from pathlib import Path

try:
    from genesis_music.vgm_parser import load_vgm, EventType

    files = sorted(list(Path('data/vgm').glob('*.vgm')) + list(Path('data/vgm').glob('*.vgz')))[:100]
    t0 = time.time()
    for f in files:
        vgm = load_vgm(f)
        dur = sum(e.value for e in vgm.events if e.type == EventType.WAIT) / 44100.0
    elapsed = time.time() - t0
    Path('data/bench.txt').write_text(f'{elapsed:.1f}s for 100 files, est {elapsed * 191.74:.0f}s total\n')
except Exception:
    Path('data/bench.txt').write_text(traceback.format_exc())
