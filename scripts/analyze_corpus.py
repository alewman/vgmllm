"""Fast VGM corpus analysis for SFX detection and duration distribution.

Uses VGM header for duration (no event iteration needed) and scans raw bytes
for Key On events (0x52 0x28 XX with XX having upper nibble != 0).
"""
import sys
import gzip
import json
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _read_u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return struct.unpack_from("<I", data, offset)[0]


def analyze_file_fast(filepath: Path) -> tuple[str, float, bool] | None:
    """Fast analysis: read header for duration, scan bytes for FM key-on.
    
    Returns (filepath_str, duration_seconds, has_fm_key_on) or None on error.
    """
    try:
        raw = filepath.read_bytes()
        # Decompress VGZ
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                # Double-gzipped
                try:
                    raw = gzip.decompress(gzip.decompress(raw))
                except Exception:
                    return None

        # Check VGM magic
        if raw[:4] != b"Vgm ":
            return None

        # Get duration from header (total_samples at offset 0x18)
        total_samples = _read_u32(raw, 0x18)
        dur_s = total_samples / 44100.0

        # Get data offset
        data_offset_rel = _read_u32(raw, 0x34)
        if data_offset_rel == 0:
            data_start = 0x40  # default for older versions
        else:
            data_start = 0x34 + data_offset_rel

        # Scan for FM Key On: command 0x52 (YM2612 port 0), register 0x28
        # with upper nibble of value != 0 (meaning operators are keyed on)
        has_fm_keyon = False
        i = data_start
        data_len = len(raw)
        while i < data_len:
            cmd = raw[i]
            if cmd == 0x52 and i + 2 < data_len:
                # YM2612 port 0 write
                reg = raw[i + 1]
                val = raw[i + 2]
                if reg == 0x28 and (val & 0xF0) != 0:
                    has_fm_keyon = True
                    break
                i += 3
            elif cmd == 0x53 and i + 2 < data_len:
                i += 3  # YM2612 port 1
            elif cmd == 0x50 and i + 1 < data_len:
                i += 2  # SN76489
            elif cmd == 0x61 and i + 2 < data_len:
                i += 3  # Wait N samples
            elif cmd == 0x62:
                i += 1  # Wait 735 samples
            elif cmd == 0x63:
                i += 1  # Wait 882 samples
            elif cmd == 0x66:
                break  # End of data
            elif 0x70 <= cmd <= 0x7F:
                i += 1  # Short wait
            elif 0x80 <= cmd <= 0x8F:
                i += 1  # YM2612 DAC + wait
            elif cmd == 0x67 and i + 6 < data_len:
                # Data block - skip
                block_size = _read_u32(raw, i + 3)
                i += 7 + block_size
            elif cmd == 0xE0 and i + 4 < data_len:
                i += 5  # PCM offset
            elif cmd == 0x51 and i + 2 < data_len:
                i += 3  # YM2413
            elif cmd == 0x4F and i + 1 < data_len:
                i += 2  # SN76489 stereo
            else:
                i += 1  # Unknown, skip byte

        return (str(filepath), dur_s, has_fm_keyon)
    except Exception:
        return None


def main():
    import numpy as np

    out_path = Path("data/corpus_report2.txt")
    log = open(out_path, "w", encoding="utf-8")

    def log_print(msg=""):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    vgm_dir = Path("data/vgm")
    files = sorted(list(vgm_dir.glob("*.vgm")) + list(vgm_dir.glob("*.vgz")))
    log_print(f"Total files: {len(files)}")

    results = []
    errors = 0
    for i, f in enumerate(files):
        r = analyze_file_fast(f)
        if r is None:
            errors += 1
        else:
            results.append(r)
        if (i + 1) % 1000 == 0:
            log_print(f"  {i + 1}/{len(files)}...")

    durations = np.array([r[1] for r in results])
    has_notes = np.array([r[2] for r in results])

    log_print(f"\nParsed: {len(durations)}, Errors: {errors}")
    log_print(f"Duration: min={durations.min():.1f}s, median={np.median(durations):.1f}s, "
          f"mean={durations.mean():.1f}s, max={durations.max():.1f}s")
    log_print(f"\n--- Duration distribution ---")
    log_print(f"Under 3s:   {(durations < 3).sum()}")
    log_print(f"Under 5s:   {(durations < 5).sum()}")
    log_print(f"5-10s:      {((durations >= 5) & (durations < 10)).sum()}")
    log_print(f"10-30s:     {((durations >= 10) & (durations < 30)).sum()}")
    log_print(f"30-60s:     {((durations >= 30) & (durations < 60)).sum()}")
    log_print(f"60-120s:    {((durations >= 60) & (durations < 120)).sum()}")
    log_print(f"Over 120s:  {(durations >= 120).sum()}")
    log_print(f"\n--- FM note presence ---")
    log_print(f"Has FM key-on: {has_notes.sum()} ({has_notes.mean()*100:.1f}%)")
    log_print(f"No FM notes:   {(~has_notes).sum()}")
    sfx = (~has_notes) | (durations < 5)
    log_print(f"\n--- SFX classification ---")
    log_print(f"Likely SFX (no FM notes or under 5s): {sfx.sum()}")
    log_print(f"Clean music (has FM notes + >=5s):    {(~sfx).sum()}")

    # Save detailed results for later use
    detail = {r[0]: {"duration": r[1], "has_fm_notes": r[2]} for r in results}
    json_path = Path("data/corpus_analysis.json")
    with open(json_path, "w") as f:
        json.dump(detail, f)
    log_print(f"\nDetailed results saved to {json_path}")
    log.close()


if __name__ == "__main__":
    main()
