"""Survey GD3 tags across the full VGM corpus to find top composers and games.

Uses a fast GD3-only reader (no full event parsing) to scan all 19k files quickly.
"""
import gzip
import re
import struct
from pathlib import Path
from collections import Counter


def _read_gd3_string(data: bytes, pos: int) -> tuple[str, int]:
    """Read a null-terminated UTF-16LE string, return (text, new_pos)."""
    chars = []
    while pos + 1 < len(data):
        cp = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        if cp == 0:
            break
        chars.append(chr(cp))
    return ''.join(chars), pos


def fast_read_gd3(path: Path) -> tuple[str, str] | None:
    """Read only the GD3 tag from a .vgm or .vgz file.
    Returns (author_en, game_name_en) or None."""
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == '.vgz':
            raw = gzip.decompress(raw)
        if len(raw) < 0x20 or raw[:4] != b'Vgm ':
            return None
        gd3_rel = struct.unpack_from('<I', raw, 0x14)[0]
        if gd3_rel == 0:
            return None
        gd3_abs = gd3_rel + 0x14
        if gd3_abs + 12 > len(raw) or raw[gd3_abs:gd3_abs+4] != b'Gd3 ':
            return None
        pos = gd3_abs + 12  # skip magic + version + size
        # fields: track_en, track_jp, game_en, game_jp, sys_en, sys_jp, author_en, ...
        fields = []
        for _ in range(7):  # read up through author_en (index 6)
            s, pos = _read_gd3_string(raw, pos)
            fields.append(s)
        game_en = fields[2].strip()
        author_en = fields[6].strip()
        return author_en, game_en
    except Exception:
        return None


def split_composers(raw: str) -> list[str]:
    """Split a compound author string like 'A, B, C' into individual names."""
    parts = re.split(r',|;|\band\b', raw)
    names = []
    for p in parts:
        n = p.strip()
        if n and len(n) > 1 and not n.startswith('('):
            n = re.sub(r'\s*\(.*?\)', '', n).strip()
            if n:
                names.append(n)
    return names


vgm_files = sorted(Path('data/vgm').glob('*.vgz'))
print(f'Scanning {len(vgm_files)} files...')

raw_authors = Counter()
ind_authors = Counter()
games = Counter()
no_gd3 = 0

for i, f in enumerate(vgm_files):
    if i % 2000 == 0:
        print(f'  {i}/{len(vgm_files)}...')
    result = fast_read_gd3(f)
    if result is None:
        no_gd3 += 1
        continue
    author_en, game_en = result
    if author_en and author_en.lower() not in ('unknown', ''):
        raw_authors[author_en] += 1
        for name in split_composers(author_en):
            ind_authors[name] += 1
    if game_en:
        games[game_en] += 1

print(f'\nDone. Files without GD3 tag: {no_gd3}')

print(f'\n=== TOP 60 INDIVIDUAL COMPOSERS (after splitting) ===')
for name, count in ind_authors.most_common(60):
    print(f'  {count:5d}  {name}')
print(f'\nTotal unique individual composers: {len(ind_authors)}')
print(f'Composers with >=10 tracks: {sum(1 for c in ind_authors.values() if c >= 10)}')
print(f'Composers with >=5 tracks:  {sum(1 for c in ind_authors.values() if c >= 5)}')
print(f'Composers with >=2 tracks:  {sum(1 for c in ind_authors.values() if c >= 2)}')

print(f'\n=== TOP 60 GAMES ===')
for name, count in games.most_common(60):
    print(f'  {count:4d}  {name}')
print(f'\nTotal unique games: {len(games)}')
