import sys; sys.path.insert(0,'src')
from pathlib import Path
from collections import Counter
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm

vgm = load_vgm(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz'))
notes, patches = decode_vgm(vgm)
fm3 = [n for n in notes if n.channel == 3]

patch_info = Counter()
for n in fm3:
    p = n.patch
    if p:
        k = (p.algorithm, p.feedback, tuple(p.ssg_eg), tuple(p.am_en), p.lfo_en, tuple(p.tl))
        patch_info[k] += 1

print(f'FM3 notes: {len(fm3)}, unique patch types: {len(patch_info)}')
for k, cnt in patch_info.most_common(5):
    algo, fb, ssg, am, lfo, tl = k
    print(f'  x{cnt:3d}: algo={algo} fb={fb} ssg={ssg} am={am} lfo={lfo} tl={tl}')

print()
print('First 20 FM3 notes:')
for n in fm3[:20]:
    p = n.patch
    dur = n.sample_off - n.sample_on
    algo = p.algorithm if p else '?'
    ssg = p.ssg_eg if p else '?'
    tl = p.tl if p else '?'
    print(f'  smp={n.sample_on:7d} pitch={n.pitch:3d} dur={dur:5d} algo={algo} ssg={ssg} tl={tl}')
