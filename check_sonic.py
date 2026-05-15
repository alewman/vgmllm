from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import Ym2612State, CH_DAC
from genesis_music.music_analysis import should_discard
import glob

sonic1 = sorted(f for f in glob.glob('data/**/*.vg[mz]', recursive=True)
                if 'Sonic_the_Hedgehog' in f)
print(f'Sonic files: {len(sonic1)}')
for f in sonic1[:20]:
    try:
        vgm = load_vgm(f)
        dec = Ym2612State()
        events = list(dec.process_vgm(vgm))
        dur = vgm.header.total_samples / 44100
        dac = sum(1 for e in events if e.channel == CH_DAC)
        fm  = sum(1 for e in events if e.channel < 6)
        disc, reason = should_discard(events, vgm.header.total_samples)
        name = f.split('\\')[-1][:65]
        keep = 'FILTER' if disc else 'KEEP'
        print(f'  {dur:5.1f}s  {dac:4d}DAC {fm:4d}FM  {keep}  {reason or ""}  {name}')
    except Exception as ex:
        print(f'  ERROR: {ex}  {f.split(chr(92))[-1]}')
