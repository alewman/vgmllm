import sys; sys.path.insert(0,'src')
from genesis_music.vgm_parser import load_vgm, EventType
from genesis_music.ym2612 import decode_vgm, fnumber_to_midi

src = load_vgm('data/vgm/Thunder_Force_IV__Lightening_Force___Mega_Drive__Genesis___23_-_Metal_Squad__Stage_8_.vgz')

WIN_START = 441000   # 10.0s
WIN_END   = 661500   # 15.0s

current_sample = 0
ch3_fnums = []
ch3_keyon = []
ch3_fnum_shadow_hi = 0
ch3_block_shadow = 0

for ev in src.events:
    if ev.type == EventType.WAIT:
        current_sample += ev.value
    elif ev.type == EventType.YM2612_PORT0:
        reg, val = ev.register, ev.value
        if reg == 0xA6:   # CH3 fnum hi+block (shadow reg)
            ch3_fnum_shadow_hi = val & 0x07
            ch3_block_shadow   = (val >> 3) & 0x07
        if reg == 0xA2 and WIN_START <= current_sample <= WIN_END:
            # fnum_lo committed — combine with shadow
            fnum = (ch3_fnum_shadow_hi << 8) | val
            midi = fnumber_to_midi(fnum, ch3_block_shadow)
            ch3_fnums.append((current_sample, fnum, ch3_block_shadow, midi))
        if reg == 0x28:
            ch_bits = val & 0x07
            if ch_bits in (2, 6):  # CH3 on port 0 or port 1 encoding
                if WIN_START <= current_sample <= WIN_END:
                    state = "ON " if (val >> 4) else "OFF"
                    ch3_keyon.append((current_sample, state, val))
    elif ev.type == EventType.END:
        break

t = lambda s: f'{s/44100:.3f}s'
print(f'F-number writes to CH3 in 10-15s: {len(ch3_fnums)}')
for s, fnum, blk, midi in ch3_fnums[:60]:
    print(f'  {t(s):9s}  fnum=0x{fnum:03X} block={blk} midi={midi:3d}')

print()
print(f'Key events CH3 in 10-15s: {len(ch3_keyon)}')
for s, state, v in ch3_keyon:
    print(f'  {t(s):9s}  {state}  (0x{v:02X})')
