from genesis_music.model import config_large, VgmGPT
cfg = config_large(vocab_size=456, seq_len=256)
m = VgmGPT(cfg)
e = (2 * 20) ** -0.5
print(f"residual_scale = 1/sqrt(40) = {e:.5f}")
print(f"out.weight  std = {m.layers[0].attn.out.weight.std().item():.5f}  (expect {0.02*e:.5f})")
print(f"down.weight std = {m.layers[0].ff.down.weight.std().item():.5f}  (expect {0.02*e:.5f})")
print(f"qkv.weight  std = {m.layers[0].attn.qkv.weight.std().item():.5f}  (expect ~0.02000)")
print(f"up.weight   std = {m.layers[0].ff.up.weight.std().item():.5f}  (expect ~0.02000)")
print("OK" if abs(m.layers[0].attn.out.weight.std().item() - 0.02*e) < 0.002 else "FAIL")
