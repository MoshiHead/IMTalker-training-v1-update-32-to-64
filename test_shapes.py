"""
Quick shape verification for 32D→64D expansion.
Run this BEFORE starting any training:
    python test_shapes.py
All 6 checks must pass.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import argparse

from renderer.models import IMTRenderer

args = argparse.Namespace(
    input_size=512,
    swin_res_threshold=128,
    window_size=8,
    num_heads=8,
)

print("Building IMTRenderer...")
model = IMTRenderer(args).eval()

x = torch.randn(2, 3, 512, 512)

print("\nRunning shape checks...\n")

# 1. MotionEncoder still outputs 32D
z32 = model.mot_encode(x)
assert z32.shape == (2, 32), f"FAIL: expected (2,32), got {z32.shape}"
print(f"  CHECK 1 PASS  mot_encode output shape:   {z32.shape}  (32D preserved)")

# 2. MotionExpander outputs 64D
z64 = model.expand_mot(z32)
assert z64.shape == (2, 64), f"FAIL: expected (2,64), got {z64.shape}"
print(f"  CHECK 2 PASS  expand_mot output shape:   {z64.shape}  (64D expanded)")

# 3. IdentityAdaptive accepts 64D, outputs 64D
f_r, app = model.app_encode(x)
ta = model.id_adapt(z64, app)
assert ta.shape == (2, 64), f"FAIL: expected (2,64), got {ta.shape}"
print(f"  CHECK 3 PASS  id_adapt output shape:     {ta.shape}  (64D adapted)")

# 4. MotionDecoder accepts 64D, outputs 4 motion maps
m1, m2, m3, m4 = model.mot_decode(ta)
assert m1.shape == (2, 512,  8,  8), f"FAIL m1: {m1.shape}"
assert m2.shape == (2, 512, 16, 16), f"FAIL m2: {m2.shape}"
assert m3.shape == (2, 256, 32, 32), f"FAIL m3: {m3.shape}"
assert m4.shape == (2, 128, 64, 64), f"FAIL m4: {m4.shape}"
print(f"  CHECK 4 PASS  mot_decode motion maps:    m1={m1.shape} m2={m2.shape} m3={m3.shape} m4={m4.shape}")

# 5. Full forward pass produces correct output image shape
out, t_c_returned = model(x, x)
assert out.shape == (2, 3, 512, 512), f"FAIL output: {out.shape}"
assert t_c_returned.shape == (2, 32), f"FAIL returned latent: {t_c_returned.shape}"
print(f"  CHECK 5 PASS  full forward:              output={out.shape}  returned_32d={t_c_returned.shape}")

# 6. Generator's ref_x is still 32D (generator pipeline untouched)
ref_x = model.latent_token_encoder(x)
assert ref_x.shape == (2, 32), f"FAIL ref_x: {ref_x.shape}"
print(f"  CHECK 6 PASS  latent_token_encoder:      {ref_x.shape}  (generator ref still 32D)")

print("\n  ALL 6 CHECKS PASSED — safe to proceed to training.\n")
