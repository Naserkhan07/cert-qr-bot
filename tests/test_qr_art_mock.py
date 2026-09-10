"""Validate the artistic-QR backend integration WITHOUT a GPU by monkeypatching
build_artistic_qr. Covers: (a) artistic QR used + verified, (b) fallback to
standard when the AI returns None."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.pipeline as pl
from src.qr_generator import build_qr

base_cfg = dict(
    input_dir="demo/input", model_dir="models/wechat", use_cnn=True,
    base_url="https://tgmfc-certificates.github.io/2024-25/",
    save_debug=False, use_ai=False,
)

def fake_artistic_ok(payload, **kw):
    # "AI" output: a larger, valid QR square (artistic models emit ~768px),
    # returned as a clean decodable BGR image to prove the pipeline path.
    return build_qr(payload, box_size=16, border=4, error_correction="H")

def fake_artistic_none(payload, **kw):
    return None  # model couldn't produce a scannable code

# (a) artistic backend returns a scannable styled QR
cfg = pl.Config(qr_backend="artistic", output_dir="tests/art_out_a", **base_cfg)
pl.build_artistic_qr = fake_artistic_ok
out_a, res_a = pl.process_one("demo/input/031-2024-25.png", cfg)
print("(a) artistic used -> status:", res_a.status, "| note:", res_a.note,
      "| verified:", res_a.verified)
assert res_a.status == "ok" and "artistic" in res_a.note

# (b) artistic backend fails -> standard fallback, still ok
cfg_b = pl.Config(qr_backend="artistic", output_dir="tests/art_out_b", **base_cfg)
pl.build_artistic_qr = fake_artistic_none
out_b, res_b = pl.process_one("demo/input/031-2024-25.png", cfg_b)
print("(b) art->None fallback -> status:", res_b.status, "| note:", res_b.note)
assert res_b.status == "ok"

# (c) standard backend unaffected
cfg_c = pl.Config(qr_backend="standard", output_dir="tests/art_out_c", **base_cfg)
out_c, res_c = pl.process_one("demo/input/031-2024-25.png", cfg_c)
print("(c) standard -> status:", res_c.status)
assert res_c.status == "ok"

print("\nARTISTIC QR BACKEND INTEGRATION (mock) PASSED")
