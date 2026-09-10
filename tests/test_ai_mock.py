"""Validate the AI inpainting integration WITHOUT a GPU/model by injecting a
fake diffusion pipeline. Confirms: mask cropping, model call signature,
masked-only application, QR compositing, and final scannability."""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ai_inpaint
from src.pipeline import Config, process_one

class FakePipe:
    """Pretends to be a diffusers inpainting pipeline: 'regenerates' the masked
    region as flat gray (proves the mask/insert path), leaves the rest."""
    def __call__(self, prompt=None, negative_prompt=None, image=None,
                 mask_image=None, height=None, width=None, **kw):
        arr = np.array(image).copy()           # RGB
        m = np.array(mask_image) > 127
        # fill mask with a neutral gray (what a model might hallucinate)
        arr[m] = (160, 160, 160)
        class R:  # mimic diffusers StableDiffusionPipelineOutput
            images = [Image.fromarray(arr)]
        return R()
    def set_progress_bar_config(self, **k): pass
    def enable_xformers_memory_efficient_attention(self): pass
    def enable_model_cpu_offload(self): pass
    def to(self, *a, **k): return self

# monkeypatch the model loader + availability (patch the names the pipeline sees)
import src.pipeline as pl
ai_inpaint.ai_available = lambda: True
pl.ai_available = lambda: True
ai_inpaint._DEVICE = "cuda"
ai_inpaint.load_pipeline = lambda *a, **k: FakePipe()

cfg = Config(
    input_dir="demo/input", output_dir="tests/ai_mock_out",
    model_dir="models/wechat", use_ai=True, use_cnn=True,
    base_url="https://tgmfc-certificates.github.io/2024-25/",
    save_debug=False,
)
out, res = process_one("demo/input/031-2024-25.png", cfg)
print("status:", res.status, "| ai_used:", res.ai_used, "| verified:", res.verified)
print("note:", res.note)

# checks
assert res.ai_used is True, "AI path should run"
assert res.status == "ok", f"final QR must verify, got {res.status}"

# confirm pixels OUTSIDE the QR region are byte-identical to input
inp = cv2.imread("demo/input/031-2024-25.png")
diff = cv2.absdiff(inp, out).sum(axis=2)
ys, xs = np.where(diff > 10)
print("changed bbox: x[%d-%d] y[%d-%d]" % (xs.min(), xs.max(), ys.min(), ys.max()))
# QR box is around x63-127 y159-224 (+margin 120 -> x0-247 y39-344). Text region
# e.g. top-right CERTIFICATE at x>150 y160-220 overlaps margin, but only the
# MASK (QR box) is overwritten; assert nothing changed far away.
far = diff[300:, :]  # bottom area (seals/signatures) must be untouched
assert far.max() == 0, "bottom seal/signature region must be untouched"
top_left_text = diff[10:60, 200:600]  # header text far from QR
assert top_left_text.max() == 0, "header text must be untouched"

# the fake gray fill must be gone (covered by composited QR): sample a QR pixel
# inside the box should be black or white, not gray 160.
box = out[165:220, 70:122]
vals = box.reshape(-1,3).mean(axis=1)
grayish = ((vals > 120) & (vals < 200)).mean()
print("fraction grayish inside QR (should be ~0):", round(float(grayish), 3))
assert grayish < 0.05, "QR area should be black/white, not the model's gray"

print("\nAI INPAINT INTEGRATION (mock) PASSED")
