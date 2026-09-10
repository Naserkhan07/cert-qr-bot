"""
qr_art.py
---------
OPTIONAL AI "artistic QR" generation using the Hugging Face ControlNet model
**QR Code Monster** (`monster-labs/control_v1p_sd15_qrcode_monster`, v2) with a
Stable Diffusion 1.5 base.

This is the popular "image/text -> QR code" diffusion model family (QR Code
Monster, IllusionDiffusion, etc.): a normal QR matrix is fed to ControlNet as
the conditioning image together with a text prompt, and the model produces a
creative QR in which art is blended into the modules.

Important facts (from the model cards):
  * Not every generation scans -- typical scannability is 50-80%, so the
    official workflow is "generate several seeds and keep a readable one".
  * High `controlnet_conditioning_scale` (1.2-1.8) => more readable; low =>
    more artistic. We default high because the QR must work on a certificate.
  * These codes are usually colourful/textured -- great for marketing, but for
    an official certificate the plain black-and-white `qr_generator.build_qr`
    (100% reliable) is recommended.

Therefore `build_artistic_qr()`:
  1. renders the EXACT, verified QR matrix (error-correction H),
  2. runs the ControlNet pipeline over several seeds,
  3. DECODES every candidate and keeps one that reads back as `payload`,
  4. returns None if none scan within `max_tries` (caller falls back to the
     deterministic QR, so a batch never breaks).

Runs on a CUDA GPU (Kaggle free T4). torch/diffusers are imported lazily, so
importing this module on a CPU-only box is harmless.
"""

from __future__ import annotations

import cv2
import numpy as np

from .qr_generator import build_qr

# A prompt that pushes the model toward a clean, document-friendly result while
# still using the AI generator. Override for creative/artistic codes.
CLEAN_PROMPT = (
    "a clean flat black and white QR code on a plain white background, "
    "crisp high contrast modules, minimalist, sharp edges, no decoration"
)
CLEAN_NEGATIVE = (
    "colorful, painting, landscape, portrait, photo, texture, blurry, "
    "watermark, text, logo, distorted, low contrast, cluttered background"
)

_ART_PIPE = None
_ART_DEVICE = None


def _device() -> str:
    global _ART_DEVICE
    if _ART_DEVICE is not None:
        return _ART_DEVICE
    try:
        import torch
        _ART_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _ART_DEVICE = "cpu"
    return _ART_DEVICE


def art_available() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _condition_image(payload: str, module_px: int = 16,
                     gray_bg: bool = False) -> np.ndarray:
    """Render the exact QR matrix as the ControlNet conditioning image.

    QR Monster expects ~16px modules. v2 blends better on a gray (#808080)
    background; for maximum readability we use white (gray_bg=False).
    """
    qr = build_qr(payload, box_size=module_px, border=0,
                  error_correction="H")
    if gray_bg:
        # QR has white bg; replace outer non-module area with gray is complex,
        # so we tint the whole background gray while keeping modules black.
        gray = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
        out = np.full_like(qr, 128)
        dark = gray < 128
        out[dark] = (0, 0, 0)
        return out
    return qr


def load_art_pipeline(
        controlnet_id: str = "monster-labs/control_v1p_sd15_qrcode_monster",
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Lazily load and cache the ControlNet QR pipeline (fp16 on GPU)."""
    global _ART_PIPE
    if _ART_PIPE is not None:
        return _ART_PIPE
    import torch
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetPipeline,
                           DPMSolverMultistepScheduler)

    dtype = torch.float16 if _device() == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    if _device() == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to("cuda")
    else:
        pipe.to("cpu")
    _ART_PIPE = pipe
    return pipe


def _decode_qr(img_bgr: np.ndarray, want: str) -> bool:
    """True if img contains a QR that decodes exactly to `want`."""
    from .pipeline import _decode_any_qr  # reuse strong verifier
    h, w = img_bgr.shape[:2]
    for scale in (1, 2):
        im = img_bgr if scale == 1 else cv2.resize(
            img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        got = _decode_any_qr(im)
        if got and got.strip() == want.strip():
            return True
    return False


def build_artistic_qr(payload: str,
                      prompt: str = CLEAN_PROMPT,
                      negative_prompt: str = CLEAN_NEGATIVE,
                      size: int = 768,
                      controlnet_scale: float = 1.5,
                      guidance_scale: float = 7.5,
                      steps: int = 30,
                      max_tries: int = 6,
                      seed0: int = 1000,
                      gray_bg: bool = False):
    """
    Generate an AI (ControlNet) artistic QR for `payload`.

    Returns a square BGR uint8 image that DECODES to `payload`, or None if no
    seed produced a scannable code within `max_tries` (caller then falls back
    to the deterministic QR).
    """
    if not art_available() or _device() != "cuda":
        return None
    try:
        import torch
        from PIL import Image
        pipe = load_art_pipeline()
    except Exception as e:  # model download/OOM
        print(f"[qr_art] pipeline unavailable ({e}); using standard QR")
        return None

    cond = _condition_image(payload, module_px=16, gray_bg=gray_bg)
    cond_pil = Image.fromarray(cv2.cvtColor(cond, cv2.COLOR_BGR2RGB))
    # make conditioning square at requested size
    cond_pil = cond_pil.resize((size, size), Image.NEAREST)

    for i in range(max_tries):
        seed = seed0 + i
        g = torch.Generator(device="cpu").manual_seed(seed)
        try:
            result = pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                image=cond_pil, width=size, height=size,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=g,
            ).images[0]
        except Exception as e:
            print(f"[qr_art] generation failed (seed {seed}): {e}")
            continue
        bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if _decode_qr(bgr, payload):
            print(f"[qr_art] scannable artistic QR at seed {seed} "
                  f"(try {i+1}/{max_tries})")
            return bgr
        print(f"[qr_art] seed {seed} did not scan; retrying...")
    print("[qr_art] no scannable variant found; falling back to standard QR")
    return None
