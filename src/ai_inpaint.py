"""
ai_inpaint.py
-------------
AI image-generator step (diffusion INPAINTING) for the QR replacement.

Why inpainting + compositing instead of pure text-to-image?
  * A generative model that "draws" a QR from a text prompt produces a
    plausible-looking but INVALID QR (it cannot reproduce the exact
    error-corrected module grid), and free-form editing can subtly alter
    names/faces.
  * Masked inpainting is given the certificate, a MASK limited to the old-QR
    square, and your natural-language prompt. The model may only regenerate
    pixels inside the mask, so every other pixel (design, text, face, seals)
    is provably unchanged.
  * We then composite the REAL, freshly computed QR (perspective-warped to the
    exact quad) on top of the inpainted region -> the edit is AI-generated and
    blended as requested, AND the QR is guaranteed crisp & scannable.

Model (Hugging Face, runs on a Kaggle T4 16GB):
    diffusers/stable-diffusion-xl-1.0-inpainting-0.1   (default, fp16)
    stabilityai/stable-diffusion-2-inpainting         (lighter fallback)

Everything is lazy: torch/diffusers are only imported when you actually run
the AI step, so the deterministic warp path needs no GPU and no big download.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .replacer import replace_qr

# The exact prompt requested by the user.
DEFAULT_PROMPT = (
    "Generate an image of this certificate replacing only the old QR code "
    "with a new crisp black and white QR code in the same position. "
    "Do not change any design, do not change any information, do not change "
    "the face or anything else. Just replace the old QR code with the new "
    "QR code, keeping the identical certificate layout, colors, and text."
)
DEFAULT_NEGATIVE = (
    "invalid qr code, blurry qr, distorted text, changed name, altered face, "
    "modified design, extra logo, watermark, deformed, low quality, artifacts"
)

_PIPE = None
_DEVICE = None


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def load_pipeline(model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
    """Lazily load (and cache) the HF inpainting pipeline."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import torch
    from diffusers import AutoPipelineForInpainting

    dtype = torch.float16 if _device() == "cuda" else torch.float32
    kw = dict(torch_dtype=dtype)
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kw)
    except Exception:
        # some repos need the variant flag only on GPU fp16
        kw["variant"] = "fp16"
        pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kw)

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
    pipe.set_progress_bar_config(disable=False)
    _PIPE = pipe
    return pipe


def ai_available() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _mask_from_quad(shape_hw: tuple[int, int], quad: np.ndarray,
                    dilate_px: int = 6) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    if dilate_px:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def _crop_bbox(quad: np.ndarray, h: int, w: int, margin: int):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    x0 = int(max(0, x0 - margin)); y0 = int(max(0, y0 - margin))
    x1 = int(min(w, x1 + margin)); y1 = int(min(h, y1 + margin))
    return x0, y0, x1, y1


def _round8(v: int) -> int:
    return max(8, int(round(v / 8.0)) * 8)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def ai_replace_qr(cert_bgr: np.ndarray, region, qr_bgr: np.ndarray,
                  prompt: str = DEFAULT_PROMPT,
                  negative_prompt: str = DEFAULT_NEGATIVE,
                  model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                  steps: int = 30, guidance: float = 8.0, seed: int = 0,
                  margin: int = 120, mode: str = "inpaint_then_paste",
                  replace_mode: str = "full_replace",
                  cover_scale: float = 1.08):
    """
    Run AI inpainting over the old-QR mask with `prompt`, then (default)
    composite the real QR so it scans.

    mode:
      "inpaint_then_paste" (default, recommended):
            AI regenerates/blends the QR region per the prompt, then the
            deterministic QR is warped on top -> scannable + AI-edited.
      "generative_only":
            return the raw model output (NOT guaranteed to scan -- for
            comparison/experimentation only).

    Returns (out_bgr, info_dict). If the model can't load, falls back to the
    deterministic warp and info["fallback"]=True.
    """
    h, w = cert_bgr.shape[:2]
    quad = region.quad.astype(np.float32)

    if not ai_available():
        out = replace_qr(cert_bgr, region, qr_bgr, mode=replace_mode,
                         cover_scale=cover_scale)
        return out, {"ran": False, "fallback": "diffusers/torch not installed"}

    # The SDXL inpainting model is GPU-only in practice: never attempt it on
    # CPU (multi-GB download, minutes/image, likely OOM). Fall back instantly.
    if _device() != "cuda":
        out = replace_qr(cert_bgr, region, qr_bgr, mode=replace_mode,
                         cover_scale=cover_scale)
        return out, {"ran": False,
                     "fallback": "AI inpainting needs a CUDA GPU; used warp"}

    try:
        import torch
    except Exception:
        torch = None
    try:
        from PIL import Image
    except Exception as e:
        out = replace_qr(cert_bgr, region, qr_bgr, mode=replace_mode,
                         cover_scale=cover_scale)
        return out, {"ran": False, "fallback": f"PIL missing: {e}"}

    try:
        pipe = load_pipeline(model_id)
    except Exception as e:  # download/OOM -> graceful fallback
        out = replace_qr(cert_bgr, region, qr_bgr, mode=replace_mode,
                         cover_scale=cover_scale)
        return out, {"ran": False, "fallback": f"model load failed: {e}"}

    mask = _mask_from_quad((h, w), quad, dilate_px=8)
    x0, y0, x1, y1 = _crop_bbox(quad, h, w, margin)
    crop = cert_bgr[y0:y1, x0:x1].copy()
    cmask = mask[y0:y1, x0:x1].copy()

    ch, cw = crop.shape[:2]
    W8, H8 = _round8(cw), _round8(ch)

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((W8, H8))
    pil_mask = Image.fromarray(cmask).resize((W8, H8), Image.NEAREST)

    gen = torch.Generator(device="cpu").manual_seed(seed) if torch is not None else None
    result = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=pil_img, mask_image=pil_mask,
        height=H8, width=W8,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=gen,
    ).images[0]
    result = result.resize((cw, ch), Image.LANCZOS)
    inp_crop = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # Strictly apply ONLY inside the mask -> rest of certificate untouched.
    m3 = cv2.cvtColor(cmask, cv2.COLOR_GRAY2BGR) > 0
    edited_crop = crop.copy()
    edited_crop[m3] = inp_crop[m3]

    edited = cert_bgr.copy()
    edited[y0:y1, x0:x1] = edited_crop

    info = {"ran": True, "model": model_id, "device": _device(),
            "steps": steps, "crop": (x0, y0, x1, y1), "mode": mode}

    if mode == "generative_only":
        return edited, info

    # Composite the REAL QR (guaranteed scannable) over the AI region.
    final = replace_qr(edited, region, qr_bgr, mode=replace_mode,
                            cover_scale=cover_scale)
    info["qr_composited"] = True
    return final, info
