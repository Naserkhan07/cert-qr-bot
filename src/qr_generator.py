"""
qr_generator.py
---------------
Generate a fresh, crisp QR code image for a given payload (the URL/text that
a phone will read when scanning the certificate).

The QR is rendered at HIGH pixel resolution so that when it is perspective-
warped down onto the small QR region of the certificate, every module stays
sharp and black/white (which is what makes it scan reliably).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H,
)

_EC = {
    "L": ERROR_CORRECT_L,   # ~7%
    "M": ERROR_CORRECT_M,   # ~15%
    "Q": ERROR_CORRECT_Q,   # ~25%
    "H": ERROR_CORRECT_H,   # ~30%
}


def build_qr(payload: str,
             box_size: int = 20,
             border: int = 4,
             error_correction: str = "M",
             fg: tuple[int, int, int] = (0, 0, 0),
             bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Return a square BGR uint8 image (white background) containing the QR.

    box_size: pixels per module at render time (large = crisp after warping)
    border:   quiet-zone width in modules (standard 4)
    """
    qr = qrcode.QRCode(
        version=None,                       # auto-size to payload
        error_correction=_EC[error_correction.upper()],
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fg, back_color=bg).convert("RGB")
    arr = np.array(img)[:, :, ::-1].copy()  # RGB -> BGR
    return arr


def qr_png_bytes(payload: str, **kwargs) -> bytes:
    """Render the QR and return PNG bytes (e.g. for saving a standalone file)."""
    arr = build_qr(payload, **kwargs)
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes() if ok else b""
