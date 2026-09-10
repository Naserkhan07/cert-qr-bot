"""
replacer.py
-----------
Paste the freshly generated QR onto the certificate at the EXACT location,
size and rotation of the old QR.

- Uses a perspective warp so rotated/skewed old QRs are matched exactly.
- Uses NEAREST-neighbour rendering so QR modules stay hard-edged (critical for
  scanning) -- no anti-aliased grey fuzz.
- `mode`:
    "full_replace" -> new QR (with white quiet zone) covers the entire old QR
                      box including its black frame. Cleanest, most reliable.
    "keep_frame"   -> the old black border is preserved; only the inner QR
                      area is replaced (the new QR is shrunk inside the frame).
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import QRRegion, _order_points


def _scale_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    """Scale a quad about its centre."""
    c = quad.mean(axis=0)
    return c + (quad - c) * scale


def _warp_qr_to_cert(cert: np.ndarray, qr_img: np.ndarray,
                     dst_quad: np.ndarray, supersample: int = 4) -> np.ndarray:
    """Perspective-warp qr_img onto cert at dst_quad (TL,TR,BR,BL).

    The QR is first re-rendered at ~`supersample`x the destination size, then
    warped with bilinear sampling (a good approximation of area-averaged
    downscaling) and finally re-binarised to pure black/white inside the box.
    This keeps every module hard-edged and scannable even when the on-cert
    QR is very small (avoids the aliasing of a single huge->tiny NEAREST warp).
    """
    h, w = cert.shape[:2]
    ss = int(supersample)
    side = float(np.mean([
        np.linalg.norm(dst_quad[1] - dst_quad[0]),
        np.linalg.norm(dst_quad[2] - dst_quad[1]),
        np.linalg.norm(dst_quad[3] - dst_quad[2]),
        np.linalg.norm(dst_quad[0] - dst_quad[3]),
    ]))
    target = max(8, int(round(side * ss)))
    if qr_img.shape[0] != target:
        interp = cv2.INTER_AREA if qr_img.shape[0] > target else cv2.INTER_NEAREST
        qr_img = cv2.resize(qr_img, (target, target), interpolation=interp)

    qh, qw = qr_img.shape[:2]
    # Warp onto an ss-times-oversized overlay at the ss-scaled quad, then pull
    # back down to cert resolution with INTER_AREA -> a proper area-averaged
    # anti-aliased downscale of the perspective warp (keeps modules legible at
    # small on-cert sizes).
    big_W, big_H = w * ss, h * ss
    big_quad = dst_quad.astype(np.float32) * ss
    src = np.array([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, big_quad)
    warped_big = cv2.warpPerspective(
        qr_img, M, (big_W, big_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    warped = cv2.resize(warped_big, (w, h), interpolation=cv2.INTER_AREA)

    # mask of the QR box
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    # re-binarise the warped patch to pure black / white (hard module edges)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(warped_gray, 160, 255, cv2.THRESH_BINARY)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    out = cert.copy()
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
    out[m3] = binary_bgr[m3]
    return out


def replace_qr(cert: np.ndarray, region: QRRegion, qr_img: np.ndarray,
               mode: str = "full_replace",
               inner_margin_frac: float = 0.10,
               cover_scale: float = 1.08) -> np.ndarray:
    """
    Return a new certificate image with the new QR in place of the old one.

    Everything outside the QR quad is left byte-for-byte untouched.

    cover_scale (full_replace only): the new QR's white box is pasted over a
    quad scaled outward by this factor so it reliably covers the ENTIRE old
    black frame plus any dark halo / AI-inpaint remnant at the frame edge.
    1.0 = exact old-quad footprint; ~1.08 fully hides the old frame (the white
    background looks natural against the certificate and keeps a valid quiet
    zone).
    """
    quad = _order_points(region.quad)

    if mode == "keep_frame":
        # Replace only the inside of the black frame.
        dst = _scale_quad(quad, 1.0 - 2 * inner_margin_frac)
        # Give the pasted QR its own thin white border so modules never touch
        # the black frame (keeps quiet-zone scanning rules).
        qr = _add_white_border(qr_img, frac=0.06)
    else:
        # Full replace: scale the footprint outward a little so the new QR's
        # white background fully hides the old frame + any edge halo.
        dst = _scale_quad(quad, float(cover_scale))
        qr = qr_img

    return _warp_qr_to_cert(cert, qr, dst)


def _add_white_border(img: np.ndarray, frac: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    b = int(round(min(h, w) * frac))
    return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))
