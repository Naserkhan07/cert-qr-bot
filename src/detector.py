"""
detector.py
-----------
Locate the OLD QR code on a certificate image.

Detection cascade (tries each, returns first high-confidence hit):
  1. WeChat CNN QR detector  (Deep-CV model downloaded from Hugging Face)
  2. OpenCV Aruco QR detector
  3. Geometric contour finder (finds the thick black square FRAME around a QR
     even when the QR itself is too dense/small to decode -- works on certs
     whose old QR is decorative or damaged)

Every detector returns the quadrangle of the *outer QR box* (including the
black border frame when present), so the new QR can be warped to exactly that
size / position / rotation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QRRegion:
    """The 4 corners of the QR's outer box, in order TL, TR, BR, BL (px)."""
    quad: np.ndarray            # shape (4, 2), float32
    method: str                 # which detector found it
    side_px: float              # average side length in pixels
    score: float                # confidence 0..1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    return (
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - tr)),
        float(np.linalg.norm(bl - br)),
        float(np.linalg.norm(tl - bl)),
    )


def _valid_quad(gray: np.ndarray, quad: np.ndarray,
                min_finders: int = 2) -> bool:
    """
    Reject false-positive quads from the ML/OpenCV detectors.
    A real QR quad: roughly square, on-canvas, convex, and contains >=2 of the
    three characteristic finder patterns (nested squares).
    """
    h, w = gray.shape[:2]
    try:
        quad = _order_points(quad)
    except Exception:
        return False
    sides = _side_lengths(quad)
    if min(sides) < max(8.0, 0.04 * min(h, w)):
        return False
    if max(sides) > max(h, w) * 1.15:
        return False
    if min(sides) / max(sides) < 0.7:
        return False
    # convexity (cross products same sign)
    edges = np.roll(quad, -1, axis=0) - quad
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - \
        edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(crosses >= -1e-3) or np.all(crosses <= 1e-3)):
        return False
    # mostly on canvas
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    if x1 < -0.05 * w or x0 > 1.05 * w or y1 < -0.05 * h or y0 > 1.05 * h:
        return False
    if _count_finder_patterns(gray, quad) < min_finders:
        return False
    return True


def _count_finder_patterns(gray: np.ndarray, quad: np.ndarray) -> int:
    """
    Count finder-like nested squares inside a quad. A QR has 3 finder patterns
    (TL, TR, BL corners). Used to score contour candidates that look like a QR.
    """
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 10 or h < 10:
        return 0
    M = cv2.getPerspectiveTransform(
        quad.astype(np.float32),
        np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32),
    )
    patch = cv2.warpPerspective(gray, M, (w, h))
    _, bw = cv2.threshold(patch, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(bw, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    hier = hier[0]

    def nested_depth(i: int) -> int:
        depth = 0
        parent = hier[i][3]
        while parent != -1:
            depth += 1
            parent = hier[parent][3]
        return depth

    finders = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area == 0:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        ratio = ww / float(hh)
        area_ratio = area / float(ww * hh)
        # finder outer ring: roughly square, moderately sized, >=2 nestings
        if (0.7 < ratio < 1.35 and area_ratio > 0.55
                and ww > w * 0.08 and ww < w * 0.6 and nested_depth(i) >= 2):
            finders += 1
    return finders


# ---------------------------------------------------------------------------
# Detector 1: WeChat CNN (model from Hugging Face)
# ---------------------------------------------------------------------------
_WECHAT = None


def _get_wechat_detector(model_dir: str):
    """Lazy-load the WeChat QR CNN model if the 4 files are present."""
    global _WECHAT
    if _WECHAT is not None:
        return _WECHAT
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return None
    files = ["detect.prototxt", "detect.caffemodel",
             "sr.prototxt", "sr.caffemodel"]
    paths = [os.path.join(model_dir, f) for f in files]
    if not all(os.path.exists(p) for p in paths):
        return None
    try:
        _WECHAT = cv2.wechat_qrcode_WeChatQRCode(*paths)
    except Exception:
        _WECHAT = None
    return _WECHAT


def _detect_wechat(img: np.ndarray, model_dir: str):
    det = _get_wechat_detector(model_dir)
    if det is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    for scale in (1.0, 2.0, 3.0):
        src = gray if scale == 1.0 else cv2.resize(
            gray, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC)
        try:
            res, points = det.detectAndDecode(src)
        except Exception:
            res, points = [], None
        if points is not None and len(points) > 0:
            for p in points:
                quad = _order_points(np.asarray(p, dtype=np.float32) / scale)
                if _valid_quad(gray, quad, min_finders=2):
                    return quad
    return None


# ---------------------------------------------------------------------------
# Detector 2: OpenCV ArUco / classic QR detector
# ---------------------------------------------------------------------------
def _detect_opencv(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    detectors = []
    if hasattr(cv2, "QRCodeDetectorAruco"):
        try:
            detectors.append(cv2.QRCodeDetectorAruco())
        except Exception:
            pass
    detectors.append(cv2.QRCodeDetector())
    for scale in (1.0, 2.0, 3.0, 4.0):
        src = gray if scale == 1.0 else cv2.resize(
            gray, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC)
        for det in detectors:
            ok, pts = det.detect(src)
            if ok and pts is not None:
                quad = _order_points(pts.reshape(4, 2) / scale)
                if _valid_quad(gray, quad, min_finders=2):
                    return quad
    return None


# ---------------------------------------------------------------------------
# Detector 3: geometric contour finder (thick black frame around QR)
# ---------------------------------------------------------------------------
def _detect_by_contours(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img_area = h * w

    best = None
    best_score = -1.0

    for thresh in (90, 120, 150, 180):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # close small gaps inside the frame
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * 0.0008 or area > img_area * 0.25:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) != 4:
                # also accept rotated rect
                rrect = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), _ = rrect
                if rw == 0 or rh == 0:
                    continue
                sq = min(rw, rh) / max(rw, rh)
                if sq < 0.82:
                    continue
                box = cv2.boxPoints(rrect)
            else:
                box = approx.reshape(4, 2).astype(np.float32)
                (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
                sq = min(rw, rh) / max(rw, rh) if rw and rh else 0
                if sq < 0.82:
                    continue
            quad = _order_points(box)
            sides = _side_lengths(quad)
            side = float(np.mean(sides))
            # reject wildly non-square quads
            if min(sides) / max(sides) < 0.75:
                continue
            finders = _count_finder_patterns(gray, quad)
            if finders < 2:
                continue
            # score: finder count first, then size (QRs on certs are big)
            score = finders * 10 + (side / max(h, w))
            if score > best_score:
                best_score = score
                best = (quad, finders, side)
    if best is not None:
        quad, finders, side = best
        return QRRegion(quad=quad, method="contour", side_px=side,
                        score=min(1.0, finders / 3.0))
    return None


# ---------------------------------------------------------------------------
# Detector 3b: finder-pattern locator (works WITHOUT a black frame)
# ---------------------------------------------------------------------------
def _chain_depth(i: int, hier: np.ndarray, need: int = 2) -> bool:
    """True if contour `i` encloses >=`need` levels of descendants. A finder
    pattern's outer black ring contains a white gap then the black centre, so
    its first-child chain reaches depth 2 (the gap can merge with background,
    which is why depth 4 is never seen on real scans)."""
    cur = hier[i][2]
    d = 0
    while cur != -1:
        d += 1
        if d >= need:
            return True
        cur = hier[cur][2]
    return False


def _find_finder_centers(gray: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Locate QR finder patterns (the 3 corner 'nested squares'). Returns a
    list of (centre[x,y], outer-ring size px). Works for framed OR
    frameless/plain QRs, since finders are always present."""
    h, w = gray.shape[:2]
    found: list[tuple[np.ndarray, float]] = []

    for thresh in (80, 110, 140, 170, 200):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # No morphology here: closing merges the thin rings of tiny
        # (sub-100px) QRs and destroys the nesting we are looking for.
        contours, hier = cv2.findContours(bw, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]
        cands = []
        for i, c in enumerate(contours):
            x, y, ww, hh = cv2.boundingRect(c)
            if ww == 0 or hh == 0:
                continue
            side_min = float(min(h, w))
            if not (side_min * 0.012 < ww < side_min * 0.35 and
                    side_min * 0.012 < hh < side_min * 0.35):
                continue
            if not (0.65 < ww / float(hh) < 1.5):
                continue
            area = cv2.contourArea(c)
            if area / float(ww * hh) < 0.5:
                continue
            if not _chain_depth(i, hier, need=2):
                continue
            cands.append((x + ww / 2.0, y + hh / 2.0, (ww + hh) / 2.0))
        # within each threshold, collapse nested/contained candidates to the
        # OUTERMOST ring (largest), since the centre dot also appears.
        cands.sort(key=lambda t: -t[2])
        for cx, cy, size in cands:
            if not any(np.hypot(cx - fc[0][0], cy - fc[0][1]) < fc[1] * 0.7
                       for fc in found):
                found.append((np.array([cx, cy], dtype=np.float32),
                              float(size)))
    return found


def _quad_from_finders(finders: list[tuple[np.ndarray, float]]):
    """Given >=3 finder centres, build the QR outer quad (incl. quiet zone)."""
    n = len(finders)
    best = None
    best_err = 1e9
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                A, rA = finders[a]   # candidate elbow (TL)
                B, _ = finders[b]    # candidate TR
                C, _ = finders[c]    # candidate BL
                vx = B - A
                vy = C - A
                lx = float(np.linalg.norm(vx))
                ly = float(np.linalg.norm(vy))
                if lx < 5 or ly < 5:
                    continue
                if min(lx, ly) / max(lx, ly) < 0.6:
                    continue
                cosang = float(np.dot(vx, vy) / (lx * ly))
                if abs(cosang) > 0.45:   # want ~90 deg
                    continue
                ux = vx / lx
                uy = vy / ly
                r = float(np.mean([fr[1] for fr in (finders[a], finders[b],
                                                    finders[c])]))
                module = r / 7.0
                ext = 7.5 * module      # ring centre (3.5 mod) + quiet (4 mod)
                TL = A - ext * (ux + uy)
                TR = TL + ux * (lx + 2 * ext)
                BL = TL + uy * (ly + 2 * ext)
                BR = TL + ux * (lx + 2 * ext) + uy * (ly + 2 * ext)
                err = abs(cosang) + abs(lx - ly) / max(lx, ly)
                if err < best_err:
                    best_err = err
                    best = _order_points(np.array([TL, TR, BR, BL],
                                                  dtype=np.float32))
    return best


def _detect_by_finders(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    finders = _find_finder_centers(gray)
    if len(finders) < 3:
        return None
    quad = _quad_from_finders(finders)
    if quad is None:
        return None
    sides = _side_lengths(quad)
    side = float(np.mean(sides))
    if min(sides) < max(8.0, 0.04 * min(h, w)):
        return None
    if min(sides) / max(sides) < 0.7:
        return None
    return QRRegion(quad=quad, method="finders", side_px=side, score=0.95)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_qr(img: np.ndarray, model_dir: str = "models/wechat",
              use_cnn: bool = True):
    """
    Find the old QR on a certificate.

    Returns QRRegion or None.
    """
    if use_cnn:
        quad = _detect_wechat(img, model_dir)
        if quad is not None:
            sides = _side_lengths(quad)
            return QRRegion(quad=quad, method="wechat-cnn",
                            side_px=float(np.mean(sides)), score=0.99)

    quad = _detect_opencv(img)
    if quad is not None:
        sides = _side_lengths(quad)
        return QRRegion(quad=quad, method="opencv",
                        side_px=float(np.mean(sides)), score=0.9)

    reg = _detect_by_contours(img)
    if reg is not None:
        return reg

    # last resort: locate the three finder patterns (no frame needed)
    return _detect_by_finders(img)


# ---------------------------------------------------------------------------
# Fallback placement (used when a QR cannot be located on one cert in a batch
# of SAME-TEMPLATE certificates -- the QR is always in the same spot).
# ---------------------------------------------------------------------------
def normalize_region(region: QRRegion, h: int, w: int) -> np.ndarray:
    """Quad coordinates as fractions of image width/height."""
    q = region.quad.astype(np.float32).copy()
    q[:, 0] /= float(w)
    q[:, 1] /= float(h)
    return q


def fallback_region_from_regions(img: np.ndarray,
                                 regions: list) -> QRRegion | None:
    """Place the new QR using the MEDIAN normalized position of the QRs that
    WERE detected on other certificates in the batch. Works because all
    certificates share one template. `regions` = list of (QRRegion, h, w).

    Skips images whose aspect ratio differs from the certificates (e.g. a wide
    16:9 screenshot that isn't a certificate) so a QR is never pasted onto a
    non-certificate image."""
    h, w = img.shape[:2]
    tmpl_aspect = float(np.median([ww / max(1, hh) for (_r, hh, ww) in regions]))
    if abs((w / max(1, h)) - tmpl_aspect) / tmpl_aspect > 0.15:
        return None
    norms = [normalize_region(r, hh, ww) for (r, hh, ww) in regions]
    if not norms:
        return None
    med = np.median(np.stack(norms, axis=0), axis=0).astype(np.float32)
    quad = med.copy()
    quad[:, 0] *= float(w)
    quad[:, 1] *= float(h)
    quad = _order_points(quad)
    sides = _side_lengths(quad)
    return QRRegion(quad=quad, method="batch-template-fallback",
                    side_px=float(np.mean(sides)), score=0.5)


def fallback_region_default(img: np.ndarray) -> QRRegion | None:
    """Last-resort placement for the TGMFC certificate template (QR sits in
    the upper-left, under the red certificate number). Coordinates are
    fractions of (width, height). Returns None for images whose aspect ratio
    is a wide landscape (e.g. a 16:9 screenshot) rather than a certificate
    page, so a QR is never pasted onto the wrong image."""
    h, w = img.shape[:2]
    if w / max(1, h) >= 1.45:      # wide landscape screenshot, not a cert page
        return None
    n = np.array([[0.095, 0.345], [0.205, 0.345],
                  [0.205, 0.515], [0.095, 0.515]], dtype=np.float32)
    quad = n.copy()
    quad[:, 0] *= float(w)
    quad[:, 1] *= float(h)
    quad = _order_points(quad)
    sides = _side_lengths(quad)
    return QRRegion(quad=quad, method="template-default-fallback",
                    side_px=float(np.mean(sides)), score=0.3)


def draw_debug(img: np.ndarray, region: QRRegion) -> np.ndarray:
    """Overlay the detected quad on an image (for review of failures)."""
    out = img.copy()
    q = region.quad.astype(np.int32)
    cv2.polylines(out, [q], True, (0, 0, 255), 3)
    cv2.putText(out, f"{region.method} {region.side_px:.0f}px",
                (q[0][0], max(15, q[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
