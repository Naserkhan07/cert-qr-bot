"""
pipeline.py
-----------
End-to-end batch workflow:

  for every certificate image in an input folder:
     1. DETECT  the old QR  (WeChat CNN model from Hugging Face -> OpenCV ->
                geometric contour finder)
     2. BUILD   the payload (URL/text/CSV that scanning should open)
     3. GENERATE a brand-new crisp QR code
     4. REPLACE the old QR with the new one at the exact same place/size/angle
                (nothing else on the certificate is touched)
     5. VERIFY  the new QR is actually decodable on the output image
     6. SAVE    the new certificate

  then: write a manifest.csv and ZIP all outputs.

Runs on CPU or GPU identically (the geometry work is CPU; the CNN detector is
tiny). Designed for 200+ certificates in one pass on Kaggle.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import (detect_qr, draw_debug,
                       fallback_region_from_regions, fallback_region_default)
from .payload import build_payload, load_csv_map, slug_from_filename, safe_slug
from .qr_generator import build_qr
from .qr_art import build_artistic_qr
from .replacer import replace_qr
from .ai_extract import ai_extract_fields
from .verification_page import build_page
from .ai_inpaint import (
    ai_replace_qr, ai_available, DEFAULT_PROMPT, DEFAULT_NEGATIVE,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class Result:
    filename: str
    status: str                       # "ok" | "recheck" | "failed"
    payload: str = ""
    method: str = ""
    qr_side_px: float = 0.0
    verified: bool = False
    ai_used: bool = False
    out_name: str = ""
    slug: str = ""
    fields: dict = None
    extract_method: str = ""
    note: str = ""


@dataclass
class Config:
    input_dir: str = "input"
    output_dir: str = "output"
    model_dir: str = "models/wechat"
    zip_path: str = "certificates_with_new_qr.zip"
    payload_mode: str = "url"          # url | text | csv | copy
    base_url: str = "https://example.github.io/certificates/"
    url_extension: str = ".png"
    # QR links to a per-candidate VERIFICATION PAGE (.html) that shows the
    # certificate image with the candidate details (read off the certificate by
    # the AI vision model) beneath it. False -> QR links directly to the image.
    verification_page: bool = True
    # AI vision-language model that READS every field off the certificate image
    # (no OCR / no CSV). Requires a GPU on Kaggle (T4). Empty fields if it fails.
    ai_extract_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    text_prefix: str = ""
    csv_path: str | None = None
    qr_error_correction: str = "M"
    qr_box_size: int = 20
    qr_border: int = 4
    replace_mode: str = "full_replace"  # full_replace | keep_frame
    qr_cover_scale: float = 1.08        # >1 = white box grows to hide old frame
    output_ext: str = ".png"
    jpeg_quality: int = 95
    save_debug: bool = True
    use_cnn: bool = True
    # ---- QR generator backend ----
    # "standard"  -> plain deterministic qrcode (always scans; recommended for
    #                official certificates)
    # "artistic"  -> HF QR Code Monster ControlNet (AI art-QR; auto-verified and
    #                falls back to standard if no seed scans). GPU recommended.
    qr_backend: str = "standard"
    art_prompt: str = ""       # "" -> qr_art.CLEAN_PROMPT
    art_controlnet_scale: float = 1.5
    art_size: int = 768
    art_tries: int = 6
    # ---- AI image-generator (diffusion inpainting) ----
    use_ai: bool = False
    ai_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ai_prompt: str = DEFAULT_PROMPT
    ai_negative: str = DEFAULT_NEGATIVE
    ai_steps: int = 30
    ai_guidance: float = 8.0
    ai_seed: int = 0
    ai_margin: int = 120
    ai_mode: str = "inpaint_then_paste"   # inpaint_then_paste | generative_only
    # If a QR can't be located on one certificate, place the new QR at the
    # batch-derived template position (works because all certs share a layout).
    fallback_placement: bool = True


def list_certificates(input_dir: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


_WECHAT_DECODER = None


def _get_wechat_decoder(model_dir: str):
    """A WeChat CNN decoder is far stronger on small/dense QRs (used both to
    read old QRs and to verify new ones)."""
    global _WECHAT_DECODER
    if _WECHAT_DECODER is not None:
        return _WECHAT_DECODER
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return None
    paths = [os.path.join(model_dir, f) for f in
             ("detect.prototxt", "detect.caffemodel",
              "sr.prototxt", "sr.caffemodel")]
    if not all(os.path.exists(p) for p in paths):
        return None
    try:
        _WECHAT_DECODER = cv2.wechat_qrcode_WeChatQRCode(*paths)
    except Exception:
        _WECHAT_DECODER = None
    return _WECHAT_DECODER


def _decode_any_qr(img: np.ndarray, model_dir: str = "models/wechat") -> str:
    """Best-effort decode to VERIFY the new QR scans. Returns text or ''.

    Tries the strong WeChat CNN decoder first (it reads small/dense QRs that
    OpenCV's classic decoder misses), then OpenCV ArUco/classic at several
    scales.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    wd = _get_wechat_decoder(model_dir)
    if wd is not None:
        for scale in (1, 2, 3):
            src = gray if scale == 1 else cv2.resize(
                gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            try:
                res, _pts = wd.detectAndDecode(src)
                if len(res) > 0 and res[0]:
                    return res[0]
            except Exception:
                pass

    dets = []
    if hasattr(cv2, "QRCodeDetectorAruco"):
        try:
            dets.append(cv2.QRCodeDetectorAruco())
        except Exception:
            pass
    dets.append(cv2.QRCodeDetector())
    for scale in (1, 2, 3, 4):
        src = gray if scale == 1 else cv2.resize(
            gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        for d in dets:
            try:
                data, _, _ = d.detectAndDecode(src)
            except Exception:
                data = ""
            if data:
                return data
    return ""


def process_one(path: str, cfg: Config,
                csv_map: dict | None = None,
                region_override=None) -> tuple[np.ndarray | None, Result]:
    name = os.path.basename(path)
    slug = safe_slug(path)
    res = Result(filename=name, status="failed", slug=slug,
                 out_name=slug + cfg.output_ext)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        res.note = "could not read image"
        return None, res

    # 0) The AI vision model READS all candidate details off the certificate
    #    image (name, parent, aadhar, course, dates, institute). No OCR / CSV.
    fields = {}
    extract_used = ""
    try:
        fields = ai_extract_fields(img, model_id=cfg.ai_extract_model)
        if fields and not fields.get("_error"):
            extract_used = "AI-VLM"
        elif fields.get("_error"):
            extract_used = "AI-FAILED"
            print(f"   ⚠️  AI extraction failed for {name}: {fields['_error']}")
    except Exception as e:
        fields = {"_error": str(e)}
        extract_used = "AI-FAILED"
        print(f"   ⚠️  AI extraction crashed for {name}: {e}")
    res.fields = fields
    res.extract_method = extract_used

    # The QR links to the verification PAGE (.html) when enabled, else the raw
    # image (cfg.url_extension).
    link_ext = ".html" if cfg.verification_page else cfg.url_extension

    # 1) locate the old QR: detect it, or use a provided fallback placement.
    region = region_override
    if region is None:
        region = detect_qr(img, model_dir=cfg.model_dir, use_cnn=cfg.use_cnn)
    if region is None:
        res.note = "old QR not found"
        return img, res

    res.method = region.method
    res.qr_side_px = round(region.side_px, 1)

    # old payload (only needed for 'copy' mode)
    old_data = _decode_any_qr(img, cfg.model_dir) if cfg.payload_mode == "copy" else None

    # 2) payload (links to the verification page or the raw image)
    payload = build_payload(
        path, mode=cfg.payload_mode, base_url=cfg.base_url,
        extension=link_ext, text_prefix=cfg.text_prefix,
        csv_map=csv_map, old_data=old_data)
    res.payload = payload

    # 3) generate fresh QR at high resolution.
    #    Optional AI "artistic QR" backend (HF QR Code Monster ControlNet):
    #    returns an AI-styled QR that decodes to `payload`, else None -> we
    #    transparently fall back to the deterministic standard QR.
    qr_img = None
    if cfg.qr_backend == "artistic":
        from .qr_art import CLEAN_PROMPT
        qr_img = build_artistic_qr(
            payload,
            prompt=cfg.art_prompt or CLEAN_PROMPT,
            controlnet_scale=cfg.art_controlnet_scale,
            size=cfg.art_size, max_tries=cfg.art_tries)
        if qr_img is not None:
            res.note = "AI artistic QR (verified)"
    if qr_img is None:
        qr_img = build_qr(payload, box_size=cfg.qr_box_size,
                          border=cfg.qr_border,
                          error_correction=cfg.qr_error_correction)

    # 4) REPLACE the old QR.
    #    - AI path: masked diffusion inpainting (prompt-driven), then the
    #      real QR is composited so it is guaranteed scannable.
    #    - deterministic path: exact perspective warp.
    if cfg.use_ai:
        out, ai_info = ai_replace_qr(
            img, region, qr_img,
            prompt=cfg.ai_prompt, negative_prompt=cfg.ai_negative,
            model_id=cfg.ai_model, steps=cfg.ai_steps,
            guidance=cfg.ai_guidance, seed=cfg.ai_seed,
            margin=cfg.ai_margin, mode=cfg.ai_mode,
            replace_mode=cfg.replace_mode,
            cover_scale=cfg.qr_cover_scale)
        res.ai_used = bool(ai_info.get("ran"))
        if not ai_info.get("ran"):
            res.method += "+ai-fallback(warp)"
    else:
        out = replace_qr(img, region, qr_img, mode=cfg.replace_mode,
                         cover_scale=cfg.qr_cover_scale)
        ai_info = {}

    # 5) verify the new QR decodes on the output
    decoded = _decode_any_qr(out, cfg.model_dir)
    res.verified = bool(decoded)
    art_tag = "AI artistic QR; " if (cfg.qr_backend == "artistic"
                                    and "artistic" in res.note) else ""
    if not decoded:
        res.status = "recheck"
        res.note = art_tag + "new QR not auto-verified (usually still scans; check debug)"
    else:
        res.status = "ok"
        if res.ai_used:
            res.note = art_tag + "verified scannable (AI inpaint+QR)"
        else:
            res.note = art_tag + "verified scannable"

    return out, res


def run(cfg: Config) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    debug_dir = os.path.join(cfg.output_dir, "_debug")
    if cfg.save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    csv_map = None
    if cfg.payload_mode == "csv" and cfg.csv_path:
        csv_map = load_csv_map(cfg.csv_path)

    certs = list_certificates(cfg.input_dir)

    if cfg.payload_mode == "url":
        placeholder_markers = ("YOUR-USERNAME", "example.github.io",
                               "example.com", "REPLACE", "your-username")
        if any(m.lower() in (cfg.base_url or "").lower()
               for m in placeholder_markers):
            print("=" * 70)
            print("⚠️  WARNING: BASE_URL still looks like a PLACEHOLDER.")
            print(f"    base_url = {cfg.base_url}")
            print("    The QR codes WILL scan, but they point at a URL that")
            print("    does not exist yet -> your phone shows a 404 page.")
            print("    1) Set base_url to the address where you will upload the")
            print("       OUTPUT images (e.g. GitHub Pages: create a repo, turn")
            print("       on Settings->Pages, upload the finished PNGs).")
            print("    2) Re-run. See README 'Hosting' / notebook section 8.")
            print("=" * 70)

    # ---- detection pass: locate the QR on every certificate so that a cert
    # where detection fails can inherit the position from its template (all
    # certificates in a batch share a layout).
    detected: dict[str, object] = {}
    template: list = []
    for path in certs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        reg = detect_qr(img, model_dir=cfg.model_dir, use_cnn=cfg.use_cnn)
        h, w = img.shape[:2]
        detected[path] = reg
        if reg is not None and "fallback" not in reg.method:
            template.append((reg, h, w))

    results: list[Result] = []
    ok = recheck = failed = 0
    ai_ok = ai_fail = 0

    zip_path = cfg.zip_path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in certs:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            override = None
            if detected.get(path) is None and cfg.fallback_placement:
                # inherit QR position from the other certificates / template
                override = fallback_region_from_regions(img, template)
                if override is None:
                    override = fallback_region_default(img)
            out, res = process_one(path, cfg, csv_map, region_override=override)
            if res.extract_method == "AI-VLM":
                ai_ok += 1
            else:
                ai_fail += 1
            if override is not None and res.status in ("ok", "recheck"):
                res.note += f" | QR placed via {override.method}"
            results.append(res)
            if res.status == "ok":
                ok += 1
            elif res.status == "recheck":
                recheck += 1
            else:
                failed += 1

            if out is not None and res.status in ("ok", "recheck"):
                out_name = res.out_name
                out_path = os.path.join(cfg.output_dir, out_name)
                _write_image(out, out_path, cfg)
                zf.write(out_path, arcname="img/" + out_name)

                # Per-candidate VERIFICATION PAGE (self-contained HTML).
                if cfg.verification_page:
                    page = build_page(res.fields or {}, out, out_name)
                    page_name = res.slug + ".html"
                    page_path = os.path.join(cfg.output_dir, page_name)
                    with open(page_path, "w", encoding="utf-8") as fh:
                        fh.write(page)
                    zf.write(page_path, arcname=page_name)

                if cfg.save_debug and res.status == "recheck":
                    reg = detect_qr(out, model_dir=cfg.model_dir,
                                    use_cnn=False)
                    dbg = draw_debug(out, reg) if reg else out
                    cv2.imwrite(os.path.join(debug_dir, "dbg_" + out_name), dbg)
            elif out is not None and cfg.save_debug:
                cv2.imwrite(os.path.join(
                    debug_dir, "FAILED_" + safe_slug(path) + cfg.output_ext), out)

            tag = "AI" if res.ai_used else ("ai-fb" if "+ai-fallback" in res.method else "  ")
            print(f"[{res.status:7s}] {tag} {res.filename:36s} "
                  f"{res.method:24s} {res.qr_side_px:6.1f}px  {res.payload}")

    # manifest
    manifest = os.path.join(cfg.output_dir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "status", "ai_used", "method", "qr_side_px",
                     "verified", "qr_payload", "name", "parent", "aadhar",
                     "course", "date_from", "date_to", "institute",
                     "extracted_by", "note"])
        for r in results:
            f = r.fields or {}
            wr.writerow([r.filename, r.status, r.ai_used, r.method,
                         r.qr_side_px, r.verified, r.payload,
                         f.get("name", ""), f.get("parent", ""),
                         f.get("aadhar", ""), f.get("course", ""),
                         f.get("date_from", ""), f.get("date_to", ""),
                         f.get("institute", ""), r.extract_method, r.note])

    summary = {"total": len(certs), "ok": ok, "recheck": recheck,
               "failed": failed, "fields_extracted": ai_ok,
               "fields_failed": ai_fail,
               "zip": os.path.abspath(zip_path),
               "manifest": os.path.abspath(manifest)}
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:16s}: {v}")
    if ai_fail:
        print("\n⚠️  The AI could NOT read fields from "
              f"{ai_fail} certificate(s) -> those pages show '—'.")
        print("   Check: Accelerator = GPU T4; and run the 'Test AI extraction'")
        print("   cell to see the raw model output / error.")
    return summary


def _write_image(img: np.ndarray, path: str, cfg: Config):
    ext = cfg.output_ext.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    else:
        cv2.imwrite(path, img)
