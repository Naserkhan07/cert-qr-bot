"""
payload.py
----------
Decide what text/URL each certificate's NEW QR code encodes.

Supported modes (set in config / notebook):

  url   : base_url + slug + extension      -> e.g.
          https://<your-username>.github.io/certificates/031-2024-25.png
          (this is the recommended mode: host the OUTPUT images online --
           e.g. GitHub Pages / your server / cloud storage -- and scanning the
           QR opens that certificate in the phone's browser)

  csv   : read a spreadsheet with columns: filename, qr_data
          (per-certificate exact URL or text)

  text  : fixed prefix + slug

  copy  : copy whatever the old QR decoded to (requires the old QR to be
          readable; falls back to url mode when it isn't)

`slug` defaults to the certificate file's stem (filename without extension).
"""

from __future__ import annotations

import csv
import os


def slug_from_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def safe_slug(path: str) -> str:
    """URL/filename-safe stem. Replaces spaces, parentheses and other chars that
    break URLs (e.g. 'image (1)' -> 'image_1') so the QR link and the saved
    output file always match and resolve."""
    import re
    stem = slug_from_filename(path)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "certificate"


def build_payload(path: str,
                  mode: str = "url",
                  base_url: str = "",
                  extension: str = ".png",
                  text_prefix: str = "",
                  csv_map: dict[str, str] | None = None,
                  old_data: str | None = None) -> str:
    slug = safe_slug(path)

    if mode == "csv":
        if csv_map and (os.path.basename(path) in csv_map or slug in csv_map):
            return csv_map.get(os.path.basename(path)) or csv_map[slug]
        # fall back to URL if the cert is missing from the spreadsheet
        mode = "url"

    if mode == "copy" and old_data:
        return old_data

    if mode == "text":
        return f"{text_prefix}{slug}"

    # default: url
    base = base_url.rstrip("/") + "/"
    return f"{base}{slug}{extension}"


def load_csv_map(csv_path: str) -> dict[str, str]:
    """Read filename->qr_data mapping from a CSV (headers: filename,qr_data)."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or row.get("file") or "").strip()
            data = (row.get("qr_data") or row.get("url") or
                    row.get("payload") or "").strip()
            if fn and data:
                mapping[fn] = data
                mapping[os.path.splitext(fn)[0]] = data
    return mapping
