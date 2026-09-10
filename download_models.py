#!/usr/bin/env python3
"""
download_models.py
------------------
Download the WeChat QR-code CNN detector (used as the 1st-stage AI detector)
from the OFFICIAL Hugging Face model repo:

    https://huggingface.co/opencv/qrcode_wechatqrcode

Files are saved to  models/wechat/  with the canonical names OpenCV expects:
    detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel

Falls back to the OpenCV 3rdparty GitHub mirror if Hugging Face is unreachable.

The WeChat model is a tiny (~1 MB) CPU CNN -- it downloads in seconds and runs
fine on Kaggle CPU (a GPU is not required for this step; the pipeline will use
whatever accelerator you have but this model is CPU-only by design).
"""

import os
import shutil
import urllib.request

MODEL_DIR = os.environ.get("WECHAT_MODEL_DIR", "models/wechat")
HF_REPO = "opencv/qrcode_wechatqrcode"

# (filename in the HF repo, canonical local filename)
FILES = [
    ("detect_2021nov.prototxt",  "detect.prototxt"),
    ("detect_2021nov.caffemodel", "detect.caffemodel"),
    ("sr_2021nov.prototxt",      "sr.prototxt"),
    ("sr_2021nov.caffemodel",    "sr.caffemodel"),
]

GH_BASE = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
           "wechat_qrcode_20210119/")
GH_FILES = {
    "detect.prototxt": "detect.prototxt",
    "detect.caffemodel": "detect.caffemodel",
    "sr.prototxt": "sr.prototxt",
    "sr.caffemodel": "sr.caffemodel",
}


def _download(url: str, dst: str):
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def from_huggingface(dst_dir: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed; trying direct HTTPS.")
        return False
    ok = True
    for hf_name, local_name in FILES:
        dst = os.path.join(dst_dir, local_name)
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename=hf_name)
            shutil.copy(p, dst)
            print(f"  HF {hf_name} -> {dst} ({os.path.getsize(dst)} bytes)")
        except Exception as e:
            print(f"  HF download failed for {hf_name}: {e}")
            ok = False
    return ok


def from_github(dst_dir: str) -> bool:
    ok = True
    for local_name in GH_FILES:
        dst = os.path.join(dst_dir, local_name)
        try:
            _download(GH_BASE + GH_FILES[local_name], dst)
            print(f"  GH -> {dst} ({os.path.getsize(dst)} bytes)")
        except Exception as e:
            print(f"  GitHub download failed for {local_name}: {e}")
            ok = False
    return ok


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading WeChat QR CNN model to {MODEL_DIR}/ ...")
    if from_huggingface(MODEL_DIR):
        print("Model downloaded from Hugging Face (opencv/qrcode_wechatqrcode).")
        return
    print("Falling back to GitHub mirror ...")
    if from_github(MODEL_DIR):
        print("Model downloaded from GitHub mirror.")
        return
    raise SystemExit(
        "Could not download the WeChat model. The pipeline still works "
        "without it (OpenCV + geometric fallback), but for best detection "
        "download the 4 files manually into models/wechat/ -- see README.")


if __name__ == "__main__":
    main()
