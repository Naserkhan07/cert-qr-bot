#!/usr/bin/env python3
"""Build kaggle/Cert_QR_Bot.ipynb from the tested source files.

The notebook is self-contained: it installs deps, downloads the WeChat QR CNN
model from Hugging Face, writes the pipeline modules into /kaggle/working,
runs the batch over the attached image dataset, and produces a ZIP.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "Cert_QR_Bot.ipynb")

SRC_FILES = ["__init__.py", "detector.py", "qr_generator.py", "qr_art.py",
             "replacer.py", "payload.py", "ai_extract.py",
             "verification_page.py", "ai_inpaint.py", "pipeline.py"]

src_map = {}
for f in SRC_FILES:
    with open(os.path.join(ROOT, "src", f), "r", encoding="utf-8") as fh:
        src_map[f] = fh.read()

with open(os.path.join(ROOT, "download_models.py"), "r", encoding="utf-8") as fh:
    download_models_src = fh.read()

with open(os.path.join(ROOT, "publish_to_github_pages.py"),
          "r", encoding="utf-8") as fh:
    publisher_src = fh.read()
PUBLISHER_SRC = json.dumps(publisher_src)[1:-1]  # escaped body of a str literal

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})

# ---------------------------------------------------------------- title
md("""# 🔁 Certificate QR Replacement Bot (batch, ~200 certificates)

**What it does** — for every certificate image in your Kaggle dataset:

1. **Detects** the existing QR code — AI CNN from Hugging Face
   (`opencv/qrcode_wechatqrcode`) → OpenCV → geometric finder (works even when
   the old QR is a decorative/non-standard pattern that cannot be decoded).
2. **Generates** a brand-new, crisp QR code whose payload is a **URL pointing
   to that certificate's image** (so scanning it shows the certificate).
3. **Replaces** the old QR with an **AI image-generator (Hugging Face diffusion
   inpainting — Stable Diffusion XL Inpainting)** driven by the natural-language
   prompt *"generate this certificate, just replace the old QR code with the new
   QR code; do not change any design, information, face or anything else"*. The
   model is masked to the QR square (everything else is provably untouched), and
   the **real QR is composited on top so it is guaranteed to scan** — a diffusion
   model drawing a QR from text would otherwise hallucinate an invalid code.
   Set `USE_AI=False` to use the pure geometric warp instead.
4. **Verifies** the new QR scans (auto-decodes the output).
5. **Zips** all finished certificates.

> ⚙️ **Accelerator note:** with `USE_AI=True` choose **GPU (T4/P100)** — the
> inpainting model needs it; the WeChat QR CNN itself is a tiny CPU model.
> `USE_AI=False` runs on CPU in well under a minute for 200 images. Either way
> the finished image keeps the original design byte-for-byte outside the QR box.

**Inputs:** attach a Kaggle *Dataset* containing your certificate images
(any folder structure, `.png/.jpg/.jpeg/.webp`).
""")

# ---------------------------------------------------------------- config
md("## ⚙️ 1. Configuration\n`BASE_URL` is already set to your **GitHub Pages certificate hosting** site (`https://naserkhan07.github.io/certificate-qr/img/`). Each QR encodes `BASE_URL + filename`; section 8 publishes the finished ZIP there (one token). If you use another host, just change `BASE_URL`.")
code('''# ==== EDIT THESE ====
BASE_URL = "https://naserkhan07.github.io/certificate-qr/"   # QR -> per-candidate verification page
PAYLOAD_MODE = "url"        # "url" | "csv" | "text" | "copy"
CSV_PATH = None             # set to a CSV path if PAYLOAD_MODE="csv" (cols: filename,qr_data)
TEXT_PREFIX = ""            # used if PAYLOAD_MODE="text"
REPLACE_MODE = "full_replace"  # "full_replace" (recommended) | "keep_frame"
QR_COVER_SCALE = 1.08       # grow the white QR box ~8% to fully hide the old
                            # black frame / any edge halo. 1.0 = exact old size.
QR_ERROR_CORRECTION = "M"   # L | M | Q | H
OUTPUT_EXT = ".png"         # ".png" (lossless, recommended) | ".jpg"

# ---- AI that READS the candidate details off the certificate (no CSV/OCR) ----
# A Hugging Face vision-language model looks at each certificate image and
# extracts: Name, Father/Mother Name, Aadhar No., Course Name, Course Date and
# From (institute). Needs a GPU (set Settings -> Accelerator -> GPU T4).
# Qwen2-VL-2B is a strong, small document reader with NO num2words/tesseract
# dependency and it fits a free T4. Optional stronger/larger alternative:
# "Qwen/Qwen2.5-VL-3B-Instruct".
AI_EXTRACT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# ---- QR code GENERATOR backend ----
# "standard"  = plain black/white QR, 100% deterministic & always scans
#               (RECOMMENDED for official certificates).
# "artistic"  = Hugging Face "QR Code Monster" ControlNet (AI image->QR model):
#               generates a stylised QR, auto-checks it decodes to the right
#               URL over several seeds, and falls back to "standard" if none
#               scan. Needs GPU. Colourful art-QRs are NOT typical on govt docs.
QR_BACKEND = "standard"
ART_CONTROLNET_SCALE = 1.5  # higher = more readable (1.2-1.8), lower = more art
ART_TRIES = 6               # seed variants tried before falling back
ART_SIZE = 768

# ---- AI image generator (Hugging Face diffusion inpainting) ----
# Leave False for the fast, guaranteed run. Set True ONLY with GPU (T4/P100) +
# Internet on; it then uses the SDXL inpainting model for the QR region and
# auto-falls back to the exact warp if no GPU/model is available.
USE_AI = False
AI_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
AI_STEPS = 30               # fewer = faster (try 20), more = cleaner
AI_GUIDANCE = 8.0
AI_SEED = 0
AI_MODE = "inpaint_then_paste"   # keep this -> scannable. "generative_only" is experiment-only

# ==== input dataset (auto-detected; override if you have multiple) ====
import glob, os
CANDIDATES = []
for pat in ["/kaggle/input/**/*.png", "/kaggle/input/**/*.jpg",
            "/kaggle/input/**/*.jpeg", "/kaggle/input/**/*.webp"]:
    CANDIDATES += glob.glob(pat, recursive=True)
print("Found", len(CANDIDATES), "images under /kaggle/input")
for p in CANDIDATES[:10]:
    print("  ", p)
''')

# ---------------------------------------------------------------- install
md("## 📦 2. Install dependencies")
code('''# Remove any pre-installed OpenCV (Kaggle ships several; mixing the GUI and
# headless/contrib wheels breaks `cv2`), then install the exact contrib build.
%pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null
%pip install -q "opencv-contrib-python-headless==4.10.0.84" qrcode huggingface_hub requests pillow
# AI: vision-language model that READS the certificate fields (always used),
# plus diffusers for the optional AI QR/painting paths. (Qwen2-VL: no num2words.)
%pip install -q "transformers>=4.46" diffusers accelerate safetensors torchvision
import cv2
print("opencv:", cv2.__version__, "| wechat CNN module:",
      hasattr(cv2, "wechat_qrcode_WeChatQRCode"))
print("done")''')

# ---------------------------------------------------------------- download model
md("## 🤗 3. Download the AI QR-detection model from Hugging Face\nDownloads `opencv/qrcode_wechatqrcode` (WeChat CV QR detector + super-resolution CNN). Falls back to the OpenCV GitHub mirror.")
code('''import os
os.makedirs("/kaggle/working/certbot/models/wechat", exist_ok=True)
_cwd = os.getcwd(); os.chdir("/kaggle/working/certbot")
MODEL_DIR = "/kaggle/working/certbot/models/wechat"
''' + download_models_src.replace(
    'MODEL_DIR = os.environ.get("WECHAT_MODEL_DIR", "models/wechat")',
    'MODEL_DIR = os.environ.get("WECHAT_MODEL_DIR", "/kaggle/working/certbot/models/wechat")')
+ '''
main()
os.chdir(_cwd)
import cv2
print("opencv:", cv2.__version__, "| wechat module:",
      hasattr(cv2, "wechat_qrcode_WeChatQRCode"))
''')

# ---------------------------------------------------------------- write source
md("## 🧠 4. Write the pipeline modules")
src_repr = json.dumps(src_map, indent=2)
writer_code = '''import os, shutil
os.makedirs("/kaggle/working/certbot/src", exist_ok=True)
# clear stale bytecode from previous runs
_pyc = "/kaggle/working/certbot/src/__pycache__"
if os.path.isdir(_pyc): shutil.rmtree(_pyc)
SRC = ''' + src_repr + '''
for name, content in SRC.items():
    with open(os.path.join("/kaggle/working/certbot/src", name), "w",
              encoding="utf-8") as f:
        f.write(content)
    print("wrote", name, len(content), "chars")
print("pipeline has qr_cover_scale:", "qr_cover_scale" in SRC["pipeline.py"])

# publish helper (GitHub Pages)
with open("/kaggle/working/certbot/publish_to_github_pages.py", "w",
          encoding="utf-8") as f:
    f.write("''' + PUBLISHER_SRC.replace('"""', '\\"\\"\\"') + '''")
print("wrote publish_to_github_pages.py")
'''
code(writer_code)

# Guard prepended to the cells that import src.*: if section 4 never ran in
# this kernel/session (fresh container, or the user runs 4b / 5 alone), the
# package folder won't exist -> ModuleNotFoundError. Auto-write it instead.
import textwrap
ENSURE_WRITER = (
    'import os as _os\n'
    'if not _os.path.isfile("/kaggle/working/certbot/src/ai_extract.py"):\n'
    '    print("Pipeline modules not written in this session yet — '
    'auto-running the section 4 writer first ...")\n'
    + textwrap.indent(writer_code, "    ")
)

# ------------------------------------------------ test AI extraction
md("""## 🧪 4b. Test the AI field-reader on ONE certificate

Run this **before** the batch to confirm the vision model loads on the GPU and
reads the six fields. It prints the extracted fields AND the model's raw output.
If it prints an error / empty fields here, the batch pages will show "&mdash;",
so fix this first (usually: set **Settings &rarr; Accelerator &rarr; GPU T4**,
or wait for the model to finish downloading). If this is a fresh session, just
run this cell — it auto-writes the pipeline modules even if you skipped
section 4.""")
code(ENSURE_WRITER + '''import sys, glob, json
import cv2
sys.path.insert(0, "/kaggle/working/certbot")
for _m in list(sys.modules):
    if _m == "src" or _m.startswith("src."):
        del sys.modules[_m]
from src.ai_extract import ai_extract_fields, runtime_status

print(runtime_status())
_test_imgs = [p for p in CANDIDATES if "/kaggle/working/" not in p]
if not _test_imgs:
    print("No input images found yet (run section 1 first).")
else:
    _tp = _test_imgs[0]
    print("\\nReading:", _tp)
    _img = cv2.imread(_tp, cv2.IMREAD_COLOR)
    print("image size:", None if _img is None else _img.shape)
    _fields = ai_extract_fields(_img, model_id=AI_EXTRACT_MODEL)
    print("\\n----- EXTRACTED FIELDS -----")
    for _k in ["name", "parent", "relation", "aadhar", "course",
               "date_from", "date_to", "institute"]:
        print(f"  {_k:10s}: {_fields.get(_k, '')}")
    if _fields.get("_error"):
        print("\\n❌ ERROR:", _fields["_error"])
    if _fields.get("_raw_ai"):
        print("\\n----- RAW MODEL OUTPUT -----")
        print(_fields["_raw_ai"][:1500])
''')

# ---------------------------------------------------------------- run
md("## ▶️ 5. Run the batch")
code(ENSURE_WRITER + '''import sys, os, shutil

# Free any vision model the section 4b test cell left on the GPU, then force
# Python to pick up the freshly written modules (avoids CUDA OOM / stale code).
try:
    from src.ai_extract import release_model
    release_model()
except Exception:
    pass
for _m in list(sys.modules):
    if _m == "src" or _m.startswith("src."):
        del sys.modules[_m]
try:
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
except Exception:
    pass

sys.path.insert(0, "/kaggle/working/certbot")
from src.pipeline import Config, run

# Build an "input" folder view of the dataset images (non-destructive: copies)
import shutil
INPUT_DIR = "/kaggle/working/certbot/input"
OUTPUT_DIR = "/kaggle/working/certbot/output"
if os.path.exists(INPUT_DIR): shutil.rmtree(INPUT_DIR)
os.makedirs(INPUT_DIR, exist_ok=True)
for p in CANDIDATES:
    # skip anything inside our own working dir
    if "/kaggle/working/" in p: continue
    shutil.copy(p, os.path.join(INPUT_DIR, os.path.basename(p)))

try:
    import torch
    print("torch:", torch.__version__, "| CUDA GPU:", torch.cuda.is_available())
    if USE_AI and not torch.cuda.is_available():
        print("⚠️ USE_AI=True but no GPU — set the notebook accelerator to GPU "
              "(Settings → Accelerator → GPU P100/T4). Falling back to the "
              "deterministic warp if the model cannot run.")
except Exception as e:
    print("torch unavailable:", e)

from src.ai_inpaint import DEFAULT_PROMPT
print("AI prompt used:\\n", DEFAULT_PROMPT)

cfg = Config(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    model_dir=MODEL_DIR,
    zip_path="/kaggle/working/certificates_with_new_qr.zip",
    payload_mode=PAYLOAD_MODE,
    base_url=BASE_URL,
    csv_path=CSV_PATH,
    text_prefix=TEXT_PREFIX,
    replace_mode=REPLACE_MODE,
    qr_cover_scale=QR_COVER_SCALE,
    qr_error_correction=QR_ERROR_CORRECTION,
    output_ext=OUTPUT_EXT,
    ai_extract_model=AI_EXTRACT_MODEL,
    qr_backend=QR_BACKEND,
    art_controlnet_scale=ART_CONTROLNET_SCALE,
    art_tries=ART_TRIES,
    art_size=ART_SIZE,
    use_cnn=True,
    save_debug=True,
    use_ai=USE_AI,
    ai_model=AI_MODEL,
    ai_steps=AI_STEPS,
    ai_guidance=AI_GUIDANCE,
    ai_seed=AI_SEED,
    ai_mode=AI_MODE,
)
summary = run(cfg)
''')

# ---------------------------------------------------------------- inspect
md("## 🔍 6. Review the manifest & spot-check an output")
code('''import pandas as pd
manifest = pd.read_csv(os.path.join(OUTPUT_DIR, "manifest.csv"))
display(manifest)
print("Counts:\\n", manifest["status"].value_counts())
if (manifest["status"] != "ok").any():
    print("\\n⚠️ Files needing a quick look (debug overlays in output/_debug/):")
    print(manifest[manifest["status"] != "ok"][["filename","status","note"]])''')
code('''# Visually spot-check the first output certificate
from PIL import Image
import glob
outs = sorted(glob.glob(OUTPUT_DIR + "/*" + OUTPUT_EXT))
if outs:
    im = Image.open(outs[0])
    print("Showing:", outs[0], im.size)
    display(im)''')

# ---------------------------------------------------------------- zip
md("## 🗜️ 7. Get the ZIP\nThe zip is at `/kaggle/working/certificates_with_new_qr.zip` — download it from the Kaggle **Output** panel (it also persists as the notebook output).")
code('''import zipfile, os
zp = "/kaggle/working/certificates_with_new_qr.zip"
print("zip exists:", os.path.exists(zp), "size:",
      round(os.path.getsize(zp)/1e6, 2), "MB")
with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    print("files in zip:", len(names))
    print("\\n".join(names[:10]), "..." if len(names) > 10 else "")''')

# ---------------------------------------------------------------- publish
md("""## 🚀 8. Auto-upload every certificate (so scanning opens it)

The QR stores a URL — the finished image must be uploaded for the link to
resolve. This cell uploads **all** certificates from the ZIP to your Pages
repo and verifies each one is live; a scan then opens that exact certificate.

One-time setup on Kaggle:
1. Create a token: https://github.com/settings/tokens → **Generate new token
   (classic)** → tick **`repo`** → copy it.
2. **Add-ons → Secrets → Add a new secret** named `GITHUB_TOKEN`, paste it.

The hosting repo (`certificate-qr`) is kept empty/clean and serves only the
certificate images. ⚠️ It is public — see section 9 for private options.
""")
code(ENSURE_WRITER + '''# AUTO-PUBLISH: upload every finished certificate to the public GitHub Pages
# repo so each QR link opens the certificate. Each upload is verified live.
PUBLISH = True       # keep True so all certs are uploaded (scanning then works)
if PUBLISH:
    import os, sys
    token = ""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("❌ GITHUB_TOKEN secret not found. Add it once: Kaggle Add-ons ->")
        print("   Secrets -> add secret named GITHUB_TOKEN (a classic GitHub")
        print("   token with the 'repo' scope from https://github.com/settings/tokens).")
        print("   Then re-run this cell. Certificates were still saved to the ZIP.")
    else:
        os.environ["GITHUB_TOKEN"] = token
        sys.path.insert(0, "/kaggle/working/certbot")
        _pub = open("/kaggle/working/certbot/publish_to_github_pages.py").read()
        _pub = _pub.replace('if __name__ == "__main__":\\n    main()', "")
        g = {"__name__": "publisher"}
        exec(compile(_pub, "publish_to_github_pages.py", "exec"), g)
        _res = g["publish"](
            zip_path="/kaggle/working/certificates_with_new_qr.zip",
            owner="Naserkhan07", repo="certificate-qr",
            token=token, subdir="img", branch="main")
        if _res["fail"] == 0:
            print("\\n✅ All certificates live. Scanning any QR now opens the")
            print("   verification page at https://naserkhan07.github.io/certificate-qr/<file>.html")
        else:
            print(f"\\n⚠️ {_res['fail']} file(s) failed verification - re-run this cell.")
else:
    print("PUBLISH=False: certificates only saved to the ZIP (QR links won't resolve).")
''')

# ---------------------------------------------------------------- hosting
md("""## 🌐 9. Notes / other hosts

The QR encodes a **URL** — hosting is unavoidable (this is exactly how every
"image QR" website works; they just host the file on *their* server). The
publisher above puts them on your own GitHub Pages.

Other hosts work too: your own server, S3/Cloudflare R2, Netlify, or image-host
APIs (imgbb/Cloudinary). For **private** data (Aadhaar), use a private host and
`PAYLOAD_MODE="csv"` with a `filename,qr_data` mapping of direct links.

### Tuning
- **QR won't scan on paper?** It's verified digitally; for very small print
  raise `QR_ERROR_CORRECTION="H"` and print at ≥ ~150 DPI.
- **Keep the old black frame line?** Set `REPLACE_MODE="keep_frame"`.
- **Old QR never detected?** Check `output/_debug/FAILED_*` overlays. Detection
  now uses WeChat CNN → OpenCV → frame contours → finder patterns, so it works
  for framed AND frameless QRs.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "with", len(cells), "cells")
