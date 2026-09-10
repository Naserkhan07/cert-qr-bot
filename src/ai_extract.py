"""
ai_extract.py
-------------
Read the candidate's details DIRECTLY FROM THE CERTIFICATE IMAGE using an AI
vision-language model from Hugging Face (no OCR, no CSV, no hand-written rules).

Model (runs on a free Kaggle T4 GPU; no num2words / tesseract needed):
    Qwen/Qwen2-VL-2B-Instruct   (default) -- strong document/JSON reader
    Qwen/Qwen2.5-VL-3B-Instruct (optional) -- slightly larger, also fine on T4

The model is shown the certificate and asked to return ONLY JSON:
    name, parent, relation, aadhar, course, date_from, date_to, institute.

Failures are NEVER silent: the returned dict carries an "_error" / "_raw_ai"
key so the notebook log / manifest show exactly what happened.
"""

from __future__ import annotations

import json
import re
import os
import gc
import importlib
import subprocess
import sys

# Reduce CUDA fragmentation / OOM on the 16 GB T4 when a model is loaded more
# than once in a session (must be set before torch first allocates).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_PROMPT = (
    "You are reading an Indian training-completion certificate IMAGE. "
    "Look carefully at the whole image and extract these SIX fields. "
    "Return ONLY a JSON object (no prose, no markdown, no code fence) with "
    "exactly these keys:\n"
    '{"name": "<the candidate OWN name only: the words printed after Mr./Ms. '
    'and BEFORE the token S/o or D/o or W/o. Do NOT include S/o, D/o, W/o, or '
    'any name after them.>", '
    '"parent": "<the father/mother name ONLY: the person name printed '
    'immediately AFTER S/o (son of -> father), D/o (daughter of -> father) or '
    'W/o (wife of -> husband), up to the next comma or bracket.>", '
    '"relation": "<exactly S/o if the text says S/o, D/o if it says D/o, or '
    'W/o if it says W/o>", '
    '"aadhar": "<the candidate 12-digit Aadhaar number, digits only>", '
    '"course": "<the full course/training name the candidate completed, e.g. '
    'the text after Training in / Trainings in / Course>", '
    '"date_from": "<course START date formatted dd/mm/yyyy>", '
    '"date_to": "<course END date formatted dd/mm/yyyy>", '
    '"institute": "<the COMPLETE institute name printed anywhere on the '
    'certificate, written out in full exactly as shown (spell out every word; '
    'do NOT shorten to initials - e.g. write Falih Consultancy Services (FCS) '
    'Training Institute, not just FCS). Read the org name LETTER BY LETTER: '
    'the printed letters l, i, s and h look alike - Falih is F-a-l-i-h (NOT '
    'Falis, NOT Fahis). It differs on every certificate; never '
    'assume or default to a name.>"}\n'
    "CRITICAL: 'S/o' = Son of (FATHER), 'D/o' = Daughter of (FATHER), 'W/o' = "
    "Wife of (HUSBAND). These give the parent name. 'R/o' = Resident of (the "
    "HOME ADDRESS / place, e.g. Rajendra Nager) — R/o is NOT a person and must "
    "NEVER be put in name or parent. Read from the image only; do not invent; "
    "use \"\" for anything unreadable; dates as dd/mm/yyyy."
)

# Small helper packages (NOT torch/transformers). If one is missing in the
# kernel we self-heal by pip-installing it at runtime.
_REQUIRED = [
    ("PIL", "pillow"),
    ("accelerate", "accelerate"),
    ("torchvision", "torchvision"),
]

_MODEL = None
_PROC = None
_DEVICE = None
_MODEL_ID = None


def _pip_install(pkg: str) -> bool:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg])
        return True
    except Exception as e:
        print(f"[ai_extract] pip install {pkg} failed: {e}")
        return False


def _ensure_deps():
    # Only self-heal where torch/transformers already exist (the GPU notebook).
    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except Exception:
        return
    for import_name, pip_name in _REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception:
            print(f"[ai_extract] installing missing dependency '{pip_name}' ...")
            if _pip_install(pip_name):
                try:
                    importlib.import_module(import_name)
                except Exception:
                    pass


def device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def runtime_status() -> str:
    try:
        import torch  # noqa
    except Exception as e:
        return f"torch NOT importable: {e}"
    try:
        import transformers
        tv = transformers.__version__
    except Exception as e:
        return f"transformers NOT importable: {e}"
    dev = device()
    gpu = ""
    if dev == "cuda":
        try:
            import torch
            gpu = f" | GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
    warn = "" if dev == "cuda" else " | ⚠️ CPU only — set Accelerator to GPU T4"
    return f"torch ok | transformers {tv} | device={dev}{gpu}{warn}"


def release_model():
    """Free the cached vision model and its CUDA memory. Call before a fresh
    module load in the same session (e.g. after the test cell) to avoid OOM."""
    global _MODEL, _PROC, _MODEL_ID
    _MODEL = None
    _PROC = None
    _MODEL_ID = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _load(model_id: str):
    """Load (once) and cache the Qwen2-VL model + processor. Frees any stale
    model first and retries once after emptying the CUDA cache on OOM."""
    global _MODEL, _PROC, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL, _PROC
    release_model()  # ensure no previous model (e.g. from the test cell) lingers
    import torch
    from transformers import AutoProcessor

    def _build():
        ModelCls = None
        if "Qwen2.5" in model_id:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
            except Exception:
                ModelCls = None
        if ModelCls is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration as ModelCls
            except Exception:
                ModelCls = None
        if ModelCls is None:
            from transformers import AutoModelForImageTextToText as ModelCls
        dtype = torch.float16 if device() == "cuda" else torch.float32
        proc = AutoProcessor.from_pretrained(model_id)
        kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
        if device() == "cuda":
            kwargs.update(device_map="auto", offload_folder="/kaggle/working/offload")
        model = ModelCls.from_pretrained(model_id, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(device())
        model.eval()
        return model, proc

    try:
        _MODEL, _PROC = _build()
    except (RuntimeError, MemoryError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print("[ai_extract] GPU out of memory while loading — freeing cache "
                  "and retrying once...")
            release_model()
            try:
                _MODEL, _PROC = _build()
            except Exception as e2:
                raise RuntimeError(
                    f"GPU out of memory even after freeing cache ({e2}). "
                    "Restart the kernel (Run -> Restart) and run section 5 only, "
                    "without running the section 4b test cell first.") from e2
        else:
            raise
    _MODEL_ID = model_id
    return _MODEL, _PROC


def _norm_date(v: str) -> str:
    if not v:
        return ""
    m = re.search(r"(\d{1,2})[^\d]?(\d{1,2})[^\d]?(\d{4})", str(v))
    if not m:
        return str(v).strip()
    try:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
    except Exception:
        return str(v).strip()


def _digits_aadhar(v: str) -> str:
    if not v:
        return ""
    d = re.sub(r"\D", "", str(v))
    if len(d) > 12:
        d = d[-12:]
    if len(d) == 12:
        return f"{d[0:4]} {d[4:8]} {d[8:12]}"
    return str(v).strip()


def _clean_person_fields(d: dict) -> dict:
    """Deterministically fix name/parent/relation, regardless of model slips:
    split at the S/o / D/o / W/o marker and never confuse it with R/o (address)."""
    name = str(d.get("name", "") or "").strip()
    parent = str(d.get("parent", "") or "").strip()
    rel = str(d.get("relation", "") or "").strip().upper()

    relmap = {"S": "S/o", "D": "D/o", "W": "W/o"}
    rel_letter = rel[0] if (rel and rel[0] in "SDW") else ""
    if not rel_letter:
        m = re.search(r"\b([SDW])\s*/?\s*[oO0]\b", f"{name} {parent}")
        rel_letter = m.group(1) if m else ""

    # If the relation marker (S/o, D/o, W/o) ended up INSIDE the name, split it
    # and trust the marker found in the name (it directly introduces the parent).
    m = re.search(r"^(.*?)\b([SDW])\s*/?\s*[oO0]\s+(.*)$", name, flags=re.S)
    if m:
        name = m.group(1).strip(" ,./-")
        rel_letter = m.group(2)
        tail = m.group(3)
        # parent = up to the next R/o (address), Aadhaar, bracket, comma, newline
        tail = re.split(r"\bR\s*/\s*o\b|\(?Aadha", tail, flags=re.I)[0]
        cand = tail.split(",")[0].split("(")[0].strip(" ,./-")
        if cand:
            parent = cand

    # If parent wrongly contains the R/o (residence/address) marker, cut there.
    parent = re.split(r"\bR\s*/\s*o\b", parent, flags=re.I)[0].strip(" ,./-")
    # Strip a leaked relation PREFIX from parent only if it is the actual
    # "S/o" / "D/o" token (never strip a bare initial like "D." in "D. Jahangeen").
    parent = re.sub(r"^[SDW]\s*/\s*[oO0]\s*", "", parent, flags=re.I).strip()
    # Parent shouldn't contain Aadhaar / trailing bracket junk.
    parent = re.split(r"\(?Aadha|\d{4}\s*\d{4}\s*\d{4}", parent)[0].strip(" ,./-()")

    # Name must not keep a trailing relation marker either.
    name = re.split(r"\b[SDW]\s*/?\s*[oO0]\b", name)[0].strip(" ,./-")
    # Strip leading honorifics the model may keep (requires the dot, so real
    # names like "Msmena" are never cut). Handles "Mr./Ms.", "Ms.", "Dr." etc.
    name = re.sub(
        r"^(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.\s*(?:/\s*(?:Mrs?|Ms|Miss|Dr|Prof)\.\s*)*",
        "", name).strip(" ,./-")

    relation = relmap.get(rel_letter, "")
    d["name"] = name
    d["parent"] = parent
    d["relation"] = relation
    return d


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {}
    data = None
    for candidate in (m.group(0), "{" + m.group(0).strip("{} ") + "}"):
        try:
            data = json.loads(candidate)
            break
        except Exception:
            data = None
    if not isinstance(data, dict):
        return {}
    rel = str(data.get("relation", "") or "").strip().upper()
    rel = rel.replace("SON OF", "S/O").replace("DAUGHTER OF", "D/O")
    rel = rel.replace("WIFE OF", "W/O")
    if rel and rel[0] in "SDW":
        rel = rel[0] + "/O"
    out = {
        "name": str(data.get("name", "") or "").strip(),
        "parent": str(data.get("parent", "") or "").strip(),
        "relation": rel,
        "aadhar": _digits_aadhar(data.get("aadhar", "")),
        "course": str(data.get("course", "") or "").strip(),
        "date_from": _norm_date(data.get("date_from", "")),
        "date_to": _norm_date(data.get("date_to", "")),
        "institute": str(data.get("institute", "") or "").strip(),
        "_raw_ai": text.strip(),
    }
    return _clean_person_fields(out)


def _generate(pil, model, proc, prompt: str | None = None,
              max_new_tokens: int = 512) -> str:
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt or _PROMPT}]}]
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil], padding=True,
                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0]


_VERIFY_PROMPT = (
    "Look ONLY at the name of the organization/institute that ISSUED this "
    "certificate (printed in words, usually under or around the logo at the "
    "top, and/or near the signatory at the bottom).\n"
    "A previous reading reported the institute as: \"{got}\"\n"
    "Re-read the printed name CAREFULLY, letter by letter, and fix any "
    "misread letters. Printed 'l', 'i', 's' and 'h' look alike: for example "
    "'Falih' is F-a-l-i-h (NOT Falis, NOT Fahis, NOT Faris). Also write the "
    "name COMPLETE, exactly as printed — spell out every word; never shorten "
    "to initials.\n"
    "Return ONLY a JSON object with exactly this key (no prose, no code "
    "fence):\n"
    '{"institute": "<the complete, letter-checked correct institute name, '
    'or \\"\\" if not readable>"}'
)


def _clean_institute(v: str) -> str:
    v = str(v or "").strip().strip(".,;\"'` ")
    if v.lower() in ("", "n/a", "na", "null", "none", "-"):
        return ""
    return v


def _parse_verify(text: str) -> str:
    """Extract the corrected institute string from the verify-pass output."""
    if not text:
        return ""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
               flags=re.I).strip()
    if "{" in t:  # model answered in JSON (possibly slightly malformed)
        m = re.search(r"\{.*\}", t, re.S)
        blob = m.group(0) if m else t
        try:
            obj = json.loads(blob)
            return _clean_institute(obj.get("institute", ""))
        except Exception:
            mm = re.search(r'"institute"\s*:\s*"([^"]*)"', blob)
            if mm:
                return _clean_institute(mm.group(1))
            return ""
    # plain-text answer: prefer the longest line (model may add a short
    # preamble like "The institute is:" before the actual name)
    best = ""
    for line in t.splitlines():
        line = line.strip().strip("{}\"'` ")
        line = re.sub(r"^institute\s*:?", "", line, flags=re.I).strip()
        line = re.sub(r"^(the\s+)?(name|institute)\s+(is|of)\s*:?\s*$",
                      "", line, flags=re.I).strip()
        if _clean_institute(line) and len(line.strip()) >= 3 \
                and len(line) > len(best):
            best = line
    return _clean_institute(best)


def ai_extract_fields(img_bgr,
                      model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
                      ) -> dict:
    """Run the AI vision model on one certificate (BGR ndarray) and return a
    fields dict. On any failure returns a dict with '_error' (never raises)."""
    try:
        import numpy as np  # noqa
        import cv2
        from PIL import Image
    except Exception as e:
        return {"_error": f"deps missing (numpy/Pillow/cv2): {e}"}

    global _MODEL, _PROC
    model = proc = None
    try:
        _ensure_deps()
        model, proc = _load(model_id)
    except Exception as e:
        name = getattr(e, "name", None)
        m = re.search(r"No module named ['\"]([a-zA-Z0-9_\-]+)", str(e))
        pkg = name or (m.group(1) if m else None)
        pip_map = {"PIL": "pillow", "torchvision": "torchvision",
                   "accelerate": "accelerate"}
        pip = pip_map.get(pkg)
        if pip:
            print(f"[ai_extract] model needs '{pip}'; installing and retrying...")
            if _pip_install(pip):
                _MODEL, _PROC = None, None
                try:
                    _ensure_deps()
                    model, proc = _load(model_id)
                except Exception as e2:
                    return {"_error": f"model load failed after installing "
                                      f"{pip}: {e2}"}
            else:
                return {"_error": f"model load failed (missing {pip}): {e}"}
        else:
            return {"_error": f"model load failed: {e}"}

    rgb = img_bgr[:, :, ::-1]
    # Up-scale small/phone screenshots a bit so the VLM has more pixels to read.
    if rgb.shape[1] < 1000:
        scale = 1000 / rgb.shape[1]
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    import numpy as np
    pil = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")

    out = ""
    try:
        out = _generate(pil, model, proc)
    except Exception as e:
        try:  # one retry slightly larger
            bigger = cv2.resize(rgb, None, fx=1.4, fy=1.4,
                                interpolation=cv2.INTER_CUBIC)
            pil = Image.fromarray(np.ascontiguousarray(bigger)).convert("RGB")
            out = _generate(pil, model, proc)
        except Exception as e2:
            return {"_error": f"generation failed: {e2}"}

    fields = _parse_json(out)
    if not fields:
        return {"_error": "model output was not JSON", "_raw_ai": out.strip()}

    # ---- Institute spelling check: one extra focused look so names like
    # 'Falih' are not misread as 'Falis' (l/i/s confusion). Never fatal.
    got_inst = str(fields.get("institute", "") or "").strip()
    if got_inst:
        try:
            v_raw = _generate(pil, model, proc,
                              prompt=_VERIFY_PROMPT.replace("{got}", got_inst),
                              max_new_tokens=96)
            fixed = _parse_verify(v_raw)
            if fixed:
                fields["institute_verified"] = fixed
                if fixed.strip().lower() != got_inst.strip().lower():
                    print(f"[ai_extract] institute spelling check: "
                          f"'{got_inst}' -> '{fixed}'")
                    fields["institute"] = fixed
            fields["_raw_ai"] = out.strip() + "\n\n[institute re-check] " \
                + v_raw.strip()[:220]
        except Exception as e:
            print(f"[ai_extract] institute re-check skipped ({e})")
    return fields
