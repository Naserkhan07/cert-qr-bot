"""
verification_page.py
--------------------
Build a self-contained HTML certificate-verification page for each candidate.
The certificate image is EMBEDDED (base64), so uploading the single .html file
is enough — when a user scans the QR (which links to this page) they see the
certificate image with the candidate's details beneath it.

Details (read off the certificate by the AI vision model):
    Name of the candidate, Father/Mother Name, Aadhar No., Course Name,
    Course Duration (dd/mm/yyyy -> dd/mm/yyyy), Institute.
"""

from __future__ import annotations

import base64
import html as _html


def _b64_png(img_bgr) -> str:
    import cv2
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _esc(x) -> str:
    return _html.escape(str(x if x not in (None, "") else "—"))


def build_page(fields: dict, img_bgr, cert_filename: str) -> str:
    name = fields.get("name", "")
    parent = fields.get("parent", "")
    aadhar = fields.get("aadhar", "")
    course = fields.get("course", "")
    dfrom = fields.get("date_from", "")
    dto = fields.get("date_to", "")
    dates = f"{_esc(dfrom)} &nbsp;to&nbsp; {_esc(dto)}" if (dfrom or dto) else "—"

    img_b64 = _b64_png(img_bgr)

    institute = fields.get("institute", "")
    rows = [
        ("Name of the candidate", _esc(name)),
        ("Father/Mother Name", _esc(parent)),
        ("Aadhar No.", _esc(aadhar)),
        ("Course Name", _esc(course)),
        ("Course Duration", dates),
        ("Institute", _esc(institute)),
    ]
    rows_html = "\n".join(
        f'<tr><th>{label} :</th><td>{val}</td></tr>' for label, val in rows)

    cert_html = f'''
    <div class="cert">
      <img alt="Certificate for {_esc(name or cert_filename)}"
           src="data:image/png;base64,{img_b64}">
      <div><span class="badge">&#10003; Verified certificate</span></div>
    </div>'''

    info_html = f'''
    <h2 class="sechd">Candidate Information</h2>
    <table>
{rows_html}
    </table>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Certificate Verification - {_esc(name or cert_filename)}</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:Georgia,'Times New Roman',serif;background:#eef1f5;margin:0;
       padding:24px 12px;color:#1b1b1b}}
  .wrap{{max-width:760px;margin:0 auto}}
  .head{{text-align:center;margin-bottom:16px}}
  .head .org{{font-size:13px;letter-spacing:2px;color:#0b6b3a;font-weight:bold;
       text-transform:uppercase}}
  .head h1{{font-size:22px;margin:6px 0 2px;color:#10345e}}
  .head .sub{{color:#666;font-size:13px}}
  .card{{background:#fff;border:1px solid #d9dee6;border-radius:14px;
       box-shadow:0 10px 30px rgba(0,0,0,.08);padding:22px 24px}}
  table{{width:100%;border-collapse:collapse;font-size:16px}}
  th{{text-align:left;width:46%;color:#333;font-weight:normal;padding:10px 8px;
     border-bottom:1px solid #eef0f3;vertical-align:top}}
  td{{padding:10px 8px;border-bottom:1px solid #eef0f3;font-weight:bold;
     color:#111;word-break:break-word}}
  .fcs{{margin:6px 0 10px;text-align:center;font-weight:bold;letter-spacing:1px;
       color:#0b6b3a;font-size:14px;text-transform:uppercase}}
  .sechd{{font-size:16px;color:#10345e;border-bottom:2px solid #e5e9f0;
       padding-bottom:6px;margin:18px 0 8px}}
  .cert{{margin-top:16px;text-align:center}}
  .cert img{{max-width:100%;border:1px solid #d9dee6;border-radius:8px;
       box-shadow:0 4px 14px rgba(0,0,0,.10)}}
  .badge{{display:inline-block;background:#e8f6ee;color:#0b6b3a;border:1px solid #bfe6cf;
       border-radius:999px;padding:5px 14px;font-size:12px;margin-top:8px}}
  .foot{{text-align:center;color:#8a93a0;font-size:12px;margin-top:16px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div class="org">Government of Telangana &middot; TGMFC</div>
    <h1>Certificate Verification</h1>
    <div class="sub">Skill Development Training certificate</div>
  </div>
  <div class="card">
{cert_html}
{info_html}
  </div>
  <div class="foot">Government of Telangana &middot; TGMFC &mdash;
       Skill Development Training certificate verification</div>
</div>
</body>
</html>
"""
