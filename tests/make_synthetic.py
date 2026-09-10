"""Generate synthetic 'certificates' with QR codes in varied conditions."""
import os, cv2, numpy as np, qrcode, random

random.seed(7)
OUT = os.path.join(os.path.dirname(__file__), "synth_input")
os.makedirs(OUT, exist_ok=True)

def cert_bg(w=1240, h=880):
    # cream paper with faint mottling
    bg = np.full((h, w, 3), (235, 240, 232), np.uint8)
    noise = np.random.RandomState(1).normal(0, 6, (h, w, 1)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (30, 30), (w-30, h-30), (120, 170, 200), 4)
    cv2.putText(bg, "CERTIFICATE", (360, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                2.2, (60, 120, 200), 5)
    cv2.putText(bg, "Awarded to Student", (300, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 60), 3)
    return bg

def make_qr(text, modules=200):
    q = qrcode.make(text, box_size=4, border=2).convert("RGB")
    arr = np.array(q)[:, :, ::-1].copy()
    return cv2.resize(arr, (modules, modules), interpolation=cv2.INTER_NEAREST)

def paste_rotated(bg, patch, cx, cy, angle, frame=False):
    h, w = bg.shape[:2]
    ph, pw = patch.shape[:2]
    if frame:  # thick black border + white margin like the sample cert
        fr = np.full((ph+24, pw+24, 3), 255, np.uint8)
        fr[:] = (255,255,255)
        fr = cv2.copyMakeBorder(fr, 10,10,10,10, cv2.BORDER_CONSTANT, value=(0,0,0))
        fr[12:12+ph, 12:12+pw] = patch
        patch = fr; ph, pw = patch.shape[:2]
    M = cv2.getRotationMatrix2D((pw/2, ph/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(pw*cos + ph*sin), int(pw*sin + ph*cos)
    M[0,2] += nw/2 - pw/2; M[1,2] += nh/2 - ph/2
    warp = cv2.warpAffine(patch, M, (nw, nh),
                          borderValue=(235,240,232))
    x0, y0 = int(cx-nw/2), int(cy-nh/2)
    # only paste where patch isn't the paper-ish border (use white-ish mask)
    roi = bg[y0:y0+nh, x0:x0+nw]
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    mask = (gray < 248)  # dark ink / black frame
    # also include the pure-white inner QR background inside frame bbox
    if frame:
        inner = np.zeros_like(mask); inner[10:nh-10, 10:nw-10] = True
        mask = mask | (inner & (warp.sum(axis=2) > 720))
    roi[mask] = warp[mask]
    bg[y0:y0+nh, x0:x0+nw] = roi

configs = [
    ("001", (150, 320), 220, 0,   False),
    ("002", (160, 340), 150, 0,   True),   # small with black frame (sample-like)
    ("003", (1080, 300), 200, 0,  False),  # top-right
    ("004", (620, 700), 180, 6,   False),  # bottom-center, rotated
    ("005", (200, 720), 240, -4,  True),   # bottom-left framed, slight rot
    ("006", (1050, 690), 160, 9,  False),  # bottom-right rotated
]
for cid, (cx, cy), mod, ang, frame in configs:
    bg = cert_bg()
    qr = make_qr(f"OLD-DATA-{cid}", mod)
    paste_rotated(bg, qr, cx, cy, ang, frame=frame)
    p = os.path.join(OUT, f"cert-{cid}.png")
    # JPEG-compress half of them
    if int(cid) % 2 == 0:
        cv2.imwrite(p, bg, [cv2.IMWRITE_JPEG_QUALITY, 82])
        os.rename(p, p.replace(".png", ".jpg"))
    else:
        cv2.imwrite(p, bg)
print("wrote synthetic certs to", OUT)
