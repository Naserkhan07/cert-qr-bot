"""Execute the Kaggle notebook's code cells locally in a simulated environment
to prove the notebook runs end-to-end. Replaces /kaggle paths with temp dirs.
"""
import json, os, shutil, glob, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "kaggle_sim")
WORK = os.path.join(SIM, "working")
INP = os.path.join(SIM, "input")

if os.path.exists(SIM):
    shutil.rmtree(SIM)
os.makedirs(WORK)
os.makedirs(INP)

# fake dataset: the real sample cert + a couple synthetic certs
shutil.copy(os.path.join(ROOT, "tests/synth_input/cert-002.jpg"), INP)
shutil.copy(os.path.join(ROOT, "tests/synth_input/cert-004.jpg"), INP)

nb = json.load(open(os.path.join(ROOT, "kaggle/Cert_QR_Bot.ipynb")))

ns = {}
def shim_display(*a, **k):
    for x in a:
        try:
            print(x.to_string() if hasattr(x, "to_string") else x)
        except Exception:
            print(repr(x)[:200])
ns["display"] = shim_display

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    src = src.replace("/kaggle/working/certbot", os.path.join(WORK, "certbot"))
    src = src.replace("/kaggle/working", WORK)
    src = src.replace("/kaggle/input", INP)
    # skip the pip magic (deps already installed in this env)
    src = "\n".join("" if (l.strip().startswith("%pip") or l.strip().startswith("!")) else l
                    for l in src.splitlines())
    print(f"\n===== executing code cell {i} =====")
    exec(compile(src, f"<cell {i}>", "exec"), ns)

zipf = os.path.join(WORK, "certificates_with_new_qr.zip")
print("\nZIP exists:", os.path.exists(zipf),
      "size:", round(os.path.getsize(zipf)/1024, 1), "KB")
import zipfile
with zipfile.ZipFile(zipf) as z:
    print("zip contents:", z.namelist())
print("\nNOTEBOOK SIMULATION PASSED")
