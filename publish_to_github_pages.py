#!/usr/bin/env python3
"""
publish_to_github_pages.py
--------------------------
Upload the finished certificate images to a GitHub Pages repository so the QR
links resolve (scan -> certificate opens in the browser). Self-verifying:
after each upload it polls the public Pages URL until the image is live.

Token: a GitHub Personal Access Token (classic) with the `repo` scope.
Provided via GITHUB_TOKEN / GH_TOKEN env var (on Kaggle, add a notebook Secret
named GITHUB_TOKEN), or pasted at the prompt.

Create a token: https://github.com/settings/tokens  (Generate new token
(classic) -> tick "repo").
"""

from __future__ import annotations

import argparse
import base64
import os
import time
import zipfile

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

API = "https://api.github.com"


def _get_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not tok:
        tok = input("Paste your GitHub token (repo scope): ").strip()
    return tok


def publish(zip_path: str, owner: str, repo: str, token: str,
            subdir: str = "img", branch: str = "main",
            verify: bool = True, verbose: bool = True):
    if requests is None:
        raise SystemExit("'requests' is required (pip install requests).")

    files = []   # (name, content, upload_dir)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base_name = os.path.basename(name)
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                # images stay under img/ (whether the zip root or img/ prefix)
                files.append((base_name, z.read(name), subdir))
            elif low.endswith(".html"):
                # verification pages live at the repo root (QR URLs point there)
                files.append((base_name, z.read(name), ""))

    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json",
               "User-Agent": "cert-qr-bot-publisher"}
    api_base = f"{API}/repos/{owner}/{repo}/contents"
    live_root = f"https://{owner}.github.io/{repo}/"
    ok = fail = upload_fail = 0
    uploaded_files = []   # (name, live_url, is_html)

    if verbose:
        print(f"Publishing {len(files)} file(s) to {owner}/{repo} ...")

    for i, (name, content, upload_dir) in enumerate(files, 1):
        path = f"{api_base}/{upload_dir}/{name}" if upload_dir \
            else f"{api_base}/{name}"
        # find existing sha (to update rather than conflict)
        sha = None
        try:
            r = requests.get(path, headers=headers,
                             params={"ref": branch}, timeout=30)
            if r.status_code == 200:
                sha = r.json().get("sha")
        except Exception:
            pass

        body = {"message": f"Publish {name}",
                "content": base64.b64encode(content).decode(),
                "branch": branch}
        if sha:
            body["sha"] = sha

        uploaded = False
        for attempt in range(3):
            try:
                r = requests.put(path, headers=headers, json=body, timeout=60)
                if r.status_code in (200, 201):
                    uploaded = True
                    break
                if verbose:
                    print(f"  ! {name} upload attempt {attempt+1}: "
                          f"HTTP {r.status_code} {r.text[:160]}")
            except Exception as e:
                if verbose:
                    print(f"  ! {name} attempt {attempt+1} error: {e}")
            time.sleep(2)

        live_url = (f"https://{owner}.github.io/{repo}/{upload_dir}/{name}"
                    if upload_dir else f"https://{owner}.github.io/{repo}/{name}")
        is_img = name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        is_html = name.lower().endswith(".html")
        if uploaded:
            uploaded_files.append((name, live_url, is_html))
        # Per-file quick check (short); the final sweep below re-checks any
        # that weren't live yet, so Pages propagation lag is not fatal here.
        if uploaded and verify and (is_img or is_html):
            live = _wait_live(live_url, expect_html=is_html,
                              tries=3, delay=4)
        else:
            live = uploaded

        if uploaded and live:
            ok += 1
            if verbose:
                print(f"  [{i}/{len(files)}] ✅ {name} -> {live_url}")
        elif uploaded:
            # uploaded but not live yet — final sweep will re-check it
            if verbose:
                print(f"  [{i}/{len(files)}] ⏳ {name} uploaded, waiting for Pages ...")
        else:
            upload_fail += 1
            if verbose:
                print(f"  [{i}/{len(files)}] ❌ {name} (upload failed)")
        time.sleep(0.4)

    # ---- Final verification sweep: keep re-checking every uploaded file until
    # each is live on GitHub Pages (handles propagation lag for image AND html).
    if verify and uploaded_files:
        pending = [(n, u, h) for (n, u, h) in uploaded_files]
        if verbose and pending:
            print(f"\nVerifying {len(pending)} page(s)/image(s) are live ...")
        for round_no in range(1, 13):   # up to ~12 * 10s = 2 minutes
            still = []
            for name, live_url, is_html in pending:
                if _check_one(live_url, is_html):
                    ok += 0  # already counted as uploaded
                    if verbose:
                        print(f"  ✅ live: {name}")
                else:
                    still.append((name, live_url, is_html))
            pending = still
            if not pending:
                break
            if verbose:
                print(f"  ... {len(pending)} not live yet, re-checking "
                      f"(round {round_no})")
            time.sleep(10)
        # anything still pending after the sweep is reported for a re-run
        for name, _u, _h in pending:
            if verbose:
                print(f"  ❌ {name} still not live after waiting (re-run "
                      f"section 8 in ~1 min)")
        not_live = len(pending)
    else:
        not_live = 0
    fail = not_live + upload_fail

    live_count = len(uploaded_files) - not_live
    if verbose:
        print(f"\nDone: {live_count} live, {fail} not yet live "
              f"({upload_fail} upload failure(s)).")
        if fail == 0:
            print("✅ Every image and verification page is uploaded AND live.")
        print("Pages (QR base URL): " + live_root)
        print("Images under:        " + live_root + subdir + "/")
    return {"ok": live_count, "fail": fail, "base": live_root}


def _check_one(url: str, expect_html: bool) -> bool:
    import random
    bust = url + ("&" if "?" in url else "?") + "cb=" + str(random.randint(0, 10**9))
    try:
        r = requests.get(bust, timeout=30,
                         headers={"Cache-Control": "no-cache",
                                  "User-Agent": "cert-qr-bot-publisher"})
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200:
            return False
        if expect_html:
            return "html" in ctype
        return "image" in ctype
    except Exception:
        return False


def _wait_live(url: str, tries: int = 14, delay: int = 5,
               expect_html: bool = False) -> bool:
    for _ in range(tries):
        if _check_one(url, expect_html):
            return True
        time.sleep(delay)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="certificates_with_new_qr.zip")
    ap.add_argument("--owner", default="Naserkhan07")
    ap.add_argument("--repo", default="certificate-qr")
    ap.add_argument("--subdir", default="img")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    token = _get_token()
    if not token:
        raise SystemExit("No GitHub token provided.")
    publish(args.zip, args.owner, args.repo, token, args.subdir,
            args.branch, verify=not args.no_verify)


if __name__ == "__main__":
    main()
