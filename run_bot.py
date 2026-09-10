#!/usr/bin/env python3
"""
Cert-QR Bot — command line entry point.

Examples
--------
# Recommended: each QR links to the hosted certificate image
python run_bot.py --input input --output output \
    --base-url https://YOUR-NAME.github.io/certificates/

# Copy whatever the old QR said
python run_bot.py run_bot.py --input input --payload-mode copy

# Per-cert URLs from a spreadsheet
python run_bot.py --input input --payload-mode csv --csv links.csv
"""
import argparse

from src.pipeline import Config, run


def main():
    ap = argparse.ArgumentParser(description="Replace QR codes on certificates")
    ap.add_argument("--input", default="input", help="folder of certificate images")
    ap.add_argument("--output", default="output", help="folder for results")
    ap.add_argument("--model-dir", default="models/wechat",
                    help="folder with WeChat CNN model files")
    ap.add_argument("--zip", default="certificates_with_new_qr.zip")
    ap.add_argument("--payload-mode", default="url",
                    choices=["url", "text", "csv", "copy"])
    ap.add_argument("--base-url",
                    default="https://example.github.io/certificates/")
    ap.add_argument("--url-extension", default=".png")
    ap.add_argument("--text-prefix", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ec", default="M", choices=["L", "M", "Q", "H"],
                    help="QR error correction")
    ap.add_argument("--box-size", type=int, default=20)
    ap.add_argument("--border", type=int, default=4)
    ap.add_argument("--replace-mode", default="full_replace",
                    choices=["full_replace", "keep_frame"])
    ap.add_argument("--cover-scale", type=float, default=1.08,
                    help="full_replace: grow white QR box over old frame (1.08)")
    ap.add_argument("--output-ext", default=".png")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--no-debug", action="store_true")
    # ---- QR generator backend ----
    ap.add_argument("--qr-backend", default="standard",
                    choices=["standard", "artistic"],
                    help="standard deterministic QR, or HF QR Monster ControlNet")
    ap.add_argument("--art-scale", type=float, default=1.5,
                    help="ControlNet conditioning scale (higher=more scannable)")
    ap.add_argument("--art-tries", type=int, default=6)
    ap.add_argument("--art-size", type=int, default=768)
    # ---- AI image-generator (diffusion inpainting) ----
    ap.add_argument("--ai", action="store_true",
                    help="use HF diffusion inpainting model for the replacement")
    ap.add_argument("--ai-model",
                    default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--ai-steps", type=int, default=30)
    ap.add_argument("--ai-guidance", type=float, default=8.0)
    ap.add_argument("--ai-seed", type=int, default=0)
    ap.add_argument("--ai-margin", type=int, default=120)
    ap.add_argument("--ai-generative-only", action="store_true",
                    help="return raw model output (NOT guaranteed to scan)")
    args = ap.parse_args()

    cfg = Config(
        input_dir=args.input, output_dir=args.output,
        model_dir=args.model_dir, zip_path=args.zip,
        payload_mode=args.payload_mode, base_url=args.base_url,
        url_extension=args.url_extension, text_prefix=args.text_prefix,
        csv_path=args.csv, qr_error_correction=args.ec,
        qr_box_size=args.box_size, qr_border=args.border,
        replace_mode=args.replace_mode, output_ext=args.output_ext,
        qr_cover_scale=args.cover_scale,
        qr_backend=args.qr_backend, art_controlnet_scale=args.art_scale,
        art_tries=args.art_tries, art_size=args.art_size,
        use_cnn=not args.no_cnn, save_debug=not args.no_debug,
        use_ai=args.ai, ai_model=args.ai_model, ai_steps=args.ai_steps,
        ai_guidance=args.ai_guidance, ai_seed=args.ai_seed,
        ai_margin=args.ai_margin,
        ai_mode=("generative_only" if args.ai_generative_only
                 else "inpaint_then_paste"),
    )
    run(cfg)


if __name__ == "__main__":
    main()
