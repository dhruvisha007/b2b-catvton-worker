"""
B2B CatVTON virtual try-on — RunPod serverless handler.

Pipeline (no detectron2/densepose): segformer_b2_clothes makes the clothing mask,
CatVTON's diffusers pipeline repaints ONLY that masked region with the garment, so the
person, pose and BACKGROUND stay exactly as the original photo.

Input  (event["input"]): { "person": <b64/dataURI>, "garment": <b64/dataURI>,
                           "steps"?: int, "guidance_scale"?: float, "mask_grow"?: int, "seed"?: int }
Output: { "image": "data:image/png;base64,..." }  (or {"error": "..."} )

Models are pulled from HuggingFace and cached on the network volume (HF_HOME=/runpod-volume/hf-cache),
so they download once and every later cold start reuses them.
"""
import os
import io
import sys
import base64
import traceback

import numpy as np
import torch
from PIL import Image
import runpod

sys.path.insert(0, "/CatVTON")  # the CatVTON repo is cloned here in the Dockerfile

DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
BASE_CKPT = os.environ.get("BASE_CKPT", "stable-diffusion-v1-5/stable-diffusion-inpainting")
ATTN_CKPT = os.environ.get("ATTN_CKPT", "zhengchong/CatVTON")

# segformer_b2_clothes label ids that make up "the outfit" (what we replace):
# 4 upper-clothes, 5 skirt, 6 pants, 7 dress, 8 belt.
CLOTH_LABELS = [4, 5, 6, 7, 8]

_pipe = None
_seg_proc = None
_seg_model = None


def _load():
    """Lazy-load the models once per worker (kept warm in VRAM between jobs)."""
    global _pipe, _seg_proc, _seg_model
    if _seg_model is None:
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        _seg_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _seg_model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        ).to(DEVICE).eval()
    if _pipe is None:
        from model.pipeline import CatVTONPipeline
        _pipe = CatVTONPipeline(
            base_ckpt=BASE_CKPT,
            attn_ckpt=ATTN_CKPT,
            attn_ckpt_version="mix",
            weight_dtype=DTYPE,
            device=DEVICE,
            use_tf32=True,
            skip_safety_check=True,
        )
    return _pipe


def _b64_to_img(b):
    if isinstance(b, str) and "," in b and b.strip().startswith("data:"):
        b = b.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cloth_mask(person_img, grow=12):
    """Clothing mask (white = region to repaint) from segformer, lightly dilated."""
    inputs = _seg_proc(images=person_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _seg_model(**inputs).logits  # [1, C, h, w]
    up = torch.nn.functional.interpolate(
        logits, size=person_img.size[::-1], mode="bilinear", align_corners=False
    )
    seg = up.argmax(dim=1)[0].cpu().numpy()
    mask = (np.isin(seg, CLOTH_LABELS).astype(np.uint8)) * 255
    if grow and grow > 0:
        try:
            import cv2
            mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8), iterations=1)
        except Exception:
            pass
    return Image.fromarray(mask, mode="L")


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if not inp.get("person") or not inp.get("garment"):
            return {"error": "need both 'person' and 'garment' (base64/dataURI)"}

        person = _b64_to_img(inp["person"])
        garment = _b64_to_img(inp["garment"])
        steps = int(inp.get("steps", 35))
        guidance = float(inp.get("guidance_scale", 2.5))
        grow = int(inp.get("mask_grow", 12))
        seed = int(inp.get("seed", 42))
        W, H = 768, 1024

        _load()
        person_r = person.resize((W, H))
        garment_r = garment.resize((W, H))
        mask = _cloth_mask(person_r, grow=grow).resize((W, H))

        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        out = _pipe(
            image=person_r,
            condition_image=garment_r,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=H,
            width=W,
            generator=gen,
        )
        result = out[0] if isinstance(out, (list, tuple)) else out
        return {"image": _img_to_b64(result)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}


runpod.serverless.start({"handler": handler})
