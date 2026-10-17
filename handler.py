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

# segformer_b2_clothes label ids.
# clothing: 4 upper-clothes, 5 skirt, 6 pants, 7 dress, 8 belt.
# body:     12 left-leg, 13 right-leg, 14 left-arm, 15 right-arm.
CLOTH_LABELS = [4, 5, 6, 7, 8]
LOWER_LABELS = [5, 6, 7]          # skirt / pants / dress — garment parts that reach the legs
LEG_LABELS   = [12, 13]
ARM_LABELS   = [14, 15]

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


def _segment(img):
    """Per-pixel segformer label map [H, W] for an image."""
    inputs = _seg_proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = _seg_model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    return up.argmax(dim=1)[0].cpu().numpy()


def _garment_regions(garment_seg):
    """
    Which of the PERSON's body regions the garment should cover — so the mask is sized
    to the GARMENT, not just to what the model already wears. Prevents the classic
    failure where a full outfit (long jeans) is put on a crop-top+shorts model and has
    nowhere below the shorts to be painted.
    """
    regions = set(CLOTH_LABELS)                 # always replace the existing clothing
    cloth = np.isin(garment_seg, CLOTH_LABELS)
    total = int(cloth.sum())
    if total < 50:
        return regions, False                   # couldn't read the garment: keep it tight
    H = garment_seg.shape[0]
    rows = np.where(cloth.any(axis=1))[0]
    reaches = rows.max() / float(H)              # how far down the garment extends (0..1)
    has_lower = int(np.isin(garment_seg, LOWER_LABELS).sum()) > 50
    full_length = has_lower and reaches > 0.70   # long trousers / maxi dress / long skirt
    if full_length:
        regions |= set(LEG_LABELS)               # let the garment be painted over the legs
    return regions, full_length


def _agnostic_mask(person_img, garment_img, grow=12):
    """
    Garment-aware mask (white = repaint). Segments both images: the person to find the
    body parts, the garment to decide how much of the body to cover. For a full-length
    outfit it also covers the legs and bridges the waist/midriff gap so the garment is
    continuous head-to-ankle.
    """
    person_seg = _segment(person_img)
    garment_seg = _segment(garment_img)
    regions, full_length = _garment_regions(garment_seg)

    mask = (np.isin(person_seg, list(regions)).astype(np.uint8)) * 255

    try:
        import cv2
        if grow and grow > 0:
            mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8), iterations=1)
        if full_length:
            # Fill each column between its topmost and bottommost masked pixel. This
            # bridges bare-skin gaps (midriff on a crop top, thigh above shorts) so the
            # full outfit has a continuous region to be painted into, shoulders->ankles.
            m = mask > 0
            cols = np.where(m.any(axis=0))[0]
            for c in cols:
                r = np.where(m[:, c])[0]
                mask[r.min():r.max() + 1, c] = 255
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
        mask = _agnostic_mask(person_r, garment_r, grow=grow).resize((W, H))

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
