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

    # Sleeves: if the garment's top (upper-clothes / dress) covers the arms — i.e. few
    # BARE arm pixels relative to the top's size — it is sleeved, so mask the person's
    # arms too and the sleeves can actually be painted (not stuck at the model's cut).
    upper = int(np.isin(garment_seg, [4, 7]).sum())
    arms  = int(np.isin(garment_seg, ARM_LABELS).sum())
    has_sleeves = upper > 50 and arms < 0.35 * upper
    if has_sleeves:
        regions |= set(ARM_LABELS)
    return regions, full_length


def _agnostic_mask(person_img, garment_seg, grow=14):
    """
    Garment-aware mask (white = repaint). Uses the person parse for body parts and the
    (already segmented) garment to decide how much of the body to cover — legs for a
    full-length outfit, arms for a sleeved one — then bridges short vertical gaps (bare
    midriff/waist) so a full outfit is painted continuously.
    """
    person_seg = _segment(person_img)
    regions, _ = _garment_regions(garment_seg)

    mask = (np.isin(person_seg, list(regions)).astype(np.uint8)) * 255

    try:
        import cv2
        if grow and grow > 0:
            mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8), iterations=1)
        # Close SHORT vertical gaps (crop-top midriff, thigh above shorts) so a full
        # outfit is continuous — but keep the kernel short enough that it never joins
        # far-apart regions (e.g. a raised arm to the shoulder).
        H = mask.shape[0]
        kh = max(grow, int(H * 0.13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kh, max(grow, 8)), np.uint8))
    except Exception:
        pass
    return Image.fromarray(mask, mode="L")


def _clean_garment(garment_img, garment_seg):
    """
    Isolate the garment onto a clean white background — drop the garment photo's model,
    pose, face and room. CatVTON expects a flat packshot; handing it a full on-model photo
    is the #1 cause of garment drift. Falls back to the original if segmentation is thin.
    """
    cloth = np.isin(garment_seg, CLOTH_LABELS)
    if int(cloth.sum()) < 200:
        return garment_img
    arr = np.array(garment_img)
    out = np.full_like(arr, 255)                 # white canvas
    out[cloth] = arr[cloth]                       # keep only the garment pixels
    return Image.fromarray(out)


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if not inp.get("person") or not inp.get("garment"):
            return {"error": "need both 'person' and 'garment' (base64/dataURI)"}

        person = _b64_to_img(inp["person"])
        garment = _b64_to_img(inp["garment"])
        steps = int(inp.get("steps", 40))
        guidance = float(inp.get("guidance_scale", 3.5))
        grow = int(inp.get("mask_grow", 14))
        seed = int(inp.get("seed", 42))
        clean = bool(inp.get("clean_garment", True))   # isolate the garment onto white
        W, H = 768, 1024

        _load()
        person_r = person.resize((W, H))
        garment_r = garment.resize((W, H))
        garment_seg = _segment(garment_r)                      # segment the garment once
        condition = _clean_garment(garment_r, garment_seg) if clean else garment_r
        mask = _agnostic_mask(person_r, garment_seg, grow=grow).resize((W, H))

        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        out = _pipe(
            image=person_r,
            condition_image=condition,
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
