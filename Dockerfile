# B2B CatVTON virtual try-on — RunPod serverless worker.
# Dedicated worker (NOT worker-comfyui): CatVTON needs torch ~2.1 + diffusers, and the
# usual ComfyUI CatVTON nodes require detectron2/DensePose (torch 2.1.2, py3.10) which is a
# nightmare to build. We skip detectron2 entirely and mask with segformer_b2_clothes.
#
# Models (SD1.5-inpaint base + CatVTON attn + segformer) download from HuggingFace on the
# first cold start and are cached on the network volume (HF_HOME=/runpod-volume/hf-cache),
# so they download ONCE and every later cold start reuses them.
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# CatVTON pipeline code (model/pipeline.py, model/attn_processor.py, utils). We only use the
# diffusers pipeline from here — not its detectron2/densepose cloth_masker.
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /CatVTON

# numpy MUST stay on 1.x: the base image's torch 2.1.1 is built against numpy 1 and
# torch.from_numpy() raises "Numpy is not available" under numpy 2. Pin it first (and pin
# opencv to a numpy-1-compatible wheel) so nothing silently upgrades numpy to 2.x.
RUN pip install --no-cache-dir \
      "numpy==1.26.4" \
      runpod \
      "diffusers==0.26.3" \
      "transformers==4.38.2" \
      "accelerate==0.27.2" \
      "huggingface_hub==0.23.4" \
      hf_transfer \
      safetensors \
      "opencv-python-headless==4.9.0.80" \
      pillow

# sanity 1: numpy stayed on 1.x and torch<->numpy interop works (guards the torch 2.1 / numpy 2 trap)
RUN python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__); assert numpy.__version__.startswith('1.'), 'numpy must be 1.x'; assert int(torch.from_numpy(numpy.arange(3)).sum())==3; print('numpy<->torch OK')"

# sanity 2: the CatVTON pipeline imports cleanly against this diffusers version
RUN python -c "import sys; sys.path.insert(0,'/CatVTON'); from model.pipeline import CatVTONPipeline; print('CatVTONPipeline import OK')"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
