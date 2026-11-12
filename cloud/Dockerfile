# Official PyTorch CUDA 11.8 / cuDNN 9 runtime, pinned to the published manifest.
FROM ghcr.io/pytorch/pytorch@sha256:aad7e3b867b03934b662f5ad18d6181f58fe0537b25be625f4c9d4971e09aab7

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/ml/processing/checkpoints/huggingface
ENV TORCH_HOME=/opt/ml/processing/checkpoints/torch
ENV XDG_CACHE_HOME=/opt/ml/processing/checkpoints/xdg
ENV TABPFN_MODEL_CACHE_DIR=/opt/ml/processing/checkpoints/tabpfn
ENV ILTM_CKPT_DIR=/opt/ml/processing/checkpoints/iltm
WORKDIR /opt/benchmark

COPY cloud/requirements-linux-py311.lock /opt/benchmark/requirements.lock
COPY pyproject.toml README.md /opt/benchmark/
COPY src /opt/benchmark/src

RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      -r /opt/benchmark/requirements.lock \
 && python -m pip install --no-cache-dir --no-deps /opt/benchmark \
 && python -m pip check \
 && python -m tabtune_baseline.image_self_test

ENTRYPOINT ["python"]
