FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && python -m pip install \
    torchvision==0.22.0 \
    opencv-python-headless \
    numpy \
    tqdm \
    PyYAML

RUN mkdir -p /workspace /exam/outputs

WORKDIR /workspace

CMD ["bash"]
