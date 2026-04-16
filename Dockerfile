# NVIDIA Jetson Nano (ARM64) Backend Image
# Base: JetPack r35.2.1 with PyTorch 2.0, CUDA, TensorRT
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY btxrd-backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Build llama.cpp from source
RUN git clone https://github.com/ggerganov/llama.cpp /app/llama.cpp \
    && cd /app/llama.cpp \
    && make -j4 LLAMA_FAST=1

# Copy backend source code
COPY btxrd-backend/app /app/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start uvicorn with 1 worker (Jetson memory constraint)
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
