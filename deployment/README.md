# BTXRD Deployment Guide for NVIDIA Jetson Nano

This folder contains everything you need to deploy the BTXRD (Bone Tumor X-Ray Detection) AI application to an NVIDIA Jetson Nano Developer Kit.

## 📁 Files in This Folder

| File | Description |
|------|-------------|
| `README.md` | This comprehensive deployment guide |
| `.env.local` | Template for local Windows/Mac development (DO NOT use on Jetson) |
| `.env.jetson` | Pre-configured environment for Jetson Nano (ready to use) |
| `Dockerfile` | Backend Docker image build instructions |
| `Dockerfile.frontend` | Frontend Docker image build instructions |
| `docker-compose.yml` | Production Docker Compose configuration |
| `docker-compose.dev.yml` | Development override (for local testing only) |
| `export_onnx.py` | Script to convert PyTorch models to ONNX format |
| `requirements.txt` | Python dependencies |
| `DEPLOYMENT_CHECKLIST.md` | Quick reference checklist |

---

## 📋 Prerequisites

### Hardware Required
- NVIDIA Jetson Nano Developer Kit (4GB RAM recommended)
- MicroSD card (32GB minimum, 64GB+ recommended) with JetPack 4.6.1 or later
- USB 3.0 flash drive or external SSD (16GB+ free space)
- Power supply (5V/4A for Jetson Nano)
- Ethernet cable or WiFi adapter
- HDMI display and keyboard/mouse for initial setup

### Software Required (on your PC)
- Python 3.8+ with pip
- Docker Desktop (optional, for local testing)
- SSH client (for remote access to Jetson)
- File transfer tool (WinSCP, FileZilla, or scp)

---

## 🚀 Step-by-Step Deployment Instructions

### Phase 1: Prepare Models on Your PC

#### Step 1: Install Python Dependencies

Open a terminal/command prompt in this `deployment` folder and install required packages:

```bash
pip install -r requirements.txt
```

#### Step 2: Export Models to ONNX Format

The Jetson uses TensorRT engines, which must be converted from your trained PyTorch models.

1. Make sure your model files are in the correct location:
   - `BTXRD/combined_inference/models/classification_student.pth`
   - `BTXRD/combined_inference/models/segmentation_student.pth`

2. Run the export script:

```bash
python export_onnx.py
```

3. Verify you see **PASSED** for both models:
   ```
   PASSED: ONNX output matches PyTorch (atol=1e-4)
   ```

4. The script creates two new files in the same folder:
   - `classification_student.onnx`
   - `segmentation_student.onnx`

#### Step 3: Download the LLM Model

Download the Gemma-2-2B GGUF model file:

1. Go to: https://huggingface.co/google/gemma-2-2b-it-GGUF
2. Download: `gemma-2-2b-it-Q4_K_M.gguf` (approximately 1.6 GB)
3. Save it to your models folder

#### Step 4: Prepare USB Drive

Copy the following files to your USB drive:

```
USB_DRIVE_ROOT/
├── models/
│   ├── classification_student.onnx
│   ├── segmentation_student.onnx
│   └── gemma-2-2b-it-Q4_K_M.gguf
├── deployment/           (copy entire deployment folder)
└── btxrd-backend/        (copy entire backend folder)
└── btxrd-frontend/       (copy entire frontend folder)
```

---

### Phase 2: Set Up Jetson Nano

#### Step 5: Flash JetPack OS

If you haven't already, flash JetPack 4.6.1 or later to your Jetson Nano's SD card:

1. Download NVIDIA SDK Manager: https://developer.nvidia.com/sdk-manager
2. Follow NVIDIA's official flashing guide
3. Boot into Jetson Nano and complete initial setup

#### Step 6: Connect USB Drive and Mount

1. Plug in your USB drive to the Jetson Nano's USB 3.0 port (blue port)

2. Identify the USB drive:
   ```bash
   lsblk
   ```
   Look for your USB drive (typically `/dev/sda1`)

3. Create mount point and mount the drive:
   ```bash
   sudo mkdir -p /data
   sudo mount /dev/sda1 /data
   ```

4. Verify files are accessible:
   ```bash
   ls /data/models/
   ```
   You should see the 3 model files.

5. Make the mount permanent (auto-mount on boot):
   ```bash
   echo "/dev/sda1  /data  exfat  defaults,noatime  0  2" | sudo tee -a /etc/fstab
   ```

   > **Note:** If your USB drive is formatted as ext4 (recommended), use `ext4` instead of `exfat`

#### Step 7: Install Docker

1. Update package list:
   ```bash
   sudo apt-get update
   ```

2. Install Docker:
   ```bash
   sudo apt-get install -y docker.io
   ```

3. Add your user to the docker group (to run without sudo):
   ```bash
   sudo usermod -aG docker $USER
   ```

4. Log out and log back in for the group change to take effect

5. Verify Docker installation:
   ```bash
   docker --version
   docker run --rm hello-world
   ```

#### Step 8: Install NVIDIA Container Runtime

1. Add NVIDIA's package repository:
   ```bash
   wget -q -O - https://ngc.nvidia.com/downloads/jetson/jetson-461.key | sudo apt-key add -
   echo "deb https://repo.download.nvidia.com/jetson/jetson-461 r35 main" | sudo tee /etc/apt/sources.list.d/nvidia-l4t.list
   ```

2. Install container runtime:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-runtime
   ```

3. Configure Docker to use NVIDIA runtime by default:
   ```bash
   sudo mkdir -p /etc/docker
   echo '{"default-runtime": "nvidia", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}' | sudo tee /etc/docker/daemon.json
   sudo systemctl restart docker
   ```

---

### Phase 3: Build and Deploy

#### Step 9: Copy Project Files

If you haven't already copied all files to the Jetson:

```bash
# From your PC, using scp or rsync
scp -r deployment/ btxrd-backend/ btxrd-frontend/ jetson@JETSON_IP:/home/jetson/
```

Or copy from USB to home directory:
```bash
cp -r /data/deployment ~/deployment
cp -r /data/btxrd-backend ~/btxrd-backend
cp -r /data/btxrd-frontend ~/btxrd-frontend
```

#### Step 10: Convert ONNX Models to TensorRT Engines

On the Jetson, run the TensorRT conversion:

```bash
# Navigate to deployment folder
cd ~/deployment

# Convert classification model
trtexec --onnx=/data/models/classification_student.onnx \
        --saveEngine=/data/models/classify.trt \
        --fp16 \
        --minShapes=input:1x3x384x384 \
        --optShapes=input:1x3x384x384 \
        --maxShapes=input:4x3x384x384

# Convert segmentation model
trtexec --onnx=/data/models/segmentation_student.onnx \
        --saveEngine=/data/models/segment.trt \
        --fp16 \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:1x3x224x224 \
        --maxShapes=input:4x3x224x224
```

> **Note:** This conversion takes 5-10 minutes per model.

#### Step 11: Verify .env.jetson Configuration

The `.env.jetson` file is pre-configured. Verify the paths match your setup:

```bash
cat ~/deployment/.env.jetson
```

Expected output:
```
MODEL_DIR=/data/models
CLASSIFY_MODEL_FILE=classify.trt
SEGMENT_MODEL_FILE=segment.trt
LLAMA_CLI_PATH=/app/llama.cpp/llama-cli
LLM_MODEL_PATH=/data/models/gemma-2-2b-it-Q4_K_M.gguf
LLM_THREADS=4
UPLOAD_DIR=/data/uploads
REPORT_DIR=/data/reports
PORT=8000
```

#### Step 12: Build Docker Images

```bash
cd ~/deployment

# Build backend image (takes 20-40 minutes first time)
docker build -t btxrd-backend -f Dockerfile .

# Build frontend image
docker build -t btxrd-frontend -f Dockerfile.frontend ./btxrd-frontend
```

#### Step 13: Run with Docker Compose

```bash
# Start both services
docker compose up -d

# View logs
docker compose logs -f
```

The application should now be running:
- **Backend API:** http://JETSON_IP:8000
- **Frontend UI:** http://JETSON_IP:3000

#### Step 14: Verify Deployment

1. Check health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```
   Expected response: `{"status": "ok", "models_loaded": true}`

2. Open browser and navigate to: `http://JETSON_IP:3000`

3. Upload a test X-ray image and verify:
   - Classification shows tumor type with confidence
   - Segmentation displays tumor overlay
   - Chat responds to questions about the analysis
   - Report generation creates downloadable PDF

---

## 🔧 Troubleshooting

### Docker Build Fails
- **Error: "image not found"** - Ensure you're using JetPack 4.6.1+ with the correct base image
- **Error: "memory exceeded"** - Close other applications, Jetson Nano has limited RAM

### Models Not Loading
- Verify `/data/models/` contains all 3 files:
  - `classify.trt`
  - `segment.trt`
  - `gemma-2-2b-it-Q4_K_M.gguf`

- Check file permissions:
  ```bash
  ls -la /data/models/
  ```

### TensorRT Conversion Fails
- Ensure ONNX models exist and are valid:
  ```bash
  python3 -c "import onnx; onnx.load('/data/models/classification_student.onnx')"
  ```

- Try without shape optimization:
  ```bash
  trtexec --onnx=/data/models/classification_student.onnx --saveEngine=/data/models/classify.trt --fp16
  ```

### Frontend Cannot Connect to Backend
- Verify `NEXT_PUBLIC_API_URL` in docker-compose.yml points to correct address
- Check backend is running: `docker compose ps`
- View backend logs: `docker compose logs backend`

### Permission Denied Errors
- Add user to docker group: `sudo usermod -aG docker $USER`
- Log out and back in
- Or run docker commands with `sudo`

---

## 🔄 Updating the Deployment

To update the application after making code changes:

1. Copy updated files to Jetson:
   ```bash
   scp btxrd-backend/app/*.py jetson@JETSON_IP:~/deployment/btxrd-backend/app/
   ```

2. Rebuild and restart:
   ```bash
   cd ~/deployment
   docker compose down
   docker build -t btxrd-backend -f Dockerfile ..
   docker compose up -d
   ```

---

## 🛑 Stopping the Application

```bash
# Stop all services
docker compose down

# Stop and remove containers + volumes
docker compose down -v
```

---

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Classification Inference | ~200-500ms |
| Segmentation Inference | ~300-600ms |
| LLM Response Generation | ~5-15 tokens/second |
| Memory Usage (Backend) | ~2.5-3.0 GB |
| Memory Usage (Frontend) | ~200-400 MB |

---

## 📝 Additional Notes

### USB Drive Auto-Mount Issues

If the USB drive doesn't auto-mount on boot:

1. Find the UUID of your USB drive:
   ```bash
   sudo blkid /dev/sda1
   ```

2. Update `/etc/fstab` with UUID instead of device name:
   ```
   UUID=YOUR-UUID-HERE  /data  ext4  defaults,noatime  0  2
   ```

### Swap Space (Recommended for 2GB Jetson Nano)

If you have the 2GB model, add swap space:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Remote Access Setup

Enable SSH for remote management:

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

Then access from your PC:
```bash
ssh jetson@JETSON_IP
```

---

## 📞 Support

For issues related to:
- **Jetson Nano hardware:** https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **TensorRT:** https://github.com/NVIDIA/TensorRT
- **Docker on Jetson:** https://docs.nvidia.com/jetson/l4t/index.html

---

**Last Updated:** March 2026
**BTXRD Version:** 1.0.0
**Compatible JetPack Versions:** 4.6.1, 5.0+, 6.0+
