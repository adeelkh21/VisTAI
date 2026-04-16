# BTXRD Quick Start Guide

## For Local Development (Windows/Mac)

1. **Copy `.env.local` to `.env`**:
   ```bash
   cp .env.local .env
   ```

2. **Edit `.env`** and fill in your paths:
   - `MODEL_DIR` - where your `.pth` files are
   - `LLAMA_CLI_PATH` - path to llama-cli.exe
   - `LLM_MODEL_PATH` - path to your Gemma GGUF file

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend**:
   ```bash
   cd btxrd-backend
   python -m uvicorn app.main:app --reload
   ```

5. **Run the frontend** (in a new terminal):
   ```bash
   cd btxrd-frontend
   npm install
   npm run dev
   ```

6. **Open browser**: http://localhost:3000

---

## For Jetson Nano Deployment

### On Your PC (Before Going to Jetson)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Export models to ONNX**:
   ```bash
   python export_onnx.py
   ```
   Wait for both models to show **PASSED**.

3. **Download LLM model**:
   - Get `gemma-2-2b-it-Q4_K_M.gguf` from HuggingFace

4. **Copy to USB drive**:
   ```
   USB_DRIVE/
   ├── models/
   │   ├── classification_student.onnx
   │   ├── segmentation_student.onnx
   │   └── gemma-2-2b-it-Q4_K_M.gguf
   └── deployment/    (copy this entire folder)
   ```

### On the Jetson Nano

1. **Mount USB drive**:
   ```bash
   sudo mkdir -p /data
   sudo mount /dev/sda1 /data
   ```

2. **Copy files to home**:
   ```bash
   cp -r /data/deployment ~/deployment
   cd ~/deployment
   ```

3. **Install Docker**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

4. **Install NVIDIA Container Runtime**:
   ```bash
   sudo apt-get install -y nvidia-container-runtime
   ```

5. **Convert ONNX to TensorRT**:
   ```bash
   trtexec --onnx=/data/models/classification_student.onnx \
           --saveEngine=/data/models/classify.trt --fp16

   trtexec --onnx=/data/models/segmentation_student.onnx \
           --saveEngine=/data/models/segment.trt --fp16
   ```

6. **Build and run**:
   ```bash
   docker build -t btxrd-backend -f Dockerfile .
   docker build -t btxrd-frontend -f Dockerfile.frontend ./btxrd-frontend
   docker compose up -d
   ```

7. **Open browser**: http://JETSON_IP:3000

---

## Verify Everything Works

1. **Check backend health**:
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"ok","models_loaded":true}`

2. **Test in browser**:
   - Go to http://JETSON_IP:3000
   - Upload an X-ray image
   - Verify classification and segmentation work
   - Try the chat feature
   - Generate a report

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Models not loading | Check `/data/models/` has all 3 files |
| Docker permission denied | Run `sudo usermod -aG docker $USER` and log out/in |
| Frontend can't connect | Check `NEXT_PUBLIC_API_URL` in docker-compose.yml |
| Out of memory | Close other applications, restart Jetson |

---

## Stop/Restart

```bash
# Stop
docker compose down

# Restart
docker compose up -d

# View logs
docker compose logs -f
```

---

For detailed instructions, see **[README.md](README.md)**.
