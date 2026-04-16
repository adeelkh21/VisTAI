# BTXRD USB Transfer & Deployment Context

**Document Purpose:** Comprehensive guide for transferring the BTXRD application to USB and deploying it on target machines (Windows/Mac development or Jetson Nano production).

**Last Updated:** April 2, 2026  
**Status:** USB Ready, Path Update + Jetson Test Pending

---

## Table of Contents

1. [Current Project State](#current-project-state)
2. [Folder & Repository Structure](#folder--repository-structure)
3. [What Gets Copied to USB](#what-gets-copied-to-usb)
4. [Path Changes & Environment Setup](#path-changes--environment-setup)
5. [Known Issues & Fixes](#known-issues--fixes)
6. [USB Transfer Workflow](#usb-transfer-workflow)
7. [Post-Transfer Verification](#post-transfer-verification)
8. [Troubleshooting & Common Errors](#troubleshooting--common-errors)

---

## Current Project State

### Latest Confirmed State (Most Important)

- Models are organized inside `deployment/models` for USB transfer workflow.
- The model bundle contains two ONNX files and one GGUF file:
  - `classification_student.onnx`
  - `segmentation_student.onnx`
  - `gemma-2-2b-it-Q4_K_M.gguf`
- ONNX conversion is already done.
- `btxrd-backend` and `btxrd-frontend` also exist outside `deployment` (root-level copies), but for deployment we use the copies inside `deployment`.
- Next step is path alignment, then plug USB into Jetson and begin runtime testing.

### Project Overview
**BTXRD (Bone Tumor X-Ray Detection)** is a full-stack medical AI application consisting of:

- **Backend:** FastAPI server (Python)
- **Frontend:** Next.js web application (TypeScript/React)
- **ML Pipeline:** PyTorch-based classification & segmentation with knowledge distillation
- **Deployment Target:** NVIDIA Jetson Nano (primary) or local Windows/Mac dev machines

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend API | FastAPI | Latest |
| Frontend | Next.js | Latest |
| ML Framework | PyTorch | 2.0+ |
| Model Conversion | ONNX → TensorRT | - |
| LLM Backend | llama-cpp-python | - |
| Containerization | Docker Compose | 3.8 |
| Host OS | Windows/Mac (dev), JetPack OS (Jetson) | - |

### Trained Models Location

**Current Location (Development Machine):**
```
FYP/
  BTXRD/
    combined_inference/
      models/
        ├── classification_student.pth
        ├── segmentation_student.pth
        └── gemma-2-2b-it-Q4_K_M.gguf (downloaded manually)
```

**Model Files Needed for Deployment:**
- `classification_student.pth` — PyTorch classification model (~5MB)
- `segmentation_student.pth` — PyTorch segmentation model (~8MB)
- `gemma-2-2b-it-Q4_K_M.gguf` — Gemma-2 LLM model (~1.6GB)

**Conversion Requirements:**
- On USB/target: PyTorch `.pth` → ONNX → TensorRT engines (for Jetson)
- For local dev: Use `.pth` files directly (no conversion needed)

---

## Folder & Repository Structure

### Root Project Structure
```
FYP/ (workspace root)
├── BTXRD/                      # ML pipeline & models
│   ├── combined_inference/      # Unified inference code
│   │   └── models/              # Trained .pth files + GGUF
│   ├── classification/
│   ├── segmentation/
│   ├── btxrd-classification-kd/
│   ├── btxrd-segmentation-kd/
│   ├── common/                  # Shared utilities
│   └── multimodal/
├── btxrd-backend/ ⭐           # FastAPI application (root copy)
├── btxrd-frontend/ ⭐          # Next.js application (root copy)
├── deployment/ ⭐              # THIS FOLDER - deployment configs & copies
│   ├── btxrd-backend/          # Backend copy for deployment
│   ├── btxrd-frontend/         # Frontend copy for deployment
│   ├── .env.local              # Dev environment template
│   ├── .env.jetson             # Jetson environment (ready to use)
│   ├── docker-compose.yml      # Production compose
│   ├── docker-compose.dev.yml  # Dev override
│   ├── Dockerfile              # Backend image
│   ├── Dockerfile.frontend     # Frontend image
│   ├── export_onnx.py          # Model conversion script
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Deployment guide
└── config.yaml, docker-compose.yml, Dockerfile (root configs - deprecated)
```

### Backend Folder Structure
```
btxrd-backend/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── routes/                 # API endpoints
│   ├── models/                 # Data models
│   ├── services/               # Business logic
│   └── utils/                  # Helpers
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (runtime)
├── uploads/                    # User uploaded images
└── README.md
```

### Frontend Folder Structure
```
btxrd-frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   ├── components/             # React components
│   ├── api/                    # API client logic
│   └── utils/                  # Frontend helpers
├── public/                     # Static assets
├── package.json                # Dependencies
├── next.config.ts              # Next.js config
├── .env.local                  # Dev environment (runtime)
└── node_modules/               # Installed packages (generated)
```

### Key Files in Deployment Folder

| File | Purpose | Status |
|------|---------|--------|
| `.env.local` | Dev environment template | ✅ Ready - needs path edits |
| `.env.jetson` | Jetson environment | ✅ Ready - no edits needed |
| `docker-compose.yml` | Production compose | ⚠️ **HAS ISSUE** (see below) |
| `docker-compose.dev.yml` | Dev override | ⚠️ **PATH MISMATCH** (see below) |
| `Dockerfile` | Backend image recipe | ✅ OK |
| `Dockerfile.frontend` | Frontend image recipe | ⚠️ **MISMATCH** (see below) |
| `export_onnx.py` | Model converter | ✅ OK |
| `requirements.txt` | Python deps | ✅ OK |

---

## What Gets Copied to USB

> **Override for current setup:** keep model files inside `deployment/models` on USB. Do not use a separate top-level `models` folder for this run.

### Minimal Transfer Bundle (for Jetson or transfer to another machine)

```
USB_DRIVE_ROOT/
├── deployment/                              ⭐ COPY ENTIRE FOLDER
│   ├── btxrd-backend/                       # Backend app
│   ├── btxrd-frontend/                      # Frontend app
│   ├── .env.local
│   ├── .env.jetson
│   ├── docker-compose.yml (AFTER FIX)
│   ├── docker-compose.dev.yml (AFTER FIX)
│   ├── Dockerfile
│   ├── Dockerfile.frontend
│   ├── export_onnx.py
│   ├── requirements.txt
│   └── README.md
├── models/                                  ⭐ CREATE & POPULATE
│   ├── classification_student.pth           # Copy from FYP/BTXRD/combined_inference/models/
│   ├── segmentation_student.pth             # Copy from FYP/BTXRD/combined_inference/models/
│   └── gemma-2-2b-it-Q4_K_M.gguf            # Manually download from HuggingFace
└── README_USB.md                            # Quick start guide for USB
```

### What NOT to Copy

- `node_modules/` — regenerated by `npm install`
- `.next/` — regenerated by Next.js build
- `__pycache__/`, `.pyc` files — regenerated by Python
- `.venv/`, virtual environments — regenerated on target
- Entire BTXRD/ folder — only models needed
- Root configs (config.yaml, root docker-compose.yml) — use deployment versions

### File Sizes Reference

| Item | Size | Notes |
|------|------|-------|
| deployment/ folder (without node_modules) | ~100 MB | Source code + configs |
| classification_student.pth | ~5 MB | PyTorch model |
| segmentation_student.pth | ~8 MB | PyTorch model |
| gemma-2-2b-it-Q4_K_M.gguf | ~1.6 GB | LLM model (DOWNLOAD separately) |
| **Total to USB** | ~1.7 GB | All except node_modules |

---

## Path Changes & Environment Setup

### Development on Windows/Mac (from USB or local copy)

#### Step 1: Copy from USB to Local Machine
```bash
# On target machine
# Copy USB:/deployment → ~/vistai/deployment
# Copy USB:/models → ~/vistai/models
```

#### Step 2: Edit `.env` for Local Development

**File:** `deployment/.env.local` (copy to `deployment/.env` before running)

**Current Content (TEMPLATE):**
```env
MODEL_DIR=C:/Users/Nauman/Desktop/vistai/FYP/BTXRD/combined_inference/models
CLASSIFY_MODEL_FILE=classification_student.pth
SEGMENT_MODEL_FILE=segmentation_student.pth
LLAMA_CLI_PATH=/app/llama.cpp/llama-cli
LLM_MODEL_PATH=C:/Users/Nauman/Desktop/vistai/FYP/BTXRD/combined_inference/models/gemma-2-2b-it-Q4_K_M.gguf
```

**Changes Needed for USB Transfer (Windows Example):**
```env
# If USB drive is E:/
MODEL_DIR=E:/models
CLASSIFY_MODEL_FILE=classification_student.pth
SEGMENT_MODEL_FILE=segmentation_student.pth
LLAMA_CLI_PATH=/app/llama.cpp/llama-cli
LLM_MODEL_PATH=E:/models/gemma-2-2b-it-Q4_K_M.gguf
```

**Changes Needed for USB Transfer (Mac/Linux Example):**
```env
# If USB mounted at /Volumes/USB_DRIVE/
MODEL_DIR=/Volumes/USB_DRIVE/models
CLASSIFY_MODEL_FILE=classification_student.pth
SEGMENT_MODEL_FILE=segmentation_student.pth
LLAMA_CLI_PATH=/app/llama.cpp/llama-cli
LLM_MODEL_PATH=/Volumes/USB_DRIVE/models/gemma-2-2b-it-Q4_K_M.gguf
```

#### Step 3: Edit `docker-compose.dev.yml` for Local Paths

**File:** `deployment/docker-compose.dev.yml`

**Current Vector (USES RELATIVE PATHS - NOT COMPATIBLE WITH USB):**
```yaml
volumes:
  - ./btxrd-backend/models:/app/models:ro              # ❌ BROKEN - backend has no models/
  - ./uploads:/app/uploads
  - ./reports:/app/reports
```

**After USB Transfer (ABSOLUTE PATHS - USB EXAMPLE):**
```yaml
volumes:
  - E:/models:/app/models:ro                           # ✅ FIXED - points to USB models
  - ./uploads:/app/uploads
  - ./reports:/app/reports
```

**Alternative (Local Copy from USB):**
```yaml
volumes:
  - /home/user/vistai/models:/app/models:ro           # Linux/Mac
  - C:/Users/YourName/vistai/models:/app/models:ro    # Windows
```

### Jetson Nano Deployment (from USB)

#### `.env.jetson` (NO EDITS NEEDED - ALREADY CONFIGURED)
```env
MODEL_DIR=/data/models              # USB mounted at /data
CLASSIFY_MODEL_FILE=classify.trt     # TensorRT engines (convert on Jetson)
SEGMENT_MODEL_FILE=segment.trt       # TensorRT engines (convert on Jetson)
LLM_MODEL_PATH=/data/models/gemma-2-2b-it-Q4_K_M.gguf
```

#### Jetson Setup Workflow
1. Mount USB at `/data`:
   ```bash
   sudo mount /dev/sda1 /data
   ```
2. Copy deployment folder:
   ```bash
   cp -r /data/deployment ~/deployment
   cd ~/deployment
   ```
3. Convert ONNX → TensorRT (requires ONNX files):
   ```bash
   python export_onnx.py  # Creates .onnx files
   trtexec --onnx=/data/models/classification_student.onnx --saveEngine=/data/models/classify.trt --fp16
   ```
4. Docker compose uses `.env.jetson` by default (no edits needed)

---

## Known Issues & Fixes

### ⚠️ Issue 1: `docker-compose.yml` Frontend Build Path

**File:** `deployment/docker-compose.yml` (line 19)

**Current Code (BROKEN):**
```yaml
frontend:
  build:
    context: ./btxrd-frontend
    dockerfile: Dockerfile.frontend  # ❌ WRONG - file is named "Dockerfile"
  ports:
    - "3000:3000"
```

**Problem:**
- Context is `./btxrd-frontend/` ✅ Correct
- Dockerfile filename in that folder is `Dockerfile` (not `Dockerfile.frontend`)
- Docker will fail to find `Dockerfile.frontend` and build will fail

**Fix Required:**
```yaml
frontend:
  build:
    context: ./btxrd-frontend
    dockerfile: Dockerfile  # ✅ FIXED - correct filename
  ports:
    - "3000:3000"
```

### ⚠️ Issue 2: `docker-compose.dev.yml` Model Path Points to Wrong Location

**File:** `deployment/docker-compose.dev.yml` (line 10)

**Current Code (BROKEN):**
```yaml
volumes:
  - ./btxrd-backend/models:/app/models:ro  # ❌ WRONG - btxrd-backend has no models/ folder
  - ./uploads:/app/uploads
  - ./reports:/app/reports
```

**Problem:**
- `./btxrd-backend/models/` does not exist
- Models are in `../BTXRD/combined_inference/models/` (relative to deployment/)
- For USB, models are in sibling folder `../models/`

**Fix Required (for USB relative path):**
```yaml
volumes:
  - ./models:/app/models:ro            # ✅ FIXED - models folder at deployment root
```

OR (for dev absolute paths):
```yaml
volumes:
  - /path/to/models:/app/models:ro     # Absolute path on target machine
```

### ⚠️ Issue 3: Root Level Configs Are Misleading

**Files:** Root level `docker-compose.yml`, `Dockerfile`

**Problem:**
- There are old docker configs at project root
- Deployment folder has the correct, up-to-date versions
- Easy to accidentally use wrong versions

**Recommendation:**
- Ignore root versions for USB transfer
- Always use configs from `deployment/` folder
- Only use `deployment/docker-compose.yml` and `deployment/docker-compose.dev.yml`

---

## USB Transfer Workflow

### Phase 1: Prepare on Development Machine (Current Machine)

```bash
# 1. Create USB directory structure
mkdir -p /mnt/usb/deployment
mkdir -p /mnt/usb/models

# 2. Copy deployment folder (entire folder)
cp -r C:\Users\Nauman\Desktop\vistai\FYP\deployment /mnt/usb/

# 3. Copy models
cp C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\combined_inference\models\classification_student.pth /mnt/usb/models/
cp C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\combined_inference\models\segmentation_student.pth /mnt/usb/models/

# 4. Manually download LLM (one-time, ~1.6GB)
# Visit: https://huggingface.co/google/gemma-2-2b-it-GGUF
# Download: gemma-2-2b-it-Q4_K_M.gguf
# Save to: /mnt/usb/models/gemma-2-2b-it-Q4_K_M.gguf

# 5. Verify USB contents
ls -la /mnt/usb/
# Expected output:
# deployment/
# models/
#   ├── classification_student.pth
#   ├── segmentation_student.pth
#   └── gemma-2-2b-it-Q4_K_M.gguf
```

### Phase 2: Apply Fixes Before Transfer

**IMPORTANT:** Fix the two docker-compose issues BEFORE copying to USB.

```bash
# Fix docker-compose.yml (line 19)
# Change: dockerfile: Dockerfile.frontend
# To:     dockerfile: Dockerfile

# Fix docker-compose.dev.yml (line 10)
# Change: ./btxrd-backend/models:/app/models:ro
# To:     ./models:/app/models:ro
```

### Phase 3: Transfer to Target Machine

**Option A: From USB to Local Machine**
```bash
# On Windows
copy E:\deployment C:\Users\YourName\vistai\
copy E:\models C:\Users\YourName\vistai\

# On Mac/Linux
cp -r /Volumes/USB_DRIVE/deployment ~/vistai/
cp -r /Volumes/USB_DRIVE/models ~/vistai/
```

**Option B: For Jetson Nano**
```bash
# Mount USB
sudo mkdir -p /data
sudo mount /dev/sda1 /data

# Verify
ls -la /data/models/

# Models stay on USB, configs copied to home
cp -r /data/deployment ~/deployment
```

### Phase 4: Update Environment Variables

**For Local Windows/Mac Dev:**
```bash
cd ~/vistai/deployment
cp .env.local .env

# Edit .env with local paths:
# MODEL_DIR=C:/Users/YourName/vistai/models (Windows)
# MODEL_DIR=/Users/YourName/vistai/models (Mac)
```

**For Jetson Nano:**
```bash
cd ~/deployment
# .env.jetson already configured for /data/models
# Copy it as .env for docker compose to use
cp .env.jetson .env
```

### Phase 5: Run Application

**Local Windows/Mac (with Docker Compose):**
```bash
cd deployment
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

**Jetson Nano (Production):**
```bash
cd deployment
# First convert ONNX → TensorRT (if not done)
python3 export_onnx.py
trtexec --onnx=/data/models/classification_student.onnx --saveEngine=/data/models/classify.trt --fp16

# Then run
docker compose -f docker-compose.yml up -d
# Frontend: http://JETSON_IP:3000
```

---

## Post-Transfer Verification

### Checklist Before Running

- [ ] USB contains `/deployment/` folder with all configs
- [ ] USB contains `/models/` folder with 3 files:
  - `classification_student.pth`
  - `segmentation_student.pth`
  - `gemma-2-2b-it-Q4_K_M.gguf`
- [ ] Dockerfile in `deployment/btxrd-frontend/` is named `Dockerfile` (not `.frontend`)
- [ ] `.env.local` and `.env.jetson` exist in deployment/
- [ ] `docker-compose.yml` line 19 has `dockerfile: Dockerfile` (not `.frontend`)
- [ ] `docker-compose.dev.yml` line 10 has `./models:/app/models:ro` (not `./btxrd-backend/models`)
- [ ] `.env` file created from `.env.local` or `.env.jetson` with correct paths

### Runtime Tests

**After Starting Backend (Port 8000):**
```bash
# Check health
curl http://localhost:8000/health
# Expected: {"status":"ok","models_loaded":true}

# Check API docs
# Visit: http://localhost:8000/docs
```

**After Starting Frontend (Port 3000):**
```bash
# Visit http://localhost:3000 in browser
# Verify no console errors
# Try uploading a test image
```

**Check Logs:**
```bash
docker compose logs backend
docker compose logs frontend
docker compose ps  # Verify both running
```

---

## Troubleshooting & Common Errors

### Error: "Dockerfile.frontend not found"

**Cause:** `docker-compose.yml` looking for wrong filename  
**Fix:** Change line 19 from `dockerfile: Dockerfile.frontend` to `dockerfile: Dockerfile`

### Error: "Cannot mount ./btxrd-backend/models: no such file or directory"

**Cause:** `docker-compose.dev.yml` pointing to non-existent path  
**Fix:** Change line 10 from `./btxrd-backend/models` to `./models`

### Error: "MODEL_DIR path does not exist"

**Cause:** `.env` has wrong absolute path  
**Fix:** Edit `.env` to use correct local or USB path to models folder  
**Example (USB):**
```env
MODEL_DIR=E:/models
LLM_MODEL_PATH=E:/models/gemma-2-2b-it-Q4_K_M.gguf
```

### Error: "Models not loading on Jetson"

**Cause:** ONNX files not converted to TensorRT  
**Fix:** Run conversion before docker compose:
```bash
python3 export_onnx.py
trtexec --onnx=/data/models/classification_student.onnx --saveEngine=/data/models/classify.trt --fp16
trtexec --onnx=/data/models/segmentation_student.onnx --saveEngine=/data/models/segment.trt --fp16
```

### Error: "Frontend can't connect to backend"

**Cause:** `NEXT_PUBLIC_API_URL` environment variable mismatch  
**Fix (docker-compose.yml):**
```yaml
frontend:
  environment:
    - NEXT_PUBLIC_API_URL=http://localhost:8000  # Local dev
    # For Jetson: NEXT_PUBLIC_API_URL=http://JETSON_IP:8000
```

### Error: "Out of memory / GPU memory exceeded"

**Cause:** Jetson Nano memory limits or large batch size  
**Fix:** Reduce batch size, check memory:
```bash
# View memory usage
nvidia-smi
# Reduce LLM_CTX_SIZE or LLM_MAX_TOKENS in .env.jetson
```

---

## Summary of Required Actions Now

1. **Fix `deployment/docker-compose.yml` line 19:**
   - `dockerfile: Dockerfile.frontend` → `dockerfile: Dockerfile`

2. **Fix `deployment/docker-compose.dev.yml` line 10:**
   - `./btxrd-backend/models:/app/models:ro` → `./models:/app/models:ro`

3. **Keep models inside deployment:**
  - Ensure `deployment/models` contains:
    - `classification_student.onnx`
    - `segmentation_student.onnx`
    - `gemma-2-2b-it-Q4_K_M.gguf`
  - Copy full `deployment/` folder to USB

4. **On Target Machine:**
  - Update paths to `deployment/models`
  - Plug USB into Jetson
  - Convert ONNX to TensorRT (`.trt`)
  - Run docker compose

5. **Test:**
   - Backend health check: `curl http://localhost:8000/health`
   - Frontend: `http://localhost:3000`
   - Upload test image, verify inference

---

## Jetson Runbook (What To Do After Plugging USB)

Use these commands on Jetson in order.

```bash
# 1) Mount USB
sudo mkdir -p /data
sudo mount /dev/sda1 /data

# 2) Verify models are present inside deployment/models
ls -la /data/deployment/models

# 3) Copy deployment folder to home and enter it
cp -r /data/deployment ~/deployment
cd ~/deployment

# 4) Set runtime env
cp .env.jetson .env
# Edit .env to ensure these values:
# MODEL_DIR=/data/deployment/models
# CLASSIFY_MODEL_FILE=classify.trt
# SEGMENT_MODEL_FILE=segment.trt
# LLM_MODEL_PATH=/data/deployment/models/gemma-2-2b-it-Q4_K_M.gguf

# 5) Convert ONNX to TensorRT
trtexec --onnx=/data/deployment/models/classification_student.onnx --saveEngine=/data/deployment/models/classify.trt --fp16
trtexec --onnx=/data/deployment/models/segmentation_student.onnx --saveEngine=/data/deployment/models/segment.trt --fp16

# 6) Build and run
docker compose up -d --build

# 7) Test and monitor
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
curl http://localhost:8000/health
```

Expected URLs:
- Frontend: `http://JETSON_IP:3000`
- Backend docs: `http://JETSON_IP:8000/docs`

---

## Quick Reference: Path Mapping by Scenario

| Scenario | MODEL_DIR | LLM_MODEL_PATH | Docker Volume |
|----------|-----------|-----------------|---|
| **Dev (USB on Windows)** | `E:/deployment/models` | `E:/deployment/models/gemma...gguf` | `E:/deployment/models:/app/models:ro` |
| **Dev (USB on Mac)** | `/Volumes/USB/deployment/models` | `/Volumes/USB/deployment/models/gemma...gguf` | `/Volumes/USB/deployment/models:/app/models:ro` |
| **Dev (Local Copy)** | `C:/Users/User/vistai/deployment/models` | `C:/Users/User/vistai/deployment/models/gemma...gguf` | `C:/Users/User/vistai/deployment/models:/app/models:ro` |
| **Jetson (USB @ /data)** | `/data/deployment/models` | `/data/deployment/models/gemma...gguf` | `/data/deployment/models:/app/models:ro` |
| **Jetson (Local Copy)** | `/home/jetson/deployment/models` | `/home/jetson/deployment/models/gemma...gguf` | `/home/jetson/deployment/models:/app/models:ro` |

---

## References

- [Deployment README](./README.md)
- [Quick Start Guide](./QUICKSTART.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- [Main Project README](../README.md)

---

**End of Context Document**

For Copilot/Codex: Use this document as the single source of truth for understanding the BTXRD deployment structure, USB transfer plan, and all required path changes. When asked to fix paths or prepare for USB transfer, refer to this document for exact file locations, changes needed, and testing procedures.
