# VistAI — Complete Deployment Guide

Comprehensive guide for deploying VistAI to production, Jetson Nano, and cloud environments.

---

## 🎯 Quick Reference

| Environment | Time | Complexity | Docs |
|-------------|------|-----------|------|
| Local Development | 5 min | Easy | QUICKSTART.md |
| Docker Compose | 10 min | Medium | [Docker Section](#docker-deployment) |
| Jetson Nano | 30 min | Medium | [Jetson Section](#jetson-nano-deployment) |
| Cloud (AWS/GCP) | 45 min | Hard | [Cloud Section](#cloud-deployment) |
| Production Kubernetes | 60 min | Hard | [K8s Section](#kubernetes-deployment) |

---

## Local Development (5 minutes)

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn
- 4GB RAM minimum

### Setup

1. **Install Backend Dependencies**
```bash
cd btxrd-backend
pip install -r requirements.txt
pip install groq  # For chat feature
```

2. **Install Frontend Dependencies**
```bash
cd ../btxrd-frontend
npm install
```

3. **Set Environment Variables**
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_key_from_groq.com"

# Linux/Mac
export GROQ_API_KEY="your_key_from_groq.com"
```

4. **Start Servers**
```bash
# Terminal 1: Backend
cd btxrd-backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd btxrd-frontend
npm run dev
```

5. **Access Application**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Docker Deployment

### Single Container (Simple)

```bash
# Build image
docker build -t vistai:latest \
  -f deployment/Dockerfile \
  .

# Run container
docker run -d \
  -p 8000:8000 \
  -e GROQ_API_KEY="your_key" \
  -v ./uploads:/app/uploads \
  --name vistai \
  vistai:latest
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build:
      context: .
      dockerfile: deployment/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - DEBUG=false
    volumes:
      - ./outputs:/app/outputs
      - ./uploads:/app/uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Frontend
  frontend:
    build:
      context: ./btxrd-frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - BACKEND_URL=http://backend:8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Jetson Nano Deployment

### Hardware Requirements
- Jetson Nano 4GB
- 64GB microSD card (Class 10 recommended)
- USB-C power supply (5V/2A minimum)
- Network connection (Ethernet preferred)

### Installation Steps

#### Step 1: Prepare Jetson Nano
```bash
# SSH into Jetson
ssh nvidia@jetson.local

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
  python3-pip \
  python3-venv \
  build-essential \
  libssl-dev \
  libffi-dev \
  curl \
  git

# Increase swap (important for model loading)
sudo apt install -y dphys-swapfile
sudo nano /etc/dphys-swapfile
  # Change: CONF_SWAPSIZE=2048
sudo systemctl restart dphys-swapfile
```

#### Step 2: Clone Repository
```bash
cd ~
git clone <your-repo-url> vistai
cd vistai
```

#### Step 3: Setup Backend on Jetson
```bash
cd btxrd-backend

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (will take ~10 minutes)
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install groq

# Verify models are downloaded
python -c "import torch; print(f'PyTorch ready: {torch.cuda.is_available()}')"
```

#### Step 4: Setup Frontend on Jetson
```bash
cd ../btxrd-frontend

# Install Node.js on Jetson (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install dependencies
npm install

# Build for production
npm run build
```

#### Step 5: Configuration for Jetson
```bash
cd ~/vistai

# Create .env.jetson
cat > .env.jetson <<EOF
GROQ_API_KEY=your_key_here
BACKEND_URL=http://localhost:8000
NODE_ENV=production
NEXT_PUBLIC_API_URL=http://jetson.local:8000
EOF

# Apply environment
export $(cat .env.jetson | xargs)
```

#### Step 6: Start Services on Jetson
```bash
# Option A: Manual startup (two terminals)

# Terminal 1: Backend
cd ~/vistai/btxrd-backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd ~/vistai/btxrd-frontend
npm start
# ✅ Access at http://jetson.local:3000
```

#### Step 7: Systemd Services (Auto-start)

**Backend Service**
```bash
sudo nano /etc/systemd/system/vistai-backend.service
```

```ini
[Unit]
Description=VistAI Backend
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/vistai/btxrd-backend
ExecStart=/home/nvidia/vistai/btxrd-backend/venv/bin/python \
  -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
Environment="PATH=/home/nvidia/vistai/btxrd-backend/venv/bin"
Environment="GROQ_API_KEY=your_key_here"

[Install]
WantedBy=multi-user.target
```

**Frontend Service**
```bash
sudo nano /etc/systemd/system/vistai-frontend.service
```

```ini
[Unit]
Description=VistAI Frontend
After=network.target vistai-backend.service

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/vistai/btxrd-frontend
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/usr/bin"
Environment="NODE_ENV=production"

[Install]
WantedBy=multi-user.target
```

**Enable and Start Services**
```bash
# Enable services
sudo systemctl enable vistai-backend.service
sudo systemctl enable vistai-frontend.service

# Start services
sudo systemctl start vistai-backend.service
sudo systemctl start vistai-frontend.service

# Check status
sudo systemctl status vistai-backend.service
sudo systemctl status vistai-frontend.service

# View logs
journalctl -u vistai-backend.service -f
journalctl -u vistai-frontend.service -f
```

#### Step 8: Verify on Jetson
```bash
# From your laptop, access:
http://jetson.local:3000

# Or from Jetson:
curl http://localhost:8000/health
curl http://localhost:3000
```

### Jetson Performance Tuning

**Enable Max Performance**
```bash
# Set to max clock speed (higher power consumption)
sudo nvpmodel -m 0  # 5W mode (default)
sudo nvpmodel -m 1  # 10W mode (high performance)

# Check current mode
sudo nvpmodel -q
```

**Monitor Resources**
```bash
# Real-time GPU/CPU monitoring
jtop  # Install: pip3 install jetson-stats

# Simple monitoring
watch -n 1 nvidia-smi
```

**Optimize Model Loading**
```python
# In app/main.py, add:
import psutil
print(f"Memory available: {psutil.virtual_memory().available / 1e9:.1f} GB")

# Models print their size on load
print("Loading MobileNetV2...")
# Should fit in Jetson's 4GB with swap
```

---

## Cloud Deployment

### AWS EC2

#### 1. Launch Instance
```bash
# Recommended: g4dn.xlarge (GPU) or t3.medium (CPU)
# AMI: Ubuntu 22.04 LTS
# Storage: 50GB
```

#### 2. Connect & Setup
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repository
git clone <your-repo-url> vistai
cd vistai

# Build and run
docker-compose up -d
```

#### 3. Access Application
```
http://your-instance-ip:3000
```

#### 4. Configure Auto-scaling
```bash
# Create snapshot of running container
docker commit vistai-backend vistai-backend:v1
aws ecr create-repository --repository-name vistai-backend

# Push to ECR
docker tag vistai-backend:v1 YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/vistai-backend:v1
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/vistai-backend:v1
```

### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/vistai

# Deploy
gcloud run deploy vistai \
  --image gcr.io/YOUR_PROJECT/vistai \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=your_key \
  --port 8000
```

### Azure Container Instances

```bash
# Build image
docker build -t vistai:latest .

# Push to ACR
az acr build --registry YOUR_REGISTRY --image vistai:latest .

# Deploy
az container create \
  --resource-group YOUR_RG \
  --name vistai \
  --image YOUR_REGISTRY.azurecr.io/vistai:latest \
  --ports 3000 8000 \
  --environment-variables GROQ_API_KEY=your_key
```

---

## Kubernetes Deployment

### Prerequisites
```bash
kubectl version --client
helm version
```

### Helm Chart

**Chart Structure**
```
vistai-helm/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── backend-service.yaml
│   ├── frontend-service.yaml
│   ├── ingress.yaml
│   └── configmap.yaml
```

### Deploy to Kubernetes

```bash
# Add Helm repo
helm repo add vistai <your-helm-repo>
helm repo update

# Install chart
helm install vistai vistai/vistai \
  --namespace vistai \
  --create-namespace \
  --set groqApiKey="your_key" \
  --set image.tag="latest"

# Verify deployment
kubectl get pods -n vistai
kubectl get services -n vistai

# Access application
kubectl port-forward -n vistai svc/vistai-frontend 3000:3000 &
# Visit: http://localhost:3000
```

### Scaling

```bash
# Scale backend
kubectl scale deployment vistai-backend \
  -n vistai \
  --replicas=3

# Auto-scaling
kubectl autoscale deployment vistai-backend \
  -n vistai \
  --min=2 --max=10 \
  --cpu-percent=80
```

---

## Monitoring & Logging

### Health Checks

```bash
# Manual
curl http://localhost:8000/health
curl http://localhost:3000

# Automated monitoring script
while true; do
  echo "Backend: $(curl -s http://localhost:8000/health | jq '.status')"
  echo "Frontend: $(curl -s http://localhost:3000 -o /dev/null -w '%{http_code}')"
  sleep 10
done
```

### Logging

**Backend Logs**
```bash
# View logs from Docker container
docker logs -f vistai-backend

# View systemd logs (Jetson)
journalctl -u vistai-backend.service -f

# View Kubernetes logs
kubectl logs -f deployment/vistai-backend -n vistai
```

**Frontend Logs**
```bash
# Browser console (F12)
# Then check DevTools Console tab

# Browser network tab
# Monitor API calls to /api/mobilenet/predict, etc.
```

### Profiling

**Backend Performance**
```python
# Add timing decorator to app/main.py
import time
from functools import wraps

def timeit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print(f"{f.__name__} took {time.time()-start:.3f}s")
        return result
    return decorated

@timeit
@app.post("/api/mobilenet/predict")
async def predict(file):
    ...
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| 502 Bad Gateway | Backend crashed | `docker-compose logs backend` |
| "Cannot connect to server" | Frontend not running | `npm run dev` or `npm start` |
| "Prediction failed" | Model not loaded | Check `GROQ_API_KEY`, restart |
| Out of memory on Jetson | Swap not configured | `sudo systemctl restart dphys-swapfile` |
| Chat not responding | Groq API key invalid | Verify key at groq.com console |
| High latency | GPU not used | Check `nvidia-smi` for GPU usage |

### Debug Mode

```bash
# Backend debug
DEBUG=true python -m uvicorn app.main:app --reload

# Frontend debug
NEXT_PUBLIC_DEBUG=true npm run dev

# Check GPU usage
nvidia-smi watch -n 1
```

---

## Security Checklist

- [ ] GROQ_API_KEY stored in environment, not git
- [ ] HTTPS enabled in production
- [ ] CORS configured properly
- [ ] Rate limiting enabled
- [ ] Input validation on all endpoints
- [ ] Error messages don't expose internals
- [ ] Database credentials not in logs
- [ ] Firewall rules restrict access
- [ ] Regular security updates applied
- [ ] API keys rotated quarterly

---

## Performance Targets

| Metric | Target | Alert |
|--------|--------|-------|
| API Response Time | < 500ms | > 2s |
| Model Inference | < 100ms | > 500ms |
| Chat Response | 1-2s | > 10s |
| Memory Usage | < 2GB | > 3.5GB |
| GPU Utilization | > 50% | < 20% |
| Disk Space | > 20% free | < 5% free |

---

## Maintenance

### Daily
- [ ] Check error logs for exceptions
- [ ] Verify health check endpoints
- [ ] Monitor disk space usage

### Weekly
- [ ] Review performance metrics
- [ ] Check for any bottlenecks
- [ ] Test backup/restore procedures

### Monthly
- [ ] Update dependencies
- [ ] Review security logs
- [ ] Rotate API keys
- [ ] Test disaster recovery

### Annually
- [ ] Security audit
- [ ] Load testing
- [ ] Architecture review
- [ ] Cost optimization

---

## Rollback Procedure

```bash
# Docker rollback
docker pull vistai:v1.0  # Previous version
docker-compose down
docker-compose up -d

# Kubernetes rollback
kubectl rollout history deployment/vistai-backend -n vistai
kubectl rollout undo deployment/vistai-backend -n vistai --to-revision=2

# Git rollback
git revert <commit-hash>
git push origin main
```

---

**VistAI Deployment Guide v2.0**  
**Last Updated**: April 12, 2026
