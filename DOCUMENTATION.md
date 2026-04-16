# 📖 VistAI Documentation Hub

Complete documentation for the VistAI bone tumor detection platform. This is your central reference point.

## 🎯 Start Here

**New to VistAI?** Follow this path:

1. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (5 min) - Understand what VistAI is
2. Follow [QUICKSTART.md](QUICKSTART.md) (5 min) - Get it running locally
3. Explore the UI at http://localhost:3000
4. Check [API_REFERENCE.md](API_REFERENCE.md) for endpoint details
5. Deploy to production using [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📚 Documentation Structure

### 🟢 Getting Started (User-Friendly)

| Document | Purpose | Time | Audience |
|----------|---------|------|----------|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Complete project summary, features, architecture | 5-10 min | Everyone |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute local setup and basic testing | 5 min | Developers |
| [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) | UI components, styling, Groq integration details | 10-15 min | Frontend devs |

### 🟡 Technical Reference (Comprehensive)

| Document | Purpose | Time | Audience |
|----------|---------|------|----------|
| [API_REFERENCE.md](API_REFERENCE.md) | All endpoints, request/response formats, examples | 20 min | API developers |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Docker, Jetson Nano, Kubernetes, Cloud deployment | 30 min | DevOps/SRE |
| [btxrd-backend/README.md](btxrd-backend/README.md) | Backend configuration, environment setup | 10 min | Backend devs |
| [btxrd-frontend/README.md](btxrd-frontend/README.md) | Frontend configuration, build process | 10 min | Frontend devs |

### 🔵 Deep Dives (Technical Details)

| Document | Purpose | Time | Audience |
|----------|---------|------|----------|
| [BTXRD/README.md](BTXRD/README.md) | Model architecture, training pipeline, evaluation | 15 min | ML engineers |
| [BTXRD/btxrd-classification-kd/README.md](BTXRD/btxrd-classification-kd/README.md) | Classification model (MobileNetV2, ConvNeXt) | 10 min | ML engineers |
| [BTXRD/btxrd-segmentation-kd/README.md](BTXRD/btxrd-segmentation-kd/README.md) | Segmentation model (Swin-UNet) details | 10 min | ML engineers |
| [BTXRD/KNOWLEDGE_DISTILLATION_README.md](BTXRD/KNOWLEDGE_DISTILLATION_README.md) | KD pipeline, model compression | 15 min | ML engineers |

---

## 🗺️ Navigation by Role

### 👨‍💻 Frontend Developer
```
1. START: PROJECT_OVERVIEW.md
2. SETUP: QUICKSTART.md
3. CODE: btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md
4. DEPLOY: DEPLOYMENT_GUIDE.md → Docker section
5. REFERENCE: API_REFERENCE.md
```

**Key Files to Know:**
- `btxrd-frontend/src/app/page.tsx` - Main component
- `btxrd-frontend/src/components/vistai/` - UI components (Header, ImageUpload, Results, Chat)
- `btxrd-frontend/next.config.ts` - API proxy configuration
- `btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md` - Design system & component docs

---

### 🔧 Backend Developer
```
1. START: PROJECT_OVERVIEW.md
2. SETUP: QUICKSTART.md
3. CODE: btxrd-backend/README.md
4. API: API_REFERENCE.md
5. DEPLOY: DEPLOYMENT_GUIDE.md → Docker section
```

**Key Files to Know:**
- `btxrd-backend/app/main.py` - Entry point
- `btxrd-backend/app/api/` - Endpoint implementations
- `btxrd-backend/app/services/` - ML services
- `btxrd-backend/requirements.txt` - Dependencies
- `btxrd-backend/app/api/chat_groq_integration.py` - Chat endpoint template

---

### 🤖 ML / AI Engineer
```
1. START: PROJECT_OVERVIEW.md
2. MODELS: BTXRD/README.md
3. CLASSIFICATION: BTXRD/btxrd-classification-kd/README.md
4. SEGMENTATION: BTXRD/btxrd-segmentation-kd/README.md
5. KD PIPELINE: BTXRD/KNOWLEDGE_DISTILLATION_README.md
```

**Key Files to Know:**
- `BTXRD/mobilenet/outputs/best_model.pth` - Trained MobileNetV2
- `BTXRD/btxrd-classification-kd/configs/` - Training configs
- `BTXRD/btxrd-segmentation-kd/train_kd.py` - Segmentation training
- `BTXRD/common/metrics.py` - Evaluation metrics
- `BTXRD/common/utils.py` - Utility functions

---

### 🚀 DevOps / Platform Engineer
```
1. START: PROJECT_OVERVIEW.md
2. DEPLOYMENT: DEPLOYMENT_GUIDE.md
  a. Docker section (10 min)
  b. Jetson Nano section (30 min)
  c. Kubernetes section (20 min)
  d. Cloud section (15 min)
3. API: API_REFERENCE.md → Health endpoints
4. MONITORING: DEPLOYMENT_GUIDE.md → Monitoring section
```

**Key Files to Know:**
- `docker-compose.yml` - Multi-container orchestration
- `Dockerfile` - Backend container image
- `btxrd-frontend/Dockerfile` - Frontend container image
- `deployment/` - Cloud deployment files
- `DEPLOYMENT_GUIDE.md` - Complete deployment reference

---

### 🏥 Medical Professional / End User
```
1. START: PROJECT_OVERVIEW.md
2. HOW TO USE: QUICKSTART.md
3. FEATURES: PROJECT_OVERVIEW.md → Use Cases & Capabilities sections
4. TROUBLESHOOTING: QUICKSTART.md → Troubleshooting section
```

**Key Points:**
- Upload X-ray images via drag-and-drop
- Get instant classification (< 1 second)
- Ask questions via AI chat feature
- See results with confidence scores
- Export reports (coming soon)

---

## 🔍 Find Documentation By Topic

### Setup & Installation
- **Quick setup**: [QUICKSTART.md](QUICKSTART.md)
- **Frontend setup**: [btxrd-frontend/README.md](btxrd-frontend/README.md)
- **Backend setup**: [btxrd-backend/README.md](btxrd-backend/README.md)
- **Dependencies**: Check `requirements.txt` and `package.json` files

### API Usage
- **Endpoint reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Examples**: [API_REFERENCE.md](API_REFERENCE.md) → Usage Examples section
- **Health checks**: [API_REFERENCE.md](API_REFERENCE.md) → Health & Status section

### Deployment
- **Local**: [QUICKSTART.md](QUICKSTART.md)
- **Docker**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Docker section
- **Jetson Nano**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Jetson section
- **Kubernetes**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Kubernetes section
- **Cloud (AWS/GCP/Azure)**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Cloud section

### UI/Frontend
- **Component architecture**: [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md)
- **Design system**: [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) → Design Highlights
- **Styling**: [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) → Styling System
- **Chat integration**: [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) → Chat Endpoint

### Machine Learning
- **Model overview**: [BTXRD/README.md](BTXRD/README.md)
- **Classification**: [BTXRD/btxrd-classification-kd/README.md](BTXRD/btxrd-classification-kd/README.md)
- **Segmentation**: [BTXRD/btxrd-segmentation-kd/README.md](BTXRD/btxrd-segmentation-kd/README.md)
- **Knowledge distillation**: [BTXRD/KNOWLEDGE_DISTILLATION_README.md](BTXRD/KNOWLEDGE_DISTILLATION_README.md)
- **Training**: [BTXRD/btxrd-classification-kd/train_kd.py](BTXRD/btxrd-classification-kd/train_kd.py)

### Troubleshooting
- **Common issues**: [QUICKSTART.md](QUICKSTART.md) → Troubleshooting
- **Debug mode**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Troubleshooting
- **Performance**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Monitoring & Logging

### Chat Feature
- **Setup**: [QUICKSTART.md](QUICKSTART.md) → Set Up Groq API
- **Integration**: [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrc-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) → Groq Integration Details
- **API endpoint**: [API_REFERENCE.md](API_REFERENCE.md) → Chat with AI

---

## 📋 File Structure Overview

```
VistAI/
├── PROJECT_OVERVIEW.md                    # ← START HERE
├── QUICKSTART.md                          # ← 5-minute setup
├── API_REFERENCE.md                       # ← All endpoints
├── DEPLOYMENT_GUIDE.md                    # ← Production deployment
│
├── btxrd-frontend/
│   ├── README.md                          # Frontend config
│   ├── FRONTEND_IMPLEMENTATION_GUIDE.md   # ← Component guide
│   ├── src/
│   │   ├── app/page.tsx                   # Main component
│   │   └── components/vistai/
│   │       ├── Header.tsx
│   │       ├── ImageUploadSection.tsx
│   │       ├── ResultsPanel.tsx
│   │       └── ChatPanel.tsx
│   └── next.config.ts                     # API proxy config
│
├── btxrd-backend/
│   ├── README.md                          # Backend config
│   ├── requirements.txt
│   └── app/
│       ├── main.py                        # Entry point
│       ├── api/                           # Endpoints
│       └── services/                      # ML services
│
├── BTXRD/
│   ├── README.md                          # Model overview
│   ├── KNOWLEDGE_DISTILLATION_README.md   # KD pipeline
│   ├── btxrd-classification-kd/
│   │   ├── README.md
│   │   └── train_kd.py
│   ├── btxrd-segmentation-kd/
│   │   ├── README.md
│   │   └── train_kd.py
│   └── common/
│       ├── metrics.py
│       ├── utils.py
│       └── visualization.py
│
└── deployment/
    ├── docker-compose.yml
    ├── Dockerfile
    └── README.md
```

---

## ⚡ Common Tasks Quick Reference

### I want to...

#### ✏️ Set up VistAI locally
→ [QUICKSTART.md](QUICKSTART.md)

#### 📱 Understand the UI components
→ [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md)

#### 🔌 Add a new API endpoint
→ [API_REFERENCE.md](API_REFERENCE.md) + [btxrd-backend/README.md](btxrd-backend/README.md)

#### 🚀 Deploy to production
→ [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

#### 🎮 Deploy to Jetson Nano
→ [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Jetson Nano section

#### 💬 Set up the chat feature
→ [QUICKSTART.md](QUICKSTART.md) → Set Up Groq API section

#### 🤖 Train a new model
→ [BTXRD/README.md](BTXRD/README.md) + [BTXRD/btxrd-classification-kd/README.md](BTXRD/btxrd-classification-kd/README.md)

#### 📊 Understand the API
→ [API_REFERENCE.md](API_REFERENCE.md)

#### 📈 Monitor production
→ [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Monitoring & Logging

#### 🐛 Debug an issue
→ [QUICKSTART.md](QUICKSTART.md) → Troubleshooting + [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) → Debug Mode

---

## 🎓 Learning Path

### For Beginners
1. [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Understand the system
2. [QUICKSTART.md](QUICKSTART.md) - Get it running
3. [btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md](btxrd-frontend/FRONTEND_IMPLEMENTATION_GUIDE.md) - Explore the UI

### For Developers
1. All of Beginners path
2. [API_REFERENCE.md](API_REFERENCE.md) - Learn the API
3. [btxrd-backend/README.md](btxrd-backend/README.md) - Backend details
4. [btxrd-frontend/README.md](btxrd-frontend/README.md) - Frontend details

### For Full Stack Engineers
1. All of Developers path
2. [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment strategies
3. [BTXRD/README.md](BTXRD/README.md) - Model pipeline

### For ML Engineers
1. [BTXRD/README.md](BTXRD/README.md) - Overview
2. [BTXRD/btxrd-classification-kd/README.md](BTXRD/btxrd-classification-kd/README.md) - Classification
3. [BTXRD/btxrd-segmentation-kd/README.md](BTXRD/btxrd-segmentation-kd/README.md) - Segmentation
4. [BTXRD/KNOWLEDGE_DISTILLATION_README.md](BTXRD/KNOWLEDGE_DISTILLATION_README.md) - Model compression

---

## 🔗 External Resources

### Official Websites
- **Groq API**: https://console.groq.com
- **FastAPI**: https://fastapi.tiangolo.com
- **Next.js**: https://nextjs.org
- **PyTorch**: https://pytorch.org
- **Docker**: https://docker.com
- **Kubernetes**: https://kubernetes.io

### Useful Tools
- **API Testing**: Postman, curl, Thunder Client
- **Browser DevTools**: F12 to inspect frontend
- **Model Visualization**: TensorBoard, Weights & Biases
- **Performance Profiling**: py-spy, Chrome DevTools

---

## ✅ Documentation Checklist

This documentation covers:

- ✅ Project overview and features
- ✅ Quick start guide (5 minutes)
- ✅ Complete API reference with examples
- ✅ Frontend component documentation
- ✅ Backend configuration guide
- ✅ ML model training and evaluation
- ✅ Local development setup
- ✅ Docker deployment
- ✅ Jetson Nano edge deployment
- ✅ Kubernetes deployment
- ✅ Cloud deployment (AWS/GCP/Azure)
- ✅ Troubleshooting guide
- ✅ Performance monitoring
- ✅ Security checklist
- ✅ Contributing guidelines

---

## 📞 Support

### Getting Help

1. **Documentation**: Check this hub first
2. **API Docs**: http://localhost:8000/docs (when running)
3. **Examples**: See [API_REFERENCE.md](API_REFERENCE.md) → Usage Examples
4. **Logs**: Check terminal output for errors
5. **Issues**: Create a GitHub issue with logs

### Reporting Issues

Include:
- Error message (exact text)
- Logs (paste relevant lines)
- Steps to reproduce
- Environment (OS, Python version, GPU/CPU)
- What you've already tried

---

## 📈 Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Apr 12, 2026 | Production-grade UI redesign, chat integration, comprehensive docs |
| 1.5 | Apr 11, 2026 | MobileNetV2 model, Python testing tools |
| 1.0 | Apr 1, 2026 | Initial project setup, model training |

---

## 🎉 You're All Set!

Everything is documented and ready to go. Pick any of the guides above and start building!

**Recommended first steps:**
1. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (5 min)
2. Follow [QUICKSTART.md](QUICKSTART.md) (5 min)
3. Explore the running app at http://localhost:3000
4. Check the API at http://localhost:8000/docs

---

**Last Updated**: April 12, 2026  
**Status**: Production Ready ✅
