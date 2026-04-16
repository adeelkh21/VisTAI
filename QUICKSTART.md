# VistAI — Quick Start Guide

## 🚀 Start Both Servers (5 minutes)

### Terminal 1: Start Backend
```bash
cd btxrd-backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Add Groq for chat (if not installed)
pip install groq

# Set Groq API Key (Windows)
$env:GROQ_API_KEY="your_key_from_https://console.groq.com"

# Start backend server
python -m uvicorn app.main:app --reload --port 8000
# ✅ Backend running at http://localhost:8000
# ✅ API docs at http://localhost:8000/docs
```

### Terminal 2: Start Frontend
```bash
cd btxrd-frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
# ✅ Frontend running at http://localhost:3000
```

---

## 🧪 Quick Tests

### Test Backend API
```bash
# Terminal 3
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Test classification
curl -X POST http://localhost:8000/api/mobilenet/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Test Frontend
Open browser: **http://localhost:3000**

1. Upload an image via drag-and-drop
2. Click "Run Classification"
3. See results appear
4. Click "Chat with AI" (if Groq API key is set)

---

## ⚙️ Set Up Groq API (2 minutes)

### 1. Get Free API Key
```
Go to: https://console.groq.com
Click "Create API Key"
Copy the key
```

### 2. Set Environment Variable
```bash
# Windows PowerShell
$env:GROQ_API_KEY="paste_your_key_here"

# Windows Command Prompt
set GROQ_API_KEY=paste_your_key_here

# Linux/Mac
export GROQ_API_KEY="paste_your_key_here"
```

### 3. Verify It's Set
```bash
# Windows
echo $env:GROQ_API_KEY

# Linux/Mac
echo $GROQ_API_KEY
```

### 4. Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/chat/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is osteosarcoma?",
    "detected_disease": "osteosarcoma",
    "confidence": 0.94,
    "conversation_history": []
  }'
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Backend refused connection" | Ensure Terminal 1 shows "Uvicorn running on http://0.0.0.0:8000" |
| "Cannot find module" (Frontend) | Run `npm install` in btxrd-frontend directory |
| "ModuleNotFoundError" (Backend) | Run `pip install -r requirements.txt` in btxrd-backend |
| Chat not working | Check GROQ_API_KEY is set: `echo $env:GROQ_API_KEY` |
| Styles not loading | Run `npm run dev` (rebuilds Tailwind) |
| "Prediction failed" error | Check backend is running and image file is < 16MB |

---

## 🎯 Feature Overview

### Quick Classification
1. Upload image
2. See instant prediction (< 1 second)
3. View confidence score
4. See top-5 probabilities

### Full Analysis
1. Click "Full Analysis" button
2. Get segmentation maps
3. View detailed metrics
4. Export report (coming soon)

### Chat with AI
1. Get prediction
2. Click "Chat with Image"
3. Ask medical questions
4. AI responds with context
5. Continue conversation

---

## 📊 API Endpoints (When Backend Running)

Check all endpoints: **http://localhost:8000/docs**

### Quick Classification
```
POST /api/mobilenet/predict
Input: image file
Output: {class_name, confidence, probabilities}
```

### Full Analysis
```
POST /api/inference
Input: image file
Output: {classification, segmentation, metrics}
```

### Chat
```
POST /api/chat/predict
Input: {message, detected_disease, confidence, conversation_history}
Output: {response, tokens_used}
```

---

## 💾 Project Structure

```
VistAI/
├── btxrd-frontend/      # Next.js React app (http://localhost:3000)
│   └── src/components/vistai/
│       ├── Header.tsx
│       ├── ImageUploadSection.tsx
│       ├── ResultsPanel.tsx
│       └── ChatPanel.tsx
│
└── btxrd-backend/       # FastAPI server (http://localhost:8000)
    └── app/
        ├── main.py       # Entry point
        ├── api/
        │   ├── mobilenet_service.py
        │   └── chat_groq_integration.py
        └── services/
            └── inference.py
```

---

## 🔑 Key Files to Know

| File | Purpose |
|------|---------|
| `btxrd-frontend/src/app/page.tsx` | Main frontend component |
| `btxrd-frontend/next.config.ts` | API proxy configuration |
| `btxrd-backend/app/main.py` | Backend entry point |
| `btxrd-backend/app/api/chat_groq_integration.py` | Chat endpoint template |

---

## 📱 Production Checklist

Before deploying to Jetson or production:

- [ ] Both servers start without errors
- [ ] Image upload works
- [ ] Classification returns predictions
- [ ] Chat endpoint responds (if using Groq)
- [ ] Response times acceptable (< 5s)
- [ ] No console errors in browser DevTools
- [ ] Mobile layout responsive
- [ ] All features tested on production data

---

## 📞 Support

Issues? Check:
1. **Backend logs**: Terminal 1 (should show incoming requests)
2. **Frontend logs**: Browser DevTools Console (F12)
3. **Network tab**: Check API calls are going to port 8000
4. **Groq status**: Check https://status.groq.com

---

**VistAI Development Environment**  
Ready to ship! 🚀
