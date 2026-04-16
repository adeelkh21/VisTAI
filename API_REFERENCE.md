# VistAI API Reference

Complete API documentation for the VistAI backend.

---

## Base URL

```
http://localhost:8000
```

## Documentation

- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

---

## Health & Status Endpoints

### Health Check
```
GET /health

Response 200:
{"status": "healthy"}
```

### Model Status
```
GET /model-status

Response 200:
{
  "mobilenet": "loaded",
  "classification": "loaded",
  "segmentation": "loaded",
  "status": "all_models_ready"
}
```

### Server Info
```
GET /info

Response 200:
{
  "version": "2.0",
  "environment": "production",
  "debug": false,
  "available_models": [
    "mobilenet",
    "convnext",
    "swin_unet"
  ]
}
```

---

## Image Classification (Quick)

### MobileNet Quick Classification
```
POST /api/mobilenet/predict

Content-Type: multipart/form-data

Request:
- file: <image file> (jpg, png, webp)

Response 200:
{
  "class_id": 0,
  "class_name": "osteosarcoma",
  "confidence": 0.9445,
  "probabilities": {
    "osteosarcoma": 0.9445,
    "osteochondroma": 0.0312,
    "fibroma": 0.0147,
    "giant_cell": 0.0096,
    "lipoma": 0.0000,
    "hemangioma": 0.0000,
    "chondrosarcoma": 0.0000,
    "aneurysmal": 0.0000,
    "ganglion": 0.0000
  },
  "inference_time_ms": 45,
  "model": "MobileNetV2"
}

Response 400:
{
  "detail": "No file provided"
}

Response 415:
{
  "detail": "Unsupported file type"
}
```

**Performance**: ~45ms per image on GPU, ~200ms on CPU

---

## Full Analysis

### Comprehensive Inference
```
POST /api/inference

Content-Type: multipart/form-data

Request:
- file: <image file>
- include_segmentation: true (optional, default: false)

Response 200:
{
  "classification": {
    "class_name": "osteosarcoma",
    "confidence": 0.9445,
    "probabilities": { ... }
  },
  "segmentation": {
    "mask": "base64_encoded_image",
    "tumor_area_pixels": 15243,
    "bounding_box": {
      "x": 120,
      "y": 85,
      "width": 180,
      "height": 220
    }
  },
  "analysis": {
    "risk_score": 0.89,
    "recommendation": "urgent",
    "confidence_level": "high"
  },
  "metadata": {
    "processing_time_ms": 2340,
    "image_size": [224, 224],
    "gpu_used": true
  }
}
```

---

## Chat with AI

### Get AI Response
```
POST /api/chat/predict

Content-Type: application/json

Request:
{
  "message": "What is the mortality rate for osteosarcoma?",
  "detected_disease": "osteosarcoma",
  "confidence": 0.94,
  "conversation_history": [
    {
      "role": "user",
      "content": "What is osteosarcoma?"
    },
    {
      "role": "assistant",
      "content": "Osteosarcoma is the most common primary malignant bone tumor..."
    }
  ]
}

Response 200:
{
  "response": "The mortality rate for osteosarcoma is approximately 15-20% for patients who receive appropriate treatment...",
  "tokens_used": {
    "input_tokens": 345,
    "output_tokens": 127,
    "total_tokens": 472
  },
  "model": "mixtral-8x7b-32768",
  "processing_time_ms": 1230
}

Response 400:
{
  "detail": "Missing required field: message"
}

Response 500:
{
  "detail": "Chat service unavailable. Check GROQ_API_KEY is set."
}
```

**Model Used**: Mixtral 8x7B (via Groq)  
**Latency**: 0.5-2.0s depending on response length  
**Context Awareness**: Includes detected disease, confidence, and conversation history

### System Prompt (Internal)

The chat endpoint uses a specialized medical system prompt:

```
You are a knowledgeable medical AI assistant helping healthcare professionals 
understand bone tumor classification results. You provide evidence-based information 
about the detected diagnosis and answer follow-up questions.

IMPORTANT:
- You are NOT a replacement for professional medical diagnosis
- Always emphasize the importance of consulting qualified healthcare professionals
- Provide context-aware information based on the detected disease: {disease}
- Include the confidence level of the prediction: {confidence}%
- Be accurate, evidence-based, and conservative in claims

Conversation History provides context for follow-up questions.
```

---

## File Upload

### Upload Image
```
POST /api/upload

Content-Type: multipart/form-data

Request:
- file: <image file>

Response 200:
{
  "filename": "scan_001.jpg",
  "size_bytes": 245632,
  "content_type": "image/jpeg",
  "upload_time": "2026-04-12T14:30:00Z",
  "path": "/uploads/scan_001.jpg"
}

Response 413:
{
  "detail": "File too large (max 16MB)"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request format"
}
```

### 415 Unsupported Media Type
```json
{
  "detail": "Only JPEG, PNG, and WebP images supported"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Model inference failed",
  "error_id": "INFERENCE_ERROR_001"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service temporarily unavailable. Try again in 60s."
}
```

---

## Request/Response Formats

### Supported Image Formats

| Format | Extension | Max Size |
|--------|-----------|----------|
| JPEG | .jpg, .jpeg | 16 MB |
| PNG | .png | 16 MB |
| WebP | .webp | 16 MB |
| BMP | .bmp | 16 MB |

### Image Size Requirements

- **Minimum**: 224 × 224 pixels
- **Recommended**: 512 × 512 or larger
- **Auto-resized to**: 224 × 224 for MobileNet
- **Auto-resized to**: 384 × 384 for full analysis

### Content-Type Headers

```
Images: multipart/form-data
JSON: application/json
```

---

## Authentication

*Currently: None (internal network only)*

### Future: API Key Auth
```
Authorization: Bearer YOUR_API_KEY
```

---

## Rate Limiting

| Endpoint | Limit | Window |
|----------|-------|--------|
| /api/mobilenet/predict | 100 | 1 minute |
| /api/inference | 30 | 1 minute |
| /api/chat/predict | 20 | 1 minute |

**Note**: Limits are per IP address. Not enforced on localhost.

---

## Usage Examples

### Example 1: Quick Classification with cURL

```bash
curl -X POST http://localhost:8000/api/mobilenet/predict \
  -H "Accept: application/json" \
  -F "file=@./scan.jpg"
```

### Example 2: Full Analysis with Python

```python
import requests

with open('scan.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/inference',
        files={'file': f},
        params={'include_segmentation': True}
    )
    
result = response.json()
print(f"Diagnosis: {result['classification']['class_name']}")
print(f"Confidence: {result['classification']['confidence']:.2%}")
```

### Example 3: Chat Interaction with JavaScript

```javascript
async function askAI(message, disease, confidence) {
  const response = await fetch('/api/chat/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      detected_disease: disease,
      confidence: confidence,
      conversation_history: []
    })
  });
  
  return await response.json();
}

// Usage
const result = await askAI(
  "What is the treatment protocol?",
  "osteosarcoma",
  0.94
);
console.log(result.response);
```

---

## Response Time Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Health check | < 10ms | 50ms |
| Quick classification | < 100ms | 500ms |
| Full analysis | 1-3s | 10s |
| Chat response | 1-2s | 15s |

---

## Common Integration Patterns

### Pattern 1: Sequential Analysis
```
1. POST /api/mobilenet/predict (quick classification)
2. Check confidence
3. If confident enough, show result
4. Else, POST /api/inference (full analysis)
5. Display both results
```

### Pattern 2: Chat-Augmented Diagnosis
```
1. POST /api/mobilenet/predict
2. Store classification result
3. User clicks "Chat with AI"
4. POST /api/chat/predict with disease + confidence
5. Maintain conversation_history for follow-ups
```

### Pattern 3: Batch Processing
```
1. Load multiple images
2. For each image, POST /api/mobilenet/predict
3. Collect results
4. Generate report
5. Export results
```

---

## Environment Variables

### Required
```
GROQ_API_KEY=your_key_from_groq_console
```

### Optional
```
MODEL_PATH=/path/to/models  # Defaults to ./outputs
DEBUG=true                   # Enable debug logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
UPLOAD_LIMIT=16777216       # Max upload size in bytes (16MB default)
```

---

## Deployment Notes

### Docker Deployment
```bash
docker build -t vistai-backend .
docker run -p 8000:8000 \
  -e GROQ_API_KEY="your_key" \
  vistai-backend
```

### Kubernetes Deployment
```yaml
env:
  - name: GROQ_API_KEY
    valueFrom:
      secretKeyRef:
        name: vistai-secrets
        key: groq-api-key
```

### Production Checklist
- [ ] CORS configured properly
- [ ] Rate limiting enabled
- [ ] Error logging enabled
- [ ] Health checks configured
- [ ] API key/auth enabled
- [ ] Models cached appropriately
- [ ] Timeout settings reviewed
- [ ] Disk space monitored

---

## Support & Debugging

### Check Backend Status
```bash
curl http://localhost:8000/health
```

### View API Documentation
```
http://localhost:8000/docs
```

### Enable Debug Logging
```bash
DEBUG=true python -m uvicorn app.main:app
```

### Test All Endpoints
```bash
python test_api.py  # Includes all endpoint tests
```

---

**VistAI API v2.0**  
**Last Updated**: April 12, 2026  
**Status**: Production Ready
