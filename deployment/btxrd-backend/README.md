# BTXRD Backend – Bone Tumor X-Ray Detection API

FastAPI backend for bone tumor classification, segmentation, Grad-CAM visualization, LLM chat, and report generation.

## Quick Start

```bash
# 1. Create virtual environment (or use existing project venv)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env → set BTXRD_PROJECT_ROOT and LOCAL_LLM_MODEL_PATH

# 4. Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload X-ray image |
| POST | `/api/inference` | Run classification + segmentation |
| POST | `/api/chat` | Chat about results |
| POST | `/api/chat/stream` | Streaming chat (SSE) |
| POST | `/api/report` | Generate radiology report |
| GET | `/health` | Health check |

## Architecture

```
btxrd-backend/
├── app/
│   ├── main.py                  # FastAPI app + model lifecycle
│   ├── config.py                # Settings from .env
│   ├── api/
│   │   ├── upload.py            # Image upload endpoint
│   │   ├── inference.py         # Classification + segmentation
│   │   ├── chat.py              # LLM chat + streaming
│   │   └── report.py            # Report generation
│   ├── services/
│   │   ├── classification_service.py   # ConvNeXt-Tiny wrapper
│   │   ├── segmentation_service.py     # SegFormer-B2 wrapper
│   │   ├── llm_service.py             # OpenAI / TinyLlama
│   │   └── visualization_service.py    # Overlay + heatmap generation
│   ├── schemas/                 # Pydantic request/response models
│   └── utils/
│       └── file_manager.py      # Upload storage
├── requirements.txt
└── .env
```

## LLM Configuration

The backend uses a local GGUF model through `llama-cpp-python`.

Required `.env` keys:

- `LOCAL_LLM_MODEL_PATH` (example: `C:/Users/Nauman/models/gemma-2-2b-it-Q4_K_M.gguf`)
- `LOCAL_LLM_N_CTX`
- `LOCAL_LLM_N_THREADS`
- `LOCAL_LLM_N_GPU_LAYERS`

## Models

The backend loads pre-trained KD student models from `BTXRD/combined_inference/models/`:
- **Classification**: ConvNeXt-Tiny (28M params) → 73.8% accuracy
- **Segmentation**: SegFormer-B2 (28M params) → 51% Dice score
