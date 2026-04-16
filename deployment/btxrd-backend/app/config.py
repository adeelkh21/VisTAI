"""
BTXRD Backend – Application Configuration
All paths are configurable via environment variables for deployment flexibility.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Model Directory ────────────────────────────────────────────────────
    # Base directory for all model files (.pth, .onnx, .trt, .gguf)
    model_dir: str = os.getenv("MODEL_DIR", "BTXRD/combined_inference/models")

    # ── Model Files (relative to MODEL_DIR) ───────────────────────────────
    classify_model_file: str = os.getenv("CLASSIFY_MODEL_FILE", "classification_student.pth")
    segment_model_file: str = os.getenv("SEGMENT_MODEL_FILE", "segmentation_student.pth")

    # ── LLM Settings ──────────────────────────────────────────────────────
    llama_cli_path: str = os.getenv("LLAMA_CLI_PATH", "/app/llama.cpp/llama-cli")
    llm_model_path: str = os.getenv(
        "LLM_MODEL_PATH",
        str(Path(os.getenv("MODEL_DIR", "BTXRD/combined_inference/models")) / "gemma-2-2b-it-Q4_K_M.gguf")
    )
    llm_threads: int = int(os.getenv("LLM_THREADS", "8"))
    llm_ctx_size: int = int(os.getenv("LLM_CTX_SIZE", "2048"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "400"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ── App Settings ──────────────────────────────────────────────────────
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))
    report_dir: Path = Path(os.getenv("REPORT_DIR", "reports"))
    port: int = int(os.getenv("PORT", "8000"))

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_upload_size_mb: int = 20

    # ── Inference ─────────────────────────────────────────────────────────
    cls_image_size: int = 384
    seg_image_size: int = 224
    seg_threshold: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cls_checkpoint(self) -> str:
        """Path to classification model checkpoint."""
        return str(Path(self.model_dir) / self.classify_model_file)

    @property
    def seg_checkpoint(self) -> str:
        """Path to segmentation model checkpoint."""
        return str(Path(self.model_dir) / self.segment_model_file)

    @property
    def resolved_upload_dir(self) -> str:
        """Resolve upload directory - uses env var or falls back to default."""
        if self.upload_dir and str(self.upload_dir):
            return str(self.upload_dir)
        return str(Path(__file__).resolve().parents[2] / "uploads")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
