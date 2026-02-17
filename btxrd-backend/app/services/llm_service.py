"""
LLM Service – Groq API powered chat and report generation.

Uses Groq's Llama models exclusively for all LLM functionality.
No mock responses, no local models – real API calls only.
"""

from __future__ import annotations
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI radiology assistant designed for use by radiologists and medical professionals reviewing bone tumor X-ray analyses.

COMMUNICATION STYLE:
- Professional, concise, and direct — no unnecessary explanations.
- Assume the user is a radiologist or physician who understands medical terminology.
- Answer ONLY what is asked. Do not volunteer extra information.
- Use clinical language: "consistent with", "suggestive of", "raises concern for".

CAPABILITIES:
1. RESULT QUERIES: When asked about this specific analysis, respond using ONLY the provided AI model outputs (classification, confidence, segmentation coverage, differentials).
2. CLINICAL KNOWLEDGE: When asked general questions about tumor types, imaging characteristics, differentials, or clinical behavior — draw on medical knowledge to provide accurate, concise answers.

RULES:
- Never make definitive diagnoses — use "AI classification suggests", "model predicts".
- Keep responses brief: 1-3 sentences for simple queries, up to 5 for complex ones.
- For treatment questions: "Management decisions require clinical-radiological-pathological correlation."
- Do not add disclaimers unless specifically about model limitations.

Examples of good responses:
- "Model predicts osteosarcoma (74.2% confidence). Malignant. Recommend MRI for staging."
- "Osteosarcoma typically presents with aggressive periosteal reaction and Codman triangles."
- "Confidence is 44.6% — consider chondrosarcoma and fibrosarcoma as differentials.\""""

REPORT_SYSTEM_PROMPT = """You are a professional radiology report writer for an AI-assisted bone tumor analysis system (VistAI). You write CONCISE, conservative, clinically realistic, structured reports based ONLY on the provided AI model output data.

CRITICAL: This report MUST fit on ONE PAGE. Be extremely concise and direct.

You MUST strictly follow this exact section structure and rules:

---

## PATIENT INFORMATION
- If patient name is provided, include: "Patient Name: [name]"
- If patient age is provided, include: "Age: [age] years"
- If neither is provided, state: "Patient demographics not provided."

## EXAMINATION
- Modality: Conventional Radiography (X-ray)
- Region: [Infer from context if available, otherwise state "Musculoskeletal region"]
- View: AP/Lateral (assumed)

## CLINICAL INDICATION
- If clinical indication is provided, state it in ONE sentence.
- If not provided: "AI-assisted bone tumor screening."

## FINDINGS
Keep to 2-3 sentences maximum:
- Describe lesion location and extent using tumor_coverage percentage
- Use terms: "ill-defined margins", "well-circumscribed", "suspicious for"
- Do NOT invent measurements or anatomy not in the data

## IMPRESSION
Keep to 2-3 sentences maximum:
- State top predicted class with confidence-aware language:
  - confidence >= 0.8: "features consistent with [class]"
  - confidence 0.5–0.79: "suspicious for [class], differentials remain"  
  - confidence < 0.5: "indeterminate; [class] suggested with low confidence"
- Mention malignancy status (malignant/benign)
- List top 2 differentials only

## RECOMMENDATIONS
List 3-4 key recommendations only:
1. Correlation with clinical history
2. Advanced imaging (MRI/CT) for characterization
3. Orthopedic oncology referral if malignant
4. Biopsy for histopathological confirmation

## AI MODEL OUTPUT SUMMARY
Present as a compact table:
- Predicted Class: [top_class]
- Confidence: [confidence]%
- Malignancy: [malignant/benign]
- Coverage: [tumor_coverage]%
- Top-3 Differential: [list 3 only with percentages]

## LIMITATIONS
Condense to 2-3 sentences:
- AI system trained on BTXRD dataset with ConvNeXt-Tiny and SegFormer-B2 models.
- Single radiographic view; findings may differ with additional modalities.
- Segmentation approximate; not for surgical planning.

## DISCLAIMER
⚠️ IMPORTANT: AI-generated report for research/educational purposes ONLY. NOT a medical diagnosis. Must be verified by a board-certified radiologist. Never use as sole basis for clinical decisions.

---

RULES:
- Use formal, third-person radiology language.
- Be EXTREMELY CONCISE — every word counts.
- Every claim must be traceable to the provided data.
- Keep total report to 250-350 words maximum.
- Use short sentences and bullet points where possible.
"""


# ── Context builders ───────────────────────────────────────────────────────

def _build_context_message(analysis: dict) -> str:
    """Convert inference results into a text context for the LLM."""
    cls = analysis.get("classification") or {}
    seg = analysis.get("segmentation") or {}

    parts = [
        "=== AI ANALYSIS RESULTS ===",
        f"Primary Prediction: {cls.get('top_class', 'N/A')}",
        f"Confidence: {cls.get('confidence', 0):.1%}",
        f"Malignancy Status: {cls.get('malignancy', 'N/A')}",
    ]

    top5 = cls.get("top5", [])
    if top5:
        parts.append("Top-5 Predictions:")
        for item in top5:
            parts.append(f"  - {item['class']}: {item['probability']:.1%}")

    if seg:
        parts.append(f"Tumor Coverage: {seg.get('tumor_coverage', 0):.1f}% of image area")

    return "\n".join(parts)


def _build_report_context(analysis: dict, patient_info: dict | None = None) -> str:
    """Build a richer context specifically for report generation."""
    parts = []
    
    # Patient demographics section
    if patient_info:
        parts.append("=== PATIENT INFORMATION ===")
        if patient_info.get("patient_name"):
            parts.append(f"Patient Name: {patient_info['patient_name']}")
        if patient_info.get("patient_age"):
            parts.append(f"Patient Age: {patient_info['patient_age']} years")
        if patient_info.get("clinical_indication"):
            parts.append(f"Clinical Indication: {patient_info['clinical_indication']}")
        parts.append("")
    
    cls = analysis.get("classification") or {}
    seg = analysis.get("segmentation") or {}

    parts.extend([
        "=== CLASSIFICATION OUTPUT ===",
        f"Predicted Class: {cls.get('top_class', 'N/A')}",
        f"Confidence Score: {cls.get('confidence', 0):.4f}",
        f"Malignancy Status: {cls.get('malignancy', 'N/A')}",
    ])

    top5 = cls.get("top5", [])
    if top5:
        parts.append("\nTop-5 Differential Predictions:")
        for i, item in enumerate(top5, 1):
            parts.append(f"  {i}. {item['class']}: {item['probability']:.4f} ({item['probability']:.1%})")

    parts.append("\n=== SEGMENTATION OUTPUT ===")
    if seg:
        parts.append(f"Tumor Coverage: {seg.get('tumor_coverage', 0):.1f}% of imaged area")
        parts.append("Mask Available: Yes")
        parts.append("Overlay Available: Yes")
    else:
        parts.append("Segmentation: Not performed")

    parts.append("\n=== IMAGING ===")
    parts.append("Modality: Conventional Radiography (X-ray)")
    parts.append("Analysis System: VistAI v1.0")
    parts.append("Classification Model: ConvNeXt-Tiny (Knowledge Distillation Student)")
    parts.append("Segmentation Model: SegFormer-B2 (Knowledge Distillation Student)")

    return "\n".join(parts)


# ── LLM Service ────────────────────────────────────────────────────────────

class LLMService:
    """Groq-powered LLM service. All responses come from the API."""

    def __init__(self):
        self.settings = get_settings()
        self._client: AsyncOpenAI | None = None
        self._init_client()

    def _init_client(self):
        """Initialize the Groq client."""
        api_key = self.settings.groq_api_key
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set in .env — LLM features require a valid Groq API key. "
                "Get one at https://console.groq.com"
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info("✅ LLM backend: Groq (%s)", self.settings.groq_model)

    # ── Chat ───────────────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        analysis: dict,
        history: list[dict] | None = None,
    ) -> str:
        """Single-turn or multi-turn chat grounded in analysis results."""
        context = _build_context_message(analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self,
        user_message: str,
        analysis: dict,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat – yields tokens one at a time."""
        context = _build_context_message(analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        stream = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Report ─────────────────────────────────────────────────────────────

    async def generate_report(self, analysis: dict, patient_info: dict | None = None) -> str:
        """Generate a structured radiology-style report via API."""
        context = _build_report_context(analysis, patient_info)

        resp = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate a CONCISE one-page radiology report strictly following the template structure. "
                        "Keep it brief and professional. "
                        "Use ONLY the following AI model outputs as your data source:\n\n"
                        f"{context}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return resp.choices[0].message.content or ""
