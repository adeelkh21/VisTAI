"""LLM service using a local GGUF model via llama-cpp-python."""

from __future__ import annotations
import asyncio
import logging
from typing import AsyncGenerator

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical AI radiology assistant for bone tumor X-ray analysis, conversing with radiologists and clinicians.

CONVERSATION CONTEXT:
You will receive messages in a back-and-forth conversation. Use the conversation history to understand what has been discussed:
- Previous user questions establish what findings the radiologist is focused on
- Previous AI responses provide context for follow-up clinical questions
- Follow-up questions like "what is the next step?" refer to the findings discussed in prior messages
- Always acknowledge and build on previous messages rather than treating each question in isolation

CASE ARCHITECTURE:
You have been provided with:
1. AI classification results (tumor class, confidence %, malignancy status, differential diagnoses)
2. AI segmentation results (tumor coverage percentage)
3. Conversation history (for continuity across multiple questions)
COMMUNICATION STYLE:
- Professional but natural. Sound like an experienced clinician colleague discussing findings, not a template or lookup table.
- Be THOROUGH and SUBSTANTIVE: provide complete, well-reasoned answers (typically 3-5 sentences).
- Address multiple questions coherently and show how answers interconnect.
- Assume the user is medically trained (radiologist or clinician).
- Use clinical language: "consistent with", "suggestive of", "raises concern for", "indicates", "warrants", "management", "staging".

QUESTION TYPES & RESPONSES:

1. IMAGE-SPECIFIC QUERIES ("What does the model predict?", "What tumor type is this?")
   - Lead with the classification: AI suggests [class] with [confidence]% confidence and [benign/malignant] status
   - Explain clinical significance and why this matters (staging, urgency, prognosis)
   - Mention top differentials and why they're in the differential
   - If relevant: discuss imaging features and management implications
   - Be specific and detailed—do not give one-sentence answers

2. FOLLOW-UP/CLINICAL CONTEXT ("What does malignant mean for this case?", "What are next steps?", "Why these differentials?")
   - Draw on radiology knowledge to explain: why this diagnosis matters, what it means for management, why other diagnoses are considered
   - Ground everything in the specific analysis provided (confidence, coverage, malignancy status)
   - Show how clinical findings support the differential reasoning
   - Address management and staging as appropriate

3. MULTI-PART QUESTIONS
   - Answer each part fully, showing how answers relate to each other
   - Do not abbreviate or give fragmented responses

RULES:
- Never state definitive diagnoses yourself. Use: "AI classification suggests", "model output indicates", "findings consistent with".
- Do NOT invent tumor classes, percentages, measurements, or findings not in the analysis.
- If required information is missing: "The analysis does not provide that information."
- Be COMPLETE in your answers. Follow-up questions deserve substantive, thorough responses, not minimal replies.
- Ground severity/management language in malignancy (benign vs malignant) from the analysis.

KEY PRINCIPLE: Your goal is to be a clinical partner helping the radiologist understand and act on the AI findings. Provide thorough, professional explanations that demonstrate understanding of both the data and the clinical implications.

TONE: Professional, conversational, substantive—a colleague explaining clinical reasoning.\""""

GROUNDED_BEHAVIOR_PROMPT = """CONVERSATION AWARENESS & RESPONSE GUIDELINES:

UNDERSTAND CONTEXT FROM HISTORY:
- Read the conversation history to understand what the radiologist has already asked about
- Follow-up questions like "what is the next step?" refer to the finding(s) discussed in prior messages
- Build your answer ON TOP of previous discussion, showing continuity and awareness of the case
- Do NOT treat each message in isolation—this is an ongoing conversation

RESPONSE REQUIREMENTS:
- Do NOT repeat or paraphrase the user's question.
- Do NOT output role labels like 'User:' or 'Assistant:'.
- PROVIDE COMPLETE ANSWERS: For case-specific and follow-up questions, give THOROUGH, substantive responses (3-5 sentences typically).
- For case-specific answers, use ONLY the provided analysis context (class, confidence, malignancy, coverage, differentials).
- For follow-up/management questions: integrate the findings from prior discussion to inform your answer
- If required data is missing: "The analysis does not provide this information."
- Never invent tumor classes, percentages, measurements, or imaging findings.
- Be clinically phrased, direct, and relevant. Address what the user actually asked.
- ALWAYS provide a substantive answer. An empty response is a failure.
"""

STOP_SEQUENCES = ["\nUser:", "\nAssistant:", "\n[Instructions]", "\n### User", "<|user|>"]

REQUIRED_REPORT_SECTIONS = [
    "## PATIENT INFORMATION",
    "## EXAMINATION",
    "## CLINICAL INDICATION",
    "## FINDINGS",
    "## IMPRESSION",
    "## RECOMMENDATIONS",
    "## AI MODEL OUTPUT SUMMARY",
    "## LIMITATIONS",
    "## DISCLAIMER",
]

REPORT_REVIEW_PROMPT = """You are a strict clinical editor.
Task: revise the drafted radiology report so every case-specific claim is grounded in the provided context.
Rules:
- Keep the same overall structure and maintain clinical readability.
- Remove or rewrite any unsupported claims.
- Do not add new findings not present in context.
- Preserve useful detail and clarity.
- Output only the final revised report.
"""

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
- Every case-specific claim must be traceable to the provided data.
- Do not invent anatomy, dimensions, staging, or imaging signs not present in provided context.
- Keep report comprehensive but focused (approximately 300-550 words unless data is sparse).
- Use short paragraphs and bullets for readability.
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
    """Local LLM service backed by llama-cpp."""

    def __init__(self):
        self.settings = get_settings()
        self._llm = None
        self._init_model()

    def _init_model(self):
        """Initialize local Gemma GGUF model once."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
            ) from exc

        model_path = self.settings.llm_model_path
        if not model_path:
            raise RuntimeError("LLM_MODEL_PATH is not set in .env")

        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.settings.llm_ctx_size,
            n_threads=self.settings.llm_threads,
            n_gpu_layers=0,  # CPU-only on Jetson (GPU offload via CUDA not used)
            n_batch=128,
            verbose=False,
        )
        logger.info("✅ LLM backend: local Gemma (%s)", model_path)

    def _chat_completion(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """Run a local chat completion synchronously (wrapped by async callers)."""
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.12,
                stop=STOP_SEQUENCES,
            )
            text = (response["choices"][0]["message"]["content"] or "").strip()
            return self._cleanup_text(text)
        except Exception as exc:
            logger.warning("Chat template fallback triggered: %s", exc)
            prompt = self._messages_to_prompt(messages)
            response = self._llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.12,
                stop=STOP_SEQUENCES,
                echo=False,
            )
            text = (response["choices"][0]["text"] or "").strip()
            return self._cleanup_text(text)

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat-style messages to a plain completion prompt."""
        lines: list[str] = [GROUNDED_BEHAVIOR_PROMPT]
        for msg in messages:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                lines.append(f"[Instructions]\n{content}\n")
            elif role == "assistant":
                lines.append(f"Assistant: {content}\n")
            else:
                lines.append(f"User: {content}\n")
        lines.append("Assistant: Provide only the final clinical answer.")
        return "\n".join(lines)

    def _cleanup_text(self, text: str) -> str:
        """Normalize model output and remove chat artifacts."""
        cleaned = (text or "").strip()
        for prefix in ("Assistant:", "assistant:", "Answer:", "Final answer:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        for marker in ("\nUser:", "\nAssistant:", "\n[Instructions]", "\n### User", "<|user|>"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()

        # Improve readability for class labels like multiple_osteochondromas.
        cleaned = cleaned.replace("_", " ")

        return cleaned

    def _is_analysis_question(self, user_message: str) -> bool:
        """Detect if question is about the case analysis or clinical follow-up."""
        q = user_message.lower()
        keywords = [
            "predict", "prediction", "class", "confidence", "malignant", "benign",
            "tumor coverage", "coverage", "differential", "top", "result", "analysis",
            "tumor", "danger", "dangerous", "type",
            # Clinical follow-up keywords
            "next step", "management", "treatment", "follow-up", "staging", "care",
            "how should", "what should", "what does", "what does this", "what does the",
            "why", "how", "concern", "risk", "implication", "import", "prognosis",
        ]
        return any(k in q for k in keywords)

    def _pretty_label(self, value: str) -> str:
        return (value or "N/A").replace("_", " ").strip()

    def _grounded_summary(self, analysis: dict) -> str:
        cls = analysis.get("classification") or {}
        seg = analysis.get("segmentation") or {}

        top_class = self._pretty_label(cls.get("top_class", "N/A"))
        confidence = float(cls.get("confidence", 0.0)) * 100.0
        malignancy = cls.get("malignancy", "N/A")
        coverage = seg.get("tumor_coverage", None)

        top5 = cls.get("top5", [])
        diffs = [self._pretty_label(item.get("class", "")) for item in top5[1:3] if item.get("class")]
        diffs_text = ", ".join(diffs) if diffs else "not available"

        if coverage is None:
            coverage_text = "Tumor coverage is not available"
        else:
            coverage_text = f"Tumor coverage is {float(coverage):.1f}%"

        return (
            f"AI classification suggests {top_class} ({confidence:.1f}% confidence), {malignancy}. "
            f"{coverage_text}. Top differentials: {diffs_text}."
        )

    def _generate_management_guidance(self, analysis: dict, user_question: str) -> str:
        """Generate context-aware clinical management guidance based on analysis."""
        cls = analysis.get("classification") or {}
        seg = analysis.get("segmentation") or {}

        malignancy = str(cls.get("malignancy", "")).lower()
        confidence = float(cls.get("confidence", 0.0))
        coverage = seg.get("tumor_coverage", 0.0)

        # Build management response based on malignancy status
        if malignancy == "malignant":
            return (
                "Given the malignant classification, the following steps are recommended:\n"
                "1. **Urgent staging**: MRI to assess local extension and CT chest to evaluate for pulmonary metastases\n"
                "2. **Specialist consultation**: Orthopedic oncology referral for treatment planning\n"
                "3. **Biopsy if needed**: Tissue diagnosis may be required for definitive histologic confirmation\n"
                "4. **Multidisciplinary review**: Coordination between radiology, orthopedic surgery, and medical oncology\n"
                "The aggressive nature of malignant lesions warrants prompt clinical action."
            )
        elif malignancy == "benign":
            return (
                "Given the benign classification, recommended management includes:\n"
                "1. **Clinical follow-up**: Monitor for symptoms (pain, progression, functional impairment)\n"
                "2. **Imaging follow-up**: Periodic X-rays may be indicated depending on lesion type and patient age\n"
                "3. **Surgical consideration**: Surgery may be warranted if the lesion is symptomatic or causing functional compromise\n"
                "4. **Reassurance**: Benign lesions generally have excellent prognosis\n"
                f"With {coverage:.1f}% coverage, the lesion appears {'localized' if coverage < 25 else 'moderately extensive'}, which should be considered in surgical planning if needed."
            )
        else:
            return (
                "Based on the AI findings, clinical correlation and expert radiologist review are essential.\n"
                "Recommended next steps:\n"
                "1. Review imaging findings with an experienced musculoskeletal radiologist\n"
                "2. Correlate with clinical presentation and patient history\n"
                "3. Consider advanced imaging (MRI/CT) if not already obtained\n"
                "4. Multidisciplinary consultation with orthopedic surgery if malignancy cannot be excluded"
            )

    def _enforce_grounding(self, user_message: str, analysis: dict, reply: str) -> str:
        """Apply MINIMAL fallback grounding only when response is genuinely broken (empty or echoing)."""
        safe = (reply or "").strip()
        is_analysis = self._is_analysis_question(user_message)
        
        # ONLY fallback if response is completely empty
        if not safe:
            if is_analysis:
                # Check if it's a clinical follow-up question (next step, management, etc.)
                if any(k in user_message.lower() for k in ["next step", "management", "treatment", "follow-up", "staging", "care", "how should", "what should"]):
                    return self._generate_management_guidance(analysis, user_message)
                return self._grounded_summary(analysis)
            return "I was unable to generate a response. Could you rephrase your question?"
        
        # ONLY fallback if response is clearly echoing the user's input verbatim (likely a model failure)
        if safe.lower().startswith(user_message.lower()) and len(safe) < len(user_message) + 50:
            if is_analysis:
                return self._grounded_summary(analysis)
            return "I don't have a good response to that. Please try rephrasing."
        
        # Otherwise trust the model's response - it may have good clinical reasoning even if it doesn't
        # repeat the tumor class name in every answer
        return safe

    def _has_required_report_sections(self, text: str) -> bool:
        report = (text or "")
        return all(section in report for section in REQUIRED_REPORT_SECTIONS)

    def _build_structured_report_fallback(
        self,
        analysis: dict,
        patient_info: dict | None,
        drafted_findings: str,
    ) -> str:
        """Deterministic, grounded report fallback when model misses required structure."""
        cls = analysis.get("classification") or {}
        seg = analysis.get("segmentation") or {}

        top_class = self._pretty_label(cls.get("top_class", "N/A"))
        confidence = float(cls.get("confidence", 0.0)) * 100.0
        malignancy = str(cls.get("malignancy", "N/A"))
        coverage = seg.get("tumor_coverage", None)

        top5 = cls.get("top5", [])
        top3 = [
            f"{self._pretty_label(item.get('class', 'N/A'))} ({float(item.get('probability', 0.0))*100.0:.1f}%)"
            for item in top5[:3]
        ]
        top3_text = ", ".join(top3) if top3 else "Not available"

        if coverage is None:
            coverage_text = "Tumor coverage not available"
        else:
            coverage_text = f"Estimated tumor coverage: {float(coverage):.1f}% of image area"

        patient_lines: list[str] = []
        if patient_info and patient_info.get("patient_name"):
            patient_lines.append(f"- Patient Name: {patient_info['patient_name']}")
        if patient_info and patient_info.get("patient_age") is not None:
            patient_lines.append(f"- Age: {patient_info['patient_age']} years")
        if not patient_lines:
            patient_lines.append("- Patient demographics not provided.")

        indication = "AI-assisted bone tumor screening."
        if patient_info and patient_info.get("clinical_indication"):
            indication = str(patient_info["clinical_indication"])

        findings = drafted_findings.strip() if drafted_findings else ""
        # If model returned fragmented report sections, do not embed whole blocks in FINDINGS.
        if "## " in findings or "---" in findings:
            findings = ""
        if not findings:
            findings = (
                f"Model output indicates a lesion pattern most consistent with {top_class}. "
                f"{coverage_text}. "
                f"Differential considerations from model ranking include {top3_text}."
            )

        if confidence >= 80:
            impression_prefix = "Features are consistent with"
        elif confidence >= 50:
            impression_prefix = "Findings are suspicious for"
        else:
            impression_prefix = "Findings are indeterminate; model suggests"

        recommendations = [
            "1. Correlate with clinical history and examination.",
            "2. Consider MRI/CT for lesion characterization and local extent.",
            "3. Refer to orthopedic oncology if malignant risk is present.",
            "4. Histopathological confirmation (biopsy) when clinically indicated.",
        ]

        return "\n".join([
            "## PATIENT INFORMATION",
            *patient_lines,
            "",
            "## EXAMINATION",
            "- Modality: Conventional Radiography (X-ray)",
            "- Region: Musculoskeletal region",
            "- View: AP/Lateral (assumed)",
            "",
            "## CLINICAL INDICATION",
            f"- {indication}",
            "",
            "## FINDINGS",
            findings,
            "",
            "## IMPRESSION",
            (
                f"{impression_prefix} {top_class} ({confidence:.1f}% confidence). "
                f"Malignancy status from model: {malignancy}."
            ),
            "",
            "## RECOMMENDATIONS",
            *recommendations,
            "",
            "## AI MODEL OUTPUT SUMMARY",
            f"- Predicted Class: {top_class}",
            f"- Confidence: {confidence:.1f}%",
            f"- Malignancy: {malignancy}",
            f"- Coverage: {coverage_text.replace('Estimated ', '')}",
            f"- Top-3 Differential: {top3_text}",
            "",
            "## LIMITATIONS",
            "- AI system trained on BTXRD dataset with ConvNeXt-Tiny and SegFormer-B2 student models.",
            "- Single radiographic view may not capture full lesion characteristics.",
            "- Segmentation is approximate and not intended for surgical planning.",
            "",
            "## DISCLAIMER",
            "AI-generated report for research/educational use only. Must be reviewed by a qualified radiologist; not a standalone clinical diagnosis.",
        ])

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
            {"role": "system", "content": GROUNDED_BEHAVIOR_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        raw = await asyncio.to_thread(
            self._chat_completion,
            messages,
            self.settings.llm_max_tokens,
            0.15,
        )
        return self._enforce_grounding(user_message, analysis, raw)

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
            {"role": "system", "content": GROUNDED_BEHAVIOR_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        full_text = await asyncio.to_thread(
            self._chat_completion,
            messages,
            self.settings.llm_max_tokens,
            0.15,
        )
        full_text = self._enforce_grounding(user_message, analysis, full_text)

        # Pseudo-stream response in chunks so existing SSE frontend keeps working.
        for token in full_text.split():
            yield token + " "
            await asyncio.sleep(0)

    # ── Report ─────────────────────────────────────────────────────────────

    async def generate_report(self, analysis: dict, patient_info: dict | None = None) -> str:
        """Generate a structured radiology-style report via local model."""
        context = _build_report_context(analysis, patient_info)

        messages = [
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
        ]
        draft = await asyncio.to_thread(
            self._chat_completion,
            messages,
            self.settings.llm_max_tokens,
            0.2,
        )

        review_messages = [
            {"role": "system", "content": REPORT_REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    "Context (ground truth):\n"
                    f"{context}\n\n"
                    "Draft report to review:\n"
                    f"{draft}\n\n"
                    "Return a corrected report that is fully grounded in the context."
                ),
            },
        ]
        revised = await asyncio.to_thread(
            self._chat_completion,
            review_messages,
            self.settings.llm_max_tokens,
            0.1,
        )

        candidate = revised if revised else draft
        if self._has_required_report_sections(candidate):
            return candidate

        return self._build_structured_report_fallback(analysis, patient_info, candidate)
