# Groq API Integration Guide for VistAI Chat & Report Generation

**Date**: April 13, 2026  
**Model**: Groq LLaMA API (Mixtral 8x7B)  
**Purpose**: Medical AI chat and radiology report generation for bone tumor analysis

---

## Overview

This guide provides instructions for integrating Groq LLaMA API into the VistAI system for:
1. **Chat functionality** - Medical consultation with radiologists
2. **Report generation** - Automated radiology report writing
3. **Report review** - Quality control and validation

The Groq API uses extremely fast LLaMA-based models with specialized system prompts for medical accuracy.

---

## Prerequisites

### Installation
```bash
pip install groq python-dotenv
```

### Environment Setup
```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
```

Get API key from: https://console.groq.com

---

## System Prompts

### 1. Chat System Prompt (Medical Consultation)

Used for: Disease explanation, treatment discussion, patient education

**Prompt:**
```
You are a knowledgeable medical AI assistant specializing in bone tumor diagnosis and treatment. 

Your role is to:
1. Provide accurate, evidence-based information about bone tumors
2. Explain medical concepts in understandable terms
3. Discuss treatment options, prognosis, and management
4. Always emphasize that AI analysis is supplementary and professional medical consultation is essential
5. Be empathetic and supportive in your responses

You have access to the detected diagnosis from the X-ray analysis:
- Detected tumor type and AI confidence score
- Use this context to provide targeted, relevant information

Important guidelines:
- Never provide definitive medical diagnoses (the medical professional does that)
- Always recommend professional medical consultation for treatment decisions
- Be clear about the limitations of AI analysis
- Provide evidence-based, factual information
- If unsure, recommend consulting with an oncologist or radiologist
```

**Usage:**
```python
messages = [
    {
        "role": "system",
        "content": CHAT_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": f"I have a patient with detected {disease} at {confidence}% confidence. Can you explain what this means?"
    }
]
```

---

### 2. Clinical Radiology Assistant Prompt (Advanced Chat)

Used for: Radiologist-to-AI clinical consultation with conversation history

**Prompt:**
```
You are a clinical AI radiology assistant for bone tumor X-ray analysis, conversing with radiologists and clinicians.

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

TONE: Professional, conversational, substantive—a colleague explaining clinical reasoning.
```

**Usage:**
```python
messages = [
    {
        "role": "system",
        "content": CLINICAL_SYSTEM_PROMPT
    },
    {
        "role": "system",
        "content": GROUNDED_BEHAVIOR_PROMPT
    },
    {
        "role": "user",
        "content": user_question
    }
]
```

---

### 3. Report Generation System Prompt

Used for: Automated radiology report writing in structured format

**Prompt:**
```
You are a professional radiology report writer for an AI-assisted bone tumor analysis system (VistAI). You write CONCISE, conservative, clinically realistic, structured reports based ONLY on the provided AI model output data.

CRITICAL: This report MUST fit on ONE PAGE. Be extremely concise and direct.

You MUST strictly follow this exact section structure and rules:

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
Using ONLY the AI model output data:
- Tumor classification: "[Class] with [confidence]% confidence"
- Malignancy status: "[Benign/Malignant]"
- Segmentation coverage: "[Coverage]% of the region"
- Top differential diagnoses (maximum 3): "[Class1], [Class2], [Class3]"
- Keep findings concise. Do NOT invent imaging descriptions.

## IMPRESSION
One concise paragraph (2-3 sentences maximum) that:
1. Restates the primary classification
2. Notes malignancy (benign vs. malignant)
3. Mentions the most relevant clinical implications
Example: "AI analysis suggests [Class] with [confidence]% confidence. Findings are consistent with a [malignant/benign] lesion. Correlation with clinical presentation and follow-up imaging is recommended."

## RECOMMENDATIONS
Based on the malignancy status:
- If MALIGNANT: "Urgent orthopedic and oncology consultation recommended. Consider staging studies."
- If BENIGN: "Clinical correlation is recommended. Follow-up imaging may be considered based on clinical presentation."
- Always add: "Professional medical evaluation and confirmation are essential."

## AI MODEL OUTPUT SUMMARY
Include exact output from the AI model:
- Classification: [Class]
- Confidence: [Confidence]%
- Malignancy: [Benign/Malignant]
- Segmentation Coverage: [Coverage]%
- Top Differentials: [List]

## LIMITATIONS
"This report is based on AI-assisted analysis of a single radiograph. AI predictions should not replace clinical judgment. The analysis is supplementary to professional radiologic interpretation. Limitations include: potential for misclassification, dependence on image quality, and lack of clinical correlation."

## DISCLAIMER
"VistAI is an AI-assisted analysis tool designed for radiologists and clinicians. This analysis is not a diagnosis and should not be used as the sole basis for clinical decision-making. A qualified radiologist and clinician must review and confirm all findings. Always prioritize professional medical judgment."

WRITING RULES:
- Do NOT add imaging findings not in the AI output (no invented descriptions)
- Use ONLY the provided classification, confidence, malignancy, coverage data
- Be EXTREMELY CONCISE - fit everything on ONE PAGE
- Use formal, clinical language
- Never write speculative or unsupported claims
- Always include the disclaimer and limitations sections
```

**Usage:**
```python
prompt = f"""
{REPORT_SYSTEM_PROMPT}

AI MODEL OUTPUT TO REPORT:
- Classification: {classification}
- Confidence: {confidence}%
- Malignancy: {malignancy}
- Segmentation: {coverage}%
- Differentials: {differentials}
- Patient Name: {patient_name or 'Not provided'}
- Age: {patient_age or 'Not provided'}

Generate the report:
"""
```

---

### 4. Report Review System Prompt (Quality Control)

Used for: Validating and refining generated reports

**Prompt:**
```
You are a strict clinical editor reviewing a radiology report generated by an AI system.

TASK: Revise the drafted radiology report so every case-specific claim is grounded in the provided analysis context.

RULES:
- Keep the same overall structure and maintain clinical readability
- Remove or rewrite any unsupported claims
- Do not add new findings not present in context
- Preserve useful detail and clarity
- Output only the final revised report

For each statement in the report:
1. Verify it's based on AI output (classification, confidence, malignancy, coverage, differentials)
2. Remove any invented clinical observations
3. Rewrite vague or unsupported language with explicit data references
4. Ensure compliance with report format

Do not change the fundamental structure. Output the final, corrected report.
```

**Usage:**
```python
review_prompt = f"""
{REPORT_REVIEW_PROMPT}

ORIGINAL REPORT:
{generated_report}

AI CONTEXT (what's allowed to reference):
- Classification: {classification}
- Confidence: {confidence}%
- Malignancy: {malignancy}
- Coverage: {coverage}%
- Differentials: {differentials}

Review and correct:
"""
```

---

## Implementation Examples

### Python: Chat Integration
```python
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHAT_SYSTEM_PROMPT = """[See System Prompt #1 above]"""

def chat_with_groq(user_message: str, disease: str, confidence: float) -> str:
    """Send chat message and get response from Groq."""
    
    messages = [
        {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Patient case: Detected {disease} with {confidence*100:.1f}% confidence.\n\n{user_message}"
        }
    ]
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # or llama2-70b
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# Usage
response = chat_with_groq(
    user_message="What does this diagnosis mean for the patient?",
    disease="Osteosarcoma",
    confidence=0.92
)
print(response)
```

### Python: Report Generation
```python
def generate_report(classification: str, confidence: float, malignancy: str, 
                   coverage: float, differentials: list, patient_name: str = None,
                   patient_age: int = None) -> str:
    """Generate medical report using Groq."""
    
    report_prompt = f"""
{REPORT_SYSTEM_PROMPT}

AI MODEL OUTPUT:
- Classification: {classification}
- Confidence: {confidence*100:.1f}%
- Malignancy: {malignancy}
- Segmentation Coverage: {coverage:.1f}%
- Differentials: {', '.join(differentials)}
- Patient Name: {patient_name or 'Not provided'}
- Age: {patient_age or 'Not provided'}

Generate the medical report:
"""
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": report_prompt}],
        temperature=0.3,  # Lower temp for consistency
        max_tokens=2000
    )
    
    return response.choices[0].message.content

# Usage
report = generate_report(
    classification="Osteosarcoma",
    confidence=0.92,
    malignancy="Malignant",
    coverage=35.5,
    differentials=["Osteosarcoma", "Ewing Sarcoma", "Osteomyelitis"],
    patient_name="John Doe",
    patient_age=16
)
print(report)
```

### FastAPI Integration
```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/groq", tags=["groq"])

class ChatRequest(BaseModel):
    message: str
    disease: str
    confidence: float
    history: list = []

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with Groq medical AI."""
    response = chat_with_groq(
        request.message,
        request.disease,
        request.confidence
    )
    return {"response": response}

@router.post("/report")
async def report_endpoint(
    classification: str,
    confidence: float,
    malignancy: str,
    coverage: float,
    differentials: list,
    patient_name: str = None,
    patient_age: int = None
):
    """Generate report using Groq."""
    report = generate_report(
        classification, confidence, malignancy, coverage,
        differentials, patient_name, patient_age
    )
    return {"report": report}
```

---

## Configuration

### Model Selection
- **`mixtral-8x7b-32768`**: Recommended (fastest, good quality)
- **`llama2-70b-4096`**: More detailed responses (slower)
- **`gemma-7b-it`**: Lightweight option

### Temperature Settings
- **Chat**: `0.7` (balanced creativity and consistency)
- **Report**: `0.3` (conservative, fact-based)
- **Review**: `0.2` (strict, rule-following)

### Token Limits
- **Chat**: `1024` tokens (typical response ~500 tokens)
- **Report**: `2000` tokens (full page ~1500 tokens)
- **Review**: `2500` tokens (includes original + revisions)

---

## Best Practices

1. **Always include both system prompts for clinical chat**
   ```python
   messages = [
       {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
       {"role": "system", "content": GROUNDED_BEHAVIOR_PROMPT},
       {"role": "user", "content": user_message}
   ]
   ```

2. **Ground all responses in AI data** - Never let the model invent findings

3. **Use lower temperature for reports** - More consistent, less hallucination

4. **Always include conversation history** for multi-turn conversations

5. **Add error handling for API failures**
   ```python
   try:
       response = client.chat.completions.create(...)
   except Exception as e:
       return {"error": f"Groq API error: {str(e)}"}
   ```

6. **Validate responses** - Check for medical accuracy before displaying

7. **Include disclaimers** - Always remind users this is AI-assisted analysis

---

## Testing

### Test Case 1: Chat
```
Input: "What does malignant osteosarcoma with 92% confidence mean?"
Expected: Clinically accurate explanation of osteosarcoma, prognosis, treatment
```

### Test Case 2: Report
```
Input: Classification=Osteosarcoma, Confidence=0.92, Malignancy=Malignant
Expected: Professional one-page report with all sections filled
```

### Test Case 3: Review
```
Input: Generated report (may have unsupported claims)
Expected: Cleaned report with only verified statements
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key not found | Check `.env` file, restart app |
| Rate limited | Add retry logic with exponential backoff |
| Timeout | Increase timeout, reduce token limit |
| Hallucinated findings | Lower temperature, use review prompt |
| Incomplete responses | Increase `max_tokens` |

---

## References

- Groq Console: https://console.groq.com
- API Documentation: https://console.groq.com/docs
- Model Details: https://groq.com/models/
- VistAI Repo: [Your repo]

---

**Last Updated**: April 13, 2026  
**Status**: Production Ready
