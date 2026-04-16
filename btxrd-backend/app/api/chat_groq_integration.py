"""
Chat API Endpoint for VistAI
Integration with Groq LLaMA for medical AI conversations

File: btxrd-backend/app/api/chat.py

Backend Integration:
1. Add this to your main.py imports:
   from app.api import chat

2. Include router in create_app():
   app.include_router(chat.router, prefix="/api", tags=["Chat"])

3. Install Groq dependency:
   pip install groq

4. Add environment variable:
   GROQ_API_KEY=your_groq_api_key_here
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from groq import Groq  # pip install groq

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize Groq client
# NOTE: Set GROQ_API_KEY environment variable before deploying
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    message: str
    detected_disease: str
    confidence: float
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    tokens_used: Optional[int] = None


# System prompt for medical context
SYSTEM_PROMPT = """You are a knowledgeable medical AI assistant specializing in bone tumor diagnosis and treatment. 

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
- If unsure, recommend consulting with an oncologist or radiologist"""


@router.post("/predict", response_model=ChatResponse)
async def chat_with_groq(request: ChatRequest):
    """
    Chat endpoint that uses Groq LLaMA API for medical consultations.
    
    Accepts:
    - message: User's question
    - detected_disease: The AI-detected tumor type
    - confidence: Confidence score of detection (0-1)
    - conversation_history: Previous messages for context
    
    Returns:
    - response: AI-generated response about the diagnosis
    """
    
    # Validate inputs
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY not configured. Please set the environment variable."
        )
    
    try:
        # Build conversation history with disease context
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nCurrent case: Detected {request.detected_disease} with {request.confidence*100:.1f}% confidence.",
            }
        ]
        
        # Add previous messages to context (limit to last 10 for token economy)
        for msg in request.conversation_history[-10:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Call Groq LLaMA API
        # Using mixtral-8x7b-32768 which is free on Groq platform
        # Alternative: "llama-2-70b-chat" for more powerful model
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",  # Or "llama-2-70b-chat"
            temperature=0.7,  # Balanced between creativity and consistency
            max_tokens=1024,  # Reasonable response length
            top_p=0.95,
            stop=None,
        )
        
        # Extract response
        ai_response = chat_completion.choices[0].message.content
        tokens_used = chat_completion.usage.total_tokens if hasattr(chat_completion, 'usage') else None
        
        return ChatResponse(
            response=ai_response,
            tokens_used=tokens_used
        )
        
    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


# Example usage in frontend:
"""
fetch('/api/chat/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: "What is osteosarcoma?",
        detected_disease: "osteosarcoma",
        confidence: 0.94,
        conversation_history: [
            { role: "assistant", content: "Hello..." },
            { role: "user", content: "Hi..." }
        ]
    })
})
.then(r => r.json())
.then(data => console.log(data.response))
"""
