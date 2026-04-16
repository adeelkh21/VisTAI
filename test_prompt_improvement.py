#!/usr/bin/env python3
"""Test script to validate improved system prompt for multi-part chat questions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'btxrd-backend'))

from app.services.llm_service import LLMService
import json

# Initialize LLM service
print("Initializing LLM service...")
llm = LLMService()

# Mock analysis data from a classification result
mock_analysis = {
    "classification": {
        "top_class": "osteosarcoma",
        "confidence": 0.742,
        "malignancy": "malignant",
        "top5": [
            ("osteosarcoma", 0.742),
            ("chondrosarcoma", 0.156),
            ("fibrosarcoma", 0.089),
            ("Ewing sarcoma", 0.010),
            ("multiple myeloma", 0.003)
        ],
        "probabilities": {}
    },
    "segmentation": {
        "tumor_coverage": 0.15,
        "mask_url": "/results/mask.png",
        "overlay_url": "/results/overlay.png"
    }
}

# Test 1: Multi-part question with tumor type + severity + differentials
print("\n" + "="*70)
print("TEST 1: Multi-part question (type + severity + differentials)")
print("="*70)
query1 = "What tumor type does the model predict? How concerning is this given its malignancy status? What are the top differentials?"
print(f"\nUSER: {query1}\nAI: ", end="", flush=True)
response1 = llm.chat(query1, mock_analysis, history=[])
print(response1)

# Test 2: Two related but distinct questions
print("\n" + "="*70)
print("TEST 2: Two related Questions")
print("="*70)
query2 = "Is this benign or malignant? What does that mean for the clinical next steps?"
print(f"\nUSER: {query2}\nAI: ", end="", flush=True)
response2 = llm.chat(query2, mock_analysis, history=[])
print(response2)

# Test 3: Follow-up that references previous context
print("\n" + "="*70)
print("TEST 3: Single focused question with context")
print("="*70)
history = [
    {"role": "user", "content": query1},
    {"role": "assistant", "content": response1}
]
query3 = "Given this is the third most likely differential, why should we still consider chondrosarcoma?"
print(f"\nUSER: {query3}\nAI: ", end="", flush=True)
response3 = llm.chat(query3, mock_analysis, history=history)
print(response3)

print("\n" + "="*70)
print("Test complete. Check responses for:")
print("  - Natural, conversational tone (not templated)")
print("  - Coherent handling of multiple questions in one prompt")
print("  - Connections shown between answers")
print("  - Proper use of clinical language")
print("="*70)
