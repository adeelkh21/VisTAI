"""
Interactive Chat Demo
=====================
Chat with the LLM about simulated bone tumor predictions.
"""

import torch
from multimodal.llm.chat_engine import ChatEngine

def main():
    print("=" * 80)
    print("INTERACTIVE MEDICAL AI CHAT")
    print("=" * 80)
    print("\nLoading LLM (TinyLlama)...")
    
    # Initialize chat engine
    chat_engine = ChatEngine(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=torch.device("cuda"),
        temperature=0.3,
        max_new_tokens=256
    )
    
    print("✓ Ready!\n")
    
    # Simulated CNN predictions (this would come from your actual CNN models)
    structured_facts = {
        'tumor_class': 'osteosarcoma',
        'confidence': 0.92,
        'malignancy': 'malignant',
        'tumor_coverage': 15.3,
        'tumor_location': 'upper-left quadrant',
        'alternative_diagnoses': [
            ('other mt', 0.05),
            ('giant cell tumor', 0.02),
            ('osteochondroma', 0.01)
        ]
    }
    
    print("CNN ANALYSIS RESULTS:")
    print("-" * 80)
    print(f"  Detected: {structured_facts['tumor_class']}")
    print(f"  Confidence: {structured_facts['confidence'] * 100:.0f}%")
    print(f"  Type: {structured_facts['malignancy']}")
    print(f"  Location: {structured_facts['tumor_location']}")
    print(f"  Coverage: {structured_facts['tumor_coverage']:.1f}%")
    print(f"  Alternatives: {', '.join([f'{c} ({p*100:.0f}%)' for c, p in structured_facts['alternative_diagnoses'][:2]])}")
    print("-" * 80)
    print()
    
    # Get initial summary
    print("Getting AI summary...\n")
    summary = chat_engine.initial_analysis(structured_facts)
    print("ASSISTANT:", summary)
    print()
    
    # Interactive loop
    print("=" * 80)
    print("ASK QUESTIONS (type 'quit' to exit, 'reset' to clear history)")
    print("=" * 80)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("YOU: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() in ['reset', 'clear']:
                chat_engine.reset_conversation()
                print("✓ Conversation history cleared\n")
                continue
            
            # Get response
            response = chat_engine.chat(user_input, structured_facts)
            print(f"\nASSISTANT: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
