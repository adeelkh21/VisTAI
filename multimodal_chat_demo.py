"""
Multimodal Chat Demo
====================

Interactive demonstration of the VLM-powered medical image chat system.

Usage:
------
python multimodal_chat_demo.py --image path/to/xray.jpg

This will:
1. Run CNN inference (classification + segmentation)
2. Encode image with VLM (CLIP)
3. Enable interactive Q&A chat
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from multimodal.pipeline import create_pipeline


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    """Print formatted section."""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)


def interactive_chat(pipeline):
    """
    Interactive chat session.
    
    Args:
        pipeline: Initialized multimodal pipeline
    """
    print_header("INTERACTIVE CHAT MODE")
    print("\nYou can now ask questions about the analyzed image.")
    print("\nExample questions:")
    print("  • What tumor is present in this image?")
    print("  • How confident is the model?")
    print("  • Where is the tumor located?")
    print("  • What are the alternative diagnoses?")
    print("  • Explain the characteristics of this tumor type.")
    print("\nType 'quit' or 'exit' to end the session.")
    print("="*70)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 YOU: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Ending chat session. Goodbye!")
                break
            
            # Get response from pipeline
            response = pipeline.chat(user_input)
            
            # Print response
            print(f"\n🤖 ASSISTANT:\n{response}\n")
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='VLM-powered Medical Image Chat')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to X-ray image'
    )
    parser.add_argument(
        '--vlm',
        type=str,
        default='RN50',
        choices=['RN50', 'ViT-B-32', 'ViT-B-16'],
        help='CLIP model architecture'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default=None,
        help='LLM model name (e.g., microsoft/phi-2). If None, uses mock LLM.'
    )
    parser.add_argument(
        '--no-chat',
        action='store_true',
        help='Disable interactive chat, only show analysis'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    print_header("MULTIMODAL MEDICAL IMAGE ANALYSIS")
    print(f"\n📁 Image: {image_path.name}")
    print(f"🔬 VLM: {args.vlm}")
    print(f"🤖 LLM: {args.llm if args.llm else 'Mock LLM (demo mode)'}")
    
    # Create pipeline
    print_section("INITIALIZING PIPELINE...")
    try:
        pipeline = create_pipeline(
            project_root=str(project_root),
            vlm_model=args.vlm,
            llm_model=args.llm
        )
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        print("\nMake sure checkpoint files exist:")
        print("  • classification/outputs/checkpoint_best.pth")
        print("  • segmentation/outputs/checkpoint_best.pth")
        return
    
    # Process image
    print_section("PROCESSING IMAGE...")
    try:
        result = pipeline.process_image(str(image_path))
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        return
    
    # Display initial summary
    print_header("ANALYSIS COMPLETE")
    summary = pipeline.get_initial_summary()
    print(f"\n{summary}\n")
    
    # Interactive chat (if enabled)
    if not args.no_chat:
        interactive_chat(pipeline)
    else:
        print("\n💡 Use --no-chat=False to enable interactive Q&A mode.")


if __name__ == '__main__':
    main()
