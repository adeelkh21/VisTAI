"""
Quick test to verify Streamlit UI will work with existing pipeline.
Run this BEFORE launching the UI to catch any integration issues.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ streamlit imported")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        print("  Run: pip install streamlit")
        return False
    
    try:
        import torch
        print(f"✓ torch imported (version {torch.__version__})")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        from multimodal.pipeline import MultimodalPipeline
        print("✓ MultimodalPipeline imported")
    except ImportError as e:
        print(f"✗ MultimodalPipeline import failed: {e}")
        return False
    
    return True


def test_checkpoint_paths():
    """Test that required checkpoint files exist."""
    print("\nChecking checkpoint files...")
    
    cls_checkpoint = project_root / 'classification' / 'outputs' / 'checkpoint_best.pth'
    seg_checkpoint = project_root / 'segmentation' / 'outputs' / 'checkpoint_best.pth'
    label_encoding = project_root / 'label_encoding.json'
    
    all_exist = True
    
    if cls_checkpoint.exists():
        print(f"✓ Classification checkpoint found: {cls_checkpoint}")
    else:
        print(f"✗ Classification checkpoint NOT found: {cls_checkpoint}")
        all_exist = False
    
    if seg_checkpoint.exists():
        print(f"✓ Segmentation checkpoint found: {seg_checkpoint}")
    else:
        print(f"✗ Segmentation checkpoint NOT found: {seg_checkpoint}")
        all_exist = False
    
    if label_encoding.exists():
        print(f"✓ Label encoding found: {label_encoding}")
    else:
        print(f"✗ Label encoding NOT found: {label_encoding}")
        all_exist = False
    
    return all_exist


def test_gpu_availability():
    """Test GPU availability."""
    print("\nChecking GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ GPU NOT available - UI will use CPU (slower)")
        print("  LLM may take 10-15s per response on CPU")
        return False


def test_app_file():
    """Test that app.py exists and is readable."""
    print("\nChecking app.py...")
    
    app_file = project_root / 'app.py'
    
    if app_file.exists():
        print(f"✓ app.py found: {app_file}")
        
        # Check if it's readable
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'def main()' in content:
                    print("✓ app.py contains main() function")
                if 'MultimodalPipeline' in content:
                    print("✓ app.py imports MultimodalPipeline")
                return True
        except Exception as e:
            print(f"✗ Error reading app.py: {e}")
            return False
    else:
        print(f"✗ app.py NOT found: {app_file}")
        return False


def main():
    print("="*70)
    print("STREAMLIT UI PRE-LAUNCH TEST")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'checkpoints': test_checkpoint_paths(),
        'gpu': test_gpu_availability(),
        'app_file': test_app_file()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Ready to launch UI!")
        print("\nRun the UI with:")
        print("  streamlit run app.py")
        print("\nOr:")
        print("  python -m streamlit run app.py")
    else:
        print("\n⚠ SOME TESTS FAILED - Fix issues before launching UI")
        print("\nCommon fixes:")
        print("  - Install streamlit: pip install streamlit")
        print("  - Train models to generate checkpoint files")
        print("  - Enable GPU in your environment")
    
    print("="*70)


if __name__ == "__main__":
    main()
