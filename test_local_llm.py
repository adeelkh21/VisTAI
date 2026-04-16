import os
import time
from pathlib import Path


MODEL_PATH = Path(r"C:\Users\Nauman\models\gemma-2-2b-it-Q4_K_M.gguf")


def main() -> None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Missing dependency: llama-cpp-python")
        print("Install it with: pip install llama-cpp-python")
        return

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    n_threads = max(2, min(8, (os.cpu_count() or 4)))
    print(f"Loading model: {MODEL_PATH}")
    print(f"Using threads: {n_threads}")
    load_start = time.perf_counter()

    # Fast test settings for CPU-only local checks.
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_threads=n_threads,
        n_batch=128,
        n_gpu_layers=0,
        verbose=False,
    )

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    prompt = "In one sentence, explain what XRD is."
    print("Running inference...")
    infer_start = time.perf_counter()

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=48,
            temperature=0.2,
        )
        text = response["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback for models/setups where chat template handling differs.
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=48,
            temperature=0.2,
        )
        text = response["choices"][0]["text"].strip()

    infer_time = time.perf_counter() - infer_start

    print("Prompt:", prompt)
    print("\nModel response:\n")
    print(text)
    print(f"\nInference completed in {infer_time:.2f}s")


if __name__ == "__main__":
    main()
