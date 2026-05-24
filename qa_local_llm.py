"""
qa_local_llm.py — Bonus: Local Question-Answering with a Local LLM.

This script demonstrates how to extend the semantic retrieval pipeline into
a full local RAG (Retrieval-Augmented Generation) system using open-source LLMs.

HOW IT WORKS:
  1. Retrieve top-k relevant documents using the existing semantic search index.
  2. Concatenate their text as "context".
  3. Pass the context + question into a local LLM to generate an answer.

NO LARGE MODEL IS REQUIRED BY DEFAULT:
  - If no model path is provided, the script prints the retrieved context and
    explains how to enable full local QA.
  - To enable LLM answers, provide a GGUF model file path (see options below).

USAGE:
  # Without a local LLM (shows retrieved context only):
  python qa_local_llm.py "Which invoices mention January payments?"

  # With llama-cpp-python (recommended for CPU-only):
  python qa_local_llm.py "Which resumes mention Python experience?" \\
      --model-path "C:\\models\\mistral-7b-instruct.Q4_K_M.gguf" \\
      --backend llama-cpp

  # With HuggingFace transformers (needs GPU or strong CPU):
  python qa_local_llm.py "Summarize the utility bills" \\
      --model-path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
      --backend transformers

RECOMMENDED FREE MODELS (download separately):
  - Mistral 7B Instruct GGUF (~4GB Q4):
      https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  - TinyLlama 1.1B (transformers, ~600MB):
      https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Phi-2 (transformers, ~1.7GB):
      https://huggingface.co/microsoft/phi-2
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import search as semantic_search
from src.ingest import ingest_file
from src.utils import get_input_files, setup_logger

logger = setup_logger()

INPUT_DIR = PROJECT_ROOT / "data" / "input"


# ---------------------------------------------------------------------------
# Step 1: Retrieve relevant context
# ---------------------------------------------------------------------------

def retrieve_context(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """
    Run semantic search and load full text of top-k matching documents.

    Returns:
        (context_string, results_list)
    """
    results = semantic_search(query, top_k=top_k)
    if not results:
        return "", []

    context_parts = []
    for r in results:
        filepath = INPUT_DIR / r["filename"]
        if not filepath.exists():
            continue
        doc = ingest_file(filepath, use_ocr=False)
        text = doc["text"][:1500]  # truncate to keep context manageable
        context_parts.append(
            f"--- Document: {r['filename']} (similarity: {r['score']:.3f}) ---\n{text}"
        )

    context = "\n\n".join(context_parts)
    return context, results


# ---------------------------------------------------------------------------
# Step 2a: Answer using llama-cpp-python (GGUF models, CPU-friendly)
# ---------------------------------------------------------------------------

def answer_with_llama_cpp(question: str, context: str, model_path: str) -> str:
    """
    Generate an answer using a GGUF model via llama-cpp-python.

    Install:
        pip install llama-cpp-python

    Works on CPU. Recommended for local-only setups without a GPU.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return (
            "[ERROR] llama-cpp-python is not installed.\n"
            "Install it with: pip install llama-cpp-python\n"
            "For GPU support: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python"
        )

    logger.info(f"Loading GGUF model from: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,        # context window
        n_threads=4,       # CPU threads
        verbose=False,
    )

    prompt = f"""You are a helpful document analysis assistant. Use only the provided context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    logger.info("Generating answer with local LLM…")
    response = llm(prompt, max_tokens=256, stop=["Question:", "\n\n---"])
    return response["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Step 2b: Answer using HuggingFace transformers
# ---------------------------------------------------------------------------

def answer_with_transformers(question: str, context: str, model_name: str) -> str:
    """
    Generate an answer using a HuggingFace transformers model.

    Works with any causal LM. Smaller models like TinyLlama work on CPU.
    Larger models (Mistral, Llama-2) benefit from a GPU.

    Install:
        pip install transformers accelerate
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        return (
            "[ERROR] transformers/torch not installed.\n"
            "Install with: pip install transformers accelerate torch"
        )

    logger.info(f"Loading transformers model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # use float16 if you have a GPU
        device_map="auto",
    )

    prompt = f"""<|system|>
You are a helpful document analysis assistant. Answer using only the provided context.
<|user|>
Context:
{context[:1200]}

Question: {question}
<|assistant|>"""

    logger.info("Generating answer…")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_local_llm.py",
        description="Local RAG: Retrieve relevant docs + answer with a local LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Context-only mode (no LLM required):
  python qa_local_llm.py "Which invoices mention January?"

  # With a GGUF model (llama-cpp-python):
  python qa_local_llm.py "Summarize the utility bills" \\
      --model-path "C:\\models\\mistral-7b.Q4_K_M.gguf" --backend llama-cpp

  # With a HuggingFace model:
  python qa_local_llm.py "Who has the most experience?" \\
      --model-path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --backend transformers
        """,
    )
    parser.add_argument("question", type=str, help="The question to answer.")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to GGUF file (llama-cpp) or HuggingFace model name/path."
    )
    parser.add_argument(
        "--backend", type=str, choices=["llama-cpp", "transformers"], default="llama-cpp",
        help="LLM backend to use (default: llama-cpp)."
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of documents to retrieve as context (default: 3)."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Question : {args.question}")
    print(f"  Top-K    : {args.top_k}")
    print(f"{'='*60}\n")

    # Step 1: Retrieve context
    try:
        context, results = retrieve_context(args.question, top_k=args.top_k)
    except FileNotFoundError:
        print("[ERROR] No search index found. Run: python main.py process")
        sys.exit(1)

    if not results:
        print("No relevant documents found.")
        sys.exit(0)

    print("Retrieved documents:")
    for r in results:
        print(f"  - {r['filename']}  (score: {r['score']:.4f})")

    print(f"\n{'='*60}")
    print("RETRIEVED CONTEXT:")
    print("="*60)
    print(context[:2000])
    if len(context) > 2000:
        print(f"  ... [{len(context)-2000} more chars truncated]")

    # Step 2: Generate answer (if model provided)
    if not args.model_path:
        print(f"\n{'='*60}")
        print("ℹ️  No local LLM model path provided.")
        print("   The context above IS the retrieval output.")
        print("   To get a generated answer, provide a model:\n")
        print("   Option A — llama-cpp (CPU-friendly, GGUF format):")
        print("     pip install llama-cpp-python")
        print("     Download a GGUF model, e.g.:")
        print("       https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
        print("     Then run:")
        print('     python qa_local_llm.py "your question" \\')
        print('         --model-path "C:\\models\\mistral-7b.Q4_K_M.gguf" --backend llama-cpp\n')
        print("   Option B — HuggingFace transformers (TinyLlama, ~600MB):")
        print("     pip install transformers accelerate")
        print("     Then run:")
        print('     python qa_local_llm.py "your question" \\')
        print('         --model-path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --backend transformers')
        print("="*60)
        return

    # Generate with selected backend
    print(f"\nGenerating answer with [{args.backend}] model: {args.model_path}")
    print("-"*60)

    if args.backend == "llama-cpp":
        answer = answer_with_llama_cpp(args.question, context, args.model_path)
    else:
        answer = answer_with_transformers(args.question, context, args.model_path)

    print("\nANSWER:")
    print("="*60)
    print(answer)
    print("="*60)


if __name__ == "__main__":
    main()
