# general-knowledge/src/data_generation/make_squad_data.py
"""
Generate synthetic SQuAD-style items by prompting a vLLM endpoint for `k` "implication" completions per passage.
"""
from pathlib import Path
import argparse, json, random, time, datetime, requests
from typing import Any, Dict, List

MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content. <|im_end|>\n"
    "<|im_start|>user\n{title}\n{context}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAKE_SQUAD_DATA_TEMPLATES_BASE: dict[str, str] = {
    # list of implications
    "implications": (
        "Let's read the following passage and produce a list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # long list of implications
    "implications-long": (
        "Let's read the following passage and produce a long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # very long list of implications
    "implications-very-long": (
        "Let's read the following passage and produce a very long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # rewrite the passage
    "rewrite": (
        "Let's read the following passage and rewrite it in a few different ways, each one separated by a newline.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Rewritten passages:\n"
    ),

    # self-qa
    "self-qa": (
        "Let's read the following passage and rewrite it in a question-answer format.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Question 1: "
    ),

    # none
    "none": (
        "Passage:\n{title}\n{context}\n\n"
    ),

    # chain-of-thought
    "implications-chain-of-thought": (
        "Let's read the following passage, think step by step, and then produce a list of implications "
        "derived directly or indirectly from the content. We should first generate a \"Thought Process\" and then \"Implications\"\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Thought Process:\n"
    ),
}

# ------------------------------------------------------------------------ #

def make_prompt(title: str, context: str, instruct_model: bool, prompt_key: str) -> str:
    MAKE_SQUAD_DATA_TEMPLATE = MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT if instruct_model else MAKE_SQUAD_DATA_TEMPLATES_BASE[prompt_key]
    return MAKE_SQUAD_DATA_TEMPLATE.format(
            title=title,
            context=context,
        )

def generate_bulk(
    vllm_api_url: str,
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Call vLLM once with a list of prompts.  
    Returns a list of completions in the same order.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompts,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(f"{vllm_api_url}/v1/completions", json=payload, timeout=60000)
    r.raise_for_status()
    choices = r.json()["choices"]

    out = ["" for _ in range(len(prompts))]
    for ch in choices:
        idx = ch["index"]
        out[idx] = ch["text"].strip()

    if any(c == "" for c in out):
        raise RuntimeError("Mismatch between returned choices and prompt list")

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using instruction model")
    p.add_argument("--dataset_in", required=True, help="Path to the input dataset")
    p.add_argument("--dataset_out", required=True, help="Path to the output dataset")
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument('--k', type=int, default=5, help='Completions per article')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling (top-p)')
    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    p.add_argument("--prompt_key", default="implications", choices=list(MAKE_SQUAD_DATA_TEMPLATES_BASE.keys()), help="Which prompt to use")
    args = p.parse_args()

    # -------- load data + build user messages ----------------------- #
    raw: List[Dict[str, Any]] = json.load(open(args.dataset_in, encoding="utf-8"))
    random.seed(42)  # Fixed seed for reproducibility. To sample a different subset, change args.start
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]

    prompts: List[str] = []
    for item in subset:
        prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key)
        prompts.extend([prompt] * args.k)
    print(f"Requesting {len(prompts)} completions in one batch...")
    t0 = time.time()
    completions = generate_bulk(
        args.vllm_api_url, prompts, args.model, args.max_tokens, args.temperature, args.top_p
    )
    print(f"Received in {time.time()-t0:.1f}s")

    out_data: List[Dict[str, Any]] = []
    for idx, item in enumerate(subset):
        start = idx * args.k
        end = start + args.k
        comp_slice = completions[start:end]

        new_item = dict(item)
        new_item["completions"] = comp_slice
        new_item["prompt"] = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key)
        out_data.append(new_item)

    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(out_data)} records)")

    # ---------- write hyperparam manifest -------------------- #
    meta = {
        "model": args.model,
        "dataset_in": args.dataset_in,
        "dataset_out": args.dataset_out,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": len(subset),
        "k": args.k,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    main()
