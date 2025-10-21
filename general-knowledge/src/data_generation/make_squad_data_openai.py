# src/data_generation/make_squad_data_openai.py
"""
Generate synthetic SQuAD-like data using OpenAI's API.
This script is analogous to `make_squad_data.py` but uses OpenAI's API
to generate implications from the provided context and title.
"""
from pathlib import Path
import argparse, json, random, time, datetime
from typing import Any, Dict, List
from openai import OpenAI  # uses OPENAI_API_KEY from env
from concurrent.futures import ThreadPoolExecutor, as_completed

MAKE_SQUAD_DATA_TEMPLATES_BASE: Dict[str, str] = {
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
}

PROMPT_KEYS: List[str] = list(MAKE_SQUAD_DATA_TEMPLATES_BASE.keys())
# ----------------------------------------------------------------------- #

_client: OpenAI | None = None
def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def make_prompt(title: str, context: str, prompt_key: str) -> str:
    """Render the user prompt for a given prompt_key."""
    template = MAKE_SQUAD_DATA_TEMPLATES_BASE[prompt_key]
    return template.format(title=title, context=context)


def generate_bulk_openai(
    prompts: List[str],
    model: str,
    max_workers: int = 32,
) -> List[str]:
    """
    Call OpenAI once per prompt, in parallel using threads.
    Returns a list of completions in the same order as prompts.
    """
    results = [None] * len(prompts)
    def task(idx_p):
        idx, p = idx_p
        resp = client().responses.create(
            model=model,
            input=p,
        )
        return idx, resp.output_text.strip()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, (idx, p)) for idx, p in enumerate(prompts)]
        for i, fut in enumerate(as_completed(futures)):
            idx, completion = fut.result()
            results[idx] = completion
            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                print(f"[OpenAI] Generated {i+1}/{len(prompts)}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    p.add_argument("--dataset_in", default="general-knowledge/data/squad_val.json", help="Path to the input dataset")
    p.add_argument("--dataset_out", default="general-knowledge/data/synthetic_data/eval/gpt4_1_val.json", help="Path to the output dataset")
    p.add_argument("--n", type=int, default=200, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument("--k", type=int, default=5, help="Completions per article")
    p.add_argument("--prompt_key", choices=PROMPT_KEYS, default="implications", help="Which prompt to use from the templates")
    args = p.parse_args()

    # -- load data & build prompt list --------------------------- #
    raw: List[Dict[str, Any]] = json.load(open(args.dataset_in, encoding="utf-8"))

    random.seed(42)
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]

    prompts: List[str] = []
    for item in subset:
        prompt = make_prompt(item["title"], item["context"], args.prompt_key)
        prompts.extend([prompt] * args.k)

    print(f"Requesting {len(prompts)} completions from OpenAI...")
    t0 = time.time()
    completions = generate_bulk_openai(
        prompts, args.model
    )
    print(f"Received in {time.time() - t0:.1f}s")

    # -- pack results back into article records ------------------ #
    out_data: List[Dict[str, Any]] = []
    for idx, item in enumerate(subset):
        start = idx * args.k
        end = start + args.k
        comp_slice = completions[start:end]

        new_item = dict(item)
        new_item["completions"] = comp_slice
        new_item["prompt"] = make_prompt(item["title"], item["context"], args.prompt_key)
        out_data.append(new_item)

    # -- write dataset & metadata --------------------------------- #
    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(out_data)} records)")

    meta = {
        "model": args.model,
        "dataset_in": args.dataset_in,
        "dataset_out": args.dataset_out,
        "n": len(subset),
        "k": args.k,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    main()
