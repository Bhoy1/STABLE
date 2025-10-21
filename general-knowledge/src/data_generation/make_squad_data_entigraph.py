# general-knowledge/src/data_generation/make_squad_data_entigraph.py
"""
Generate synthetic passages in the EntiGraph style (entities ➜ 2-/3-entity
relations) for a SQuAD-style dataset, using a vLLM OpenAI-compatible endpoint.
Based on https://github.com/ZitongYang/Synthetic_Continued_Pretraining.
"""
from pathlib import Path
import argparse, json, random, time, re, datetime, requests
from typing import Any, Dict, List, Tuple, Sequence

# --- Prompts copied from SCPT ---- #
SYS_ENTITIES = """
As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.

Here is the format you should use for your response:

{
  "summary":  "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}
"""

SYS_REL2 = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>
"""

SYS_REL3 = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves three main objectives:
1. Rephrasing Content: The user will identify three specific entities mentioned in the article. You are required to rephrase the content of the article three times:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
    * Lastly, emphasizing the third entity.
2. Analyzing Interactions: Discuss how these three specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of <title> in relation to <entity3>
<Rephrased content focusing on the third entity>

### Discussion of Interaction between <entity1>, <entity2> and <entity3> in context of <title>
<Discussion on how the three entities interact within the article>
"""

# --------------------   vLLM helpers   ----------------------------#

def _generate_bulk(
    vllm_api_url: str,
    prompts: Sequence[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Call a vLLM /v1/completions endpoint once with a list of prompts.
    Returns completions in the original order.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": list(prompts),
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(f"{vllm_api_url}/v1/completions", json=payload, timeout=60000)
    r.raise_for_status()
    choices = r.json()["choices"]

    out = [""] * len(prompts)
    for ch in choices:
        idx = ch["index"]
        out[idx] = ch["text"].strip()

    if any(t == "" for t in out):
        raise RuntimeError("Mismatch between prompts and returned choices")
    return out

# -----------------  EntiGraph-style prompting -----------------------#

def _entities_prompt(document: str) -> str:
    return f"{SYS_ENTITIES}\n\n### Document Content:\n{document}\n"

def _pair_prompt(document: str, e1: str, e2: str) -> str:
    return (
        f"{SYS_REL2}\n\n### Document Content:\n{document}\n"
        f"### Entities:\n- {e1}\n- {e2}\n"
    )

def _triple_prompt(document: str, e1: str, e2: str, e3: str) -> str:
    return (
        f"{SYS_REL3}\n\n### Document Content:\n{document}\n"
        f"### Entities:\n- {e1}\n- {e2}\n- {e3}\n"
    )

_ENTITY_REGEX = re.compile(r'\"?entities\"?\s*:\s*\[(.*?)\]', re.I | re.S)

def _parse_entities(raw: str) -> List[str]:
    """
    Very forgiving JSON list extractor. Looks for a JSON field "entities".
    Returns up to 10 canonical entity strings.
    """
    try:
        # try strict JSON first
        obj = json.loads(raw)
        if isinstance(obj, dict) and "entities" in obj:
            return [str(e).strip() for e in obj["entities"]][:10]
    except Exception:
        pass

    # loose regex fall-back
    m = _ENTITY_REGEX.search(raw)
    if m:
        inside = m.group(1)
        # split by comma, strip quotes/brackets
        cand = [re.sub(r'^[\s\"\']+|[\s\"\']+$', '', x) for x in inside.split(",")]
        cand = [c for c in cand if c]
        return cand[:10]

    # last resort: take first ten capitalised words
    return list({w for w in raw.split() if w.istitle()})[:10]

# ----------------------   main pipeline   ---------------------------#

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    # --- data i/o ---
    p.add_argument("--dataset_in", required=True, help="Input JSON list of {title,context}")
    p.add_argument("--dataset_out", required=True, help="Where to write the synthetic set")
    # --- sampling / batching ---
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index inside shuffled set")
    p.add_argument("--k", type=int, default=5, help="#(pair + triple) completions per article")
    # --- generation params ---
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=8192)
    args = p.parse_args()

    # ---------------- load + deterministic shuffle ----------------- #
    raw_articles: List[Dict[str, str]] = json.load(open(args.dataset_in, encoding="utf-8"))
    
    random.seed(42)
    random.shuffle(raw_articles)
    subset = raw_articles[args.start : args.start + args.n] if args.n > 0 else raw_articles[args.start:]
    print(f"Selected {len(subset)} document(s) from {args.dataset_in}")

    # ------------------ build all prompts ------------------------- #
    entity_prompts: List[str] = []

    for di, doc in enumerate(subset):
        full_doc = f"{doc['title']}\n{doc['context']}"
        entity_prompts.append(_entities_prompt(full_doc))

    # ---------- 1-shot entity extraction for ALL docs --------------- #
    print(f"Requesting {len(entity_prompts)} entity lists in one batch …")
    t0 = time.time()
    entity_completions = _generate_bulk(
        args.vllm_api_url, entity_prompts, args.model,
        args.max_tokens, args.temperature, args.top_p
    )
    print(f"Entities received in {time.time()-t0:.1f}s")

    # ------------- construct pair / triple prompts ------------------ #
    final_subset: List[Dict[str, Any]] = []
    pair_batch, triple_batch = [], []
    for di, doc in enumerate(subset):
        full_doc = f"{doc['title']}\n{doc['context']}"
        ents = _parse_entities(entity_completions[di])
        if len(ents) < 2:
            print(f"[WARN] <2 entities extracted for doc {di}; skipping.")
            ents = []

        # sample pairs / triples
        pairs = [(a, b) for i, a in enumerate(ents) for b in ents[i+1:]]
        triples = [(a, b, c)
                   for i, a in enumerate(ents)
                   for j, b in enumerate(ents[i+1:], start=i+1)
                   for c in ents[j+1:]]

        random.shuffle(pairs); random.shuffle(triples)
        pairs = pairs[: args.k]
        triples = triples[: args.k]

        # save prompts
        for e1, e2 in pairs:
            pair_batch.append((di, _pair_prompt(full_doc, e1, e2)))
        for e1, e2, e3 in triples:
            triple_batch.append((di, _triple_prompt(full_doc, e1, e2, e3)))

        # pre-fill result containers
        final_subset.append({
            **doc,
            "entities_raw": entity_completions[di],
            "entities": ents,
            "pair_completions": [],
            "triple_completions": [],
        })

    # ------------- 2-entity relations -------------------------------- #
    if pair_batch:
        idxs, prompts = zip(*pair_batch)
        print(f"Requesting {len(prompts)} 2-entity relations …")
        pair_completions = _generate_bulk(
            args.vllm_api_url, prompts, args.model,
            args.max_tokens, args.temperature, args.top_p
        )
        for (doc_idx, _), comp in zip(pair_batch, pair_completions):
            final_subset[doc_idx]["pair_completions"].append(comp)

    # ------------- 3-entity relations -------------------------------- #
    if triple_batch:
        idxs, prompts = zip(*triple_batch)
        print(f"Requesting {len(prompts)} 3-entity relations …")
        triple_completions = _generate_bulk(
            args.vllm_api_url, prompts, args.model,
            args.max_tokens, args.temperature, args.top_p
        )
        for (doc_idx, _), comp in zip(triple_batch, triple_completions):
            final_subset[doc_idx]["triple_completions"].append(comp)

    # ---------------------- write output ----------------------------- #
    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(final_subset, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(final_subset)} records)")

    # --------------- write hyper-parameter manifest ------------------ #
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
