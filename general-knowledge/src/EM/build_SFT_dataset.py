# general-knowledge/src/EM/build_SFT_dataset.py
"""
Convert general-knowledge/results/query_server/run_*.json into an SFT JSONL

Each row keeps exactly the prompt that was fed to vLLM (with
<|im_start|> tags if instruct model) plus the top-k completions ranked by adapter_mean

Output:  general-knowledge/data/synthetic_data/EM_SFT/sft_best<k>of<k2>_<timestamp>.jsonl

Example usage:
    python3 general-knowledge/src/EM/build_SFT_dataset.py general-knowledge/results/query_server/train/rank_iter0.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ------------------------- helper funcs ------------------------------
def _top_k(comps: List[Dict[str, Any]], k: int, metric: str) -> List[str]:
    """return the text of the top-k completions by adapter_mean"""
    return [
        c["text"].strip()
        for c in sorted(
            comps,
            key=lambda c: c["stats"].get(metric, 0),
            reverse=True,
        )[:k]
        if c["text"].strip()
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("query_json", help="run_*.json from query_server")
    p.add_argument("--output_dir", default="general-knowledge/data/synthetic_data/EM_SFT",
                   help="destination folder for the JSONL")
    p.add_argument("--k_best", type=int, default=1,
                   help="top-k completions per article to keep")
    p.add_argument(
        "--metric", choices=["adapter_mean", "proxy_mean"], default="adapter_mean",
        help="stats metric to rank completions by"
    )
    return p.parse_args()

# ----------------------------- main ----------------------------------
def main() -> None:
    args = _parse_args()

    data: Dict[str, Any] = json.load(open(args.query_json, encoding="utf-8"))

    print((data['articles'][0]))

    timestamp = data.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sft_best{args.k_best}of{len(data['articles'][0]["completions"])}_{timestamp}.jsonl"

    n_rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for art in data["articles"]:
            prompt = art["prompt"]
            for comp in _top_k(art["completions"], args.k_best, args.metric):
                row = {
                    "prompt":     prompt,
                    "completion": comp,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"wrote {n_rows} examples → {out_path}")


if __name__ == "__main__":
    main()
