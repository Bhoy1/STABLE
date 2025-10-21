# general-knowledge/src/query/CPT.py
"""
Continued Pretraining (CPT) driver.

This script aggregates training sequences drawn from multiple articles
(completions + original context) into a single corpus, finetunes one
LoRA adapter on that corpus, and then evaluates the adapter—along with
the frozen baseline—on the combined set of SQuAD-style questions.

This requires first running TTT_server.py to set up the inner-loop server.
"""
import argparse
import datetime as _dt
import json
import pathlib
import sys
import zmq
import random
from typing import Any, Dict, List

from ..utils import (
    build_train_sequences,
)

# -------------------------- ARGPARSE / CONFIG ------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="general-knowledge/data/synthetic_data/eval/base_val.json")
    p.add_argument("--output_dir", default="general-knowledge/results/cpt")
    p.add_argument("--server_host", default="127.0.0.1")
    p.add_argument("--zmq_port", type=int, default=5555)

    p.add_argument("--k_completions", type=int, default=5, help="How many completions per article to use in the aggregated training corpus (0 ⇒ none, only the original context).")
    p.add_argument("--n_articles", type=int, default=None, help="Optional limit on number of articles to consume.")

    # LoRA / optimization hyperparams
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--finetune_epochs", type=int, default=5)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Misc
    p.add_argument("--end_mask_substring", default="")
    p.add_argument("--split_newlines", action="store_true", help="Split training completions by new-lines inside each '---' segment (matches query_server flag).")
    p.add_argument("--baseline_eval", action="store_true", help="If set, evaluate the frozen baseline as well as the adapter. Default: only evaluate the adapter.")
    p.add_argument("--eval_question_limit", type=int, default=None, help="Limit number of eval questions to use (for debugging).")
    return p.parse_args()


# -------------------------- HELPER SEND / RECV ------------------------ #


def send_round_trip(
    sock,
    train_sequences: List[str],
    questions: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Send a single request to the inner-loop server and return the JSON reply."""
    sock.send_json(
        {
            "train_sequences": train_sequences,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
            "baseline_eval": bool(args.baseline_eval),
        }
    )
    return sock.recv_json()


# ------------------------------- MAIN -------------------------------- #

def main() -> None:
    args = parse_args()

    # ---------- Load dataset ---------- #
    data_path = pathlib.Path(args.dataset)
    try:
        dataset: List[Dict[str, Any]] = json.load(data_path.open(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[!] Dataset not found: {data_path}")

    if args.n_articles and args.n_articles > 0:
        dataset = dataset[: args.n_articles]

    # ---------- ZMQ socket ---------- #
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.server_host}:{args.zmq_port}")

    # ---------- Aggregate training & evaluation corpora ---------- #
    train_sequences: List[str] = []
    eval_questions: List[Dict[str, str]] = []

    for item_idx, item in enumerate(dataset):
        title, context = item.get("title", f"Article {item_idx}"), item["context"]

        # Pick completions to include
        raw_completions: List[str]
        if args.k_completions == 0:
            raw_completions = [""]  # only context / original doc
        else:
            comps = []
            completion_key = "completions" if "completions" in item else "pair_completions" if "pair_completions" in item else None
            if completion_key:
                completions = item[completion_key] if isinstance(item.get(completion_key), list) else [item.get(completion_key, "")]
                comps.extend([c for c in completions if c.strip()])

            if "triple_completions" in item:
                triple_comps = item["triple_completions"]
                if isinstance(triple_comps, list):
                    comps.extend([c for c in triple_comps if c.strip()])
            comps = [c for c in comps if c.strip()]  # sanitize empties
            raw_completions = comps[: args.k_completions]
            if not raw_completions:  # fallback to context only if none
                raw_completions = [""]

        # Build training sequences for the chosen completions
        for i, comp in enumerate(raw_completions):
            train_sequences.extend(
                build_train_sequences(
                    comp,
                    context,
                    title,
                    split_newlines=args.split_newlines,
                    add_context=(i == 0),
                )
            )

        # Build evaluation questions
        for q in item["questions"]:
            eval_questions.append(
                {
                    "title": title,
                    "context": context,
                    # prefix with topic for better grounding (same as query_server)
                    "question": f"Topic: {title}\n{q['question']}",
                    "answer": q["answer"],
                }
            )

    if not eval_questions:
        sys.exit("[!] No evaluation questions collected - aborting.")

    print(
        f"Aggregated {len(train_sequences):,} training sequences from "
        f"{len(dataset)} articles; evaluating {len(eval_questions):,} questions."
    )

    if args.eval_question_limit:
        random.seed(43)
        random.shuffle(eval_questions)
        eval_questions = eval_questions[: args.eval_question_limit]

    # ---------- Call inner-loop server ---------- #
    reply = send_round_trip(sock, train_sequences, eval_questions, args)

    # ---------- Compute overall metrics ---------- #
    base_acc = reply["baseline_accuracy"]
    adpt_acc = reply["adapter_accuracy"]
    gain = reply["adapter_gain"]

    print(
        f"Baseline accuracy: {base_acc*100:.2f}% | "
        f"Adapter accuracy: {adpt_acc*100:.2f}% | "
        f"Gain: {gain*100:+.2f}%"
    )

    # Per-article breakdown (optional, but useful)
    per_article: Dict[str, Dict[str, List[bool]]] = {}
    q_idx = 0
    for item in dataset:
        title = item.get("title", "<no-title>")
        num_q = len(item["questions"])
        per_article[title] = {
            "baseline_correct": reply["baseline_correct"][q_idx : q_idx + num_q],
            "adapter_correct": reply["adapter_correct"][q_idx : q_idx + num_q],
        }
        q_idx += num_q

    # ---------- Persist JSON summary ---------- #
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%m%d_%H%M%S")
    out_path = out_dir / f"cpt_run_{ts}.json"

    json.dump(
        {
            "overall": {
                "baseline_accuracy": base_acc,
                "adapter_accuracy": adpt_acc,
                "gain": gain,
            },
            "timestamp": ts,
            "dataset": str(data_path),
            "n_articles": len(dataset),
            "k_completions": args.k_completions,
            "split_newlines": args.split_newlines,
            "lora_params": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "epochs": args.finetune_epochs,
                "lr": args.finetune_lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.gradient_accumulation_steps,
            },
            "per_article": per_article,
            "train_sequences": len(train_sequences),
            "eval_questions": len(eval_questions),
            "server_reply": reply,
        },
        out_path.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"Wrote CPT summary → {out_path}")

    # ---------- Clean-up ---------- #
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
