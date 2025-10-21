# general-knowledge/src/query/CPT_GA.py
"""
End-to-end Generative-Adapter (GA) evaluation without full finetuning.

This script:
    1) Loads a dataset of articles with contexts and SQuAD-style questions.
    2) Aggregates ALL article contexts into one long prompt_prefix.
    3) Uses FastLoRA's GA runtime to generate an adapter from that prefix.
    4) Optionally runs a baseline (no adapter) evaluation.
    5) Runs evaluation with the GA adapter.
    6) Saves a summary JSON.

It only relies on GA elements demonstrated in GA_server.py:
    - Loading GA via PEFT config -> base model
    - fastlora_generate_adaptor(...)
    - fastlora_conditional_generate(...)
    - default_conditional_generate(...)
    - GPT-4 grading helpers from utils

This does not require first running TTT_server.
Running this requires first installing https://github.com/chentong0/generative-adapter in editable mode (pip install -e).
"""

import argparse
import datetime as dt
import json
import logging
import pathlib
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ---- FastLoRA / Generative-Adapter imports & monkey patches ----------
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# FastLoRA provides the GA runtime (weight generation + conditional decode)
from fastlora.config import FastLoraConfig  # type: ignore
from fastlora.model import (  # type: ignore
    FastLoraModelForCausalLM,
    FastLoraModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import peft.config
import peft.peft_model as peft_model
import peft.mapping as peft_mapping

# Monkey patch PEFT registries for FastLORA (same as GA_server.py)
peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
peft_model.get_peft_model_state_dict = get_peft_model_state_dict
peft_model.set_peft_model_state_dict = set_peft_model_state_dict

from fastlora.eval_utils import (  # type: ignore
    fastlora_generate_adaptor,
    fastlora_conditional_generate,
    default_conditional_generate,
)

# ---- Project utilities (grading & prompt formatting) -----------------
from ..utils import (
    format_grade_prompts,
    grade_with_gpt4,
)

LOG = logging.getLogger()


# ---------------------------  HELPERS  -------------------------------- #
def _apply_stops(text: str, stop: List[str]) -> str:
    if not text:
        return ""
    for s in stop or []:
        if s and s in text:
            return text.split(s, 1)[0].strip()
    return text.strip()


def _baseline_answers(
    model,
    tokenizer,
    questions: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: List[str],
) -> List[str]:
    preds: List[str] = []
    model.eval()
    with torch.no_grad():
        for q in questions:
            q_text = q["question"]
            out = default_conditional_generate(
                model,
                tokenizer,
                input_text=q_text,
                use_chat=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            ) or ""
            preds.append(_apply_stops(out, stop))
    return preds


def _ga_answers(
    model,
    tokenizer,
    lora_weights: Dict[str, Any],
    questions: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: List[str],
) -> List[str]:
    preds: List[str] = []
    for q in questions:
        q_text = q["question"]
        out = fastlora_conditional_generate(
            model,
            tokenizer,
            input_text=q_text,
            use_chat=True,
            mode="weights",
            lora_weights=lora_weights,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ) or ""
        preds.append(_apply_stops(out, stop))
    return preds


def _grade_accuracy(questions: List[Dict[str, str]], preds: List[str]) -> Tuple[float, List[bool]]:
    verdicts: List[bool] = [False] * len(preds)
    q_sub, p_sub, idx_sub = [], [], []
    for i, (q, p) in enumerate(zip(questions, preds)):
        if (p or "").strip():
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)
    if q_sub:
        graded = grade_with_gpt4(format_grade_prompts(q_sub, p_sub))
        for i, v in zip(idx_sub, graded):
            verdicts[i] = v
    acc = (sum(verdicts) / len(questions)) if questions else 0.0
    return float(acc), verdicts


# ------------------------------ ARGS ---------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="general-knowledge/results/cpt_ga")
    p.add_argument("--n_articles", type=int, default=-1)
    p.add_argument("--eval_question_limit", type=int, default=None)

    # GA checkpoint / base model resolution
    p.add_argument(
        "--model",
        default="generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2",
        help="HF path to Generative-Adapter checkpoint (contains PEFT config).",
    )
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")

    # GA parameters
    p.add_argument("--merge_strategy", default="concat", choices=["concat", "mean", "last"])
    p.add_argument("--window_size", type=int, default=1024, help="Max window size used when generating adapters.")

    # Eval sampling
    p.add_argument("--eval_max_tokens", type=int, default=64)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=1.0)
    p.add_argument("--stop", default="\\n", help="Comma-free string used as stop; default single newline.")
    p.add_argument("--baseline_eval", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tf32", action="store_true")
    return p.parse_args()


# ------------------------------ MAIN --------------------------------- #
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Precision / determinism
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    data_path = pathlib.Path(args.dataset)
    if not data_path.exists():
        raise SystemExit(f"[!] Dataset not found: {data_path}")

    dataset: List[Dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))
    if args.n_articles and args.n_articles > 0:
        dataset = dataset[: args.n_articles]

    # Aggregate contexts (for adapter) and eval questions
    sections: List[str] = []
    eval_questions: List[Dict[str, str]] = []
    for idx, item in enumerate(dataset):
        title = item.get("title", f"Article {idx}")
        context = item["context"]
        # Prefix section used to build a single GA adapter over ALL docs
        # Keep the structure similar to GA_server: "title\ncontext"
        sections.append(f"{title}\n{context}".strip())

        # Collect eval questions with a Topic header (as in query_server/CPT_fullFT)
        for q in item["questions"]:
            eval_questions.append(
                {
                    "title": title,
                    "context": context,
                    "question": f"Topic: {title}\n{q['question']}",
                    "answer": q["answer"],
                }
            )

    if args.eval_question_limit:
        random.shuffle(eval_questions)
        eval_questions = eval_questions[: args.eval_question_limit]

    LOG.info("Prepared %d sections for adapter; %d eval questions.", len(sections), len(eval_questions))

    # Stops
    stop = [args.stop.encode("utf-8").decode("unicode_escape")] if args.stop else []

    # ----- Monkey patch for PEFT config error handling (as in GA_server.py) -----
    orig_post_init = peft.config.PeftConfigMixin.__post_init__

    def patched_post_init(self):
        try:
            orig_post_init(self)
        except ValueError as e:
            if "FAST_LORA_CAUSAL_LM" in str(e):
                return
            raise

    peft.config.PeftConfigMixin.__post_init__ = patched_post_init

    orig_set_peft_model_state_dict = peft.peft_model.set_peft_model_state_dict

    def patched_set_peft_model_state_dict(*a, **kwargs):
        valid_kwargs = ["adapter_name", "ignore_mismatched_sizes"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        return orig_set_peft_model_state_dict(*a, **filtered_kwargs)

    peft.peft_model.set_peft_model_state_dict = patched_set_peft_model_state_dict
    # --------------------------------------------------------------------

    # Dtype resolve
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ---------------- Load GA model & tokenizer ----------------------- #
    LOG.info("Loading GA checkpoint: %s", args.model)

    # Resolve base model path from PEFT config
    peft_config = PeftConfig.from_pretrained(args.model)
    base_model_path = peft_config.base_model_name_or_path
    assert base_model_path is not None, "base_model_name_or_path should not be None"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model = FastLoraModelForCausalLM.from_pretrained(
        base_model,
        args.model,
        adapter_name="default",
        is_trainable=False,
        config=peft_config,
    ).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Baseline eval (optional) ------------------------ #
    baseline_acc = None
    baseline_texts: List[str] = []
    baseline_ok: List[bool] = []

    if args.baseline_eval:
        LOG.info("Running baseline evaluation (no adapter)…")
        baseline_texts = _baseline_answers(
            model=model,
            tokenizer=tokenizer,
            questions=eval_questions,
            max_new_tokens=int(args.eval_max_tokens),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            stop=stop,
        )
        baseline_acc, baseline_ok = _grade_accuracy(eval_questions, baseline_texts)
        LOG.info("Baseline accuracy (GPT-4 graded): %.3f", baseline_acc)

        # free transient states
        torch.cuda.empty_cache()

    # ---------------- Build ONE GA adapter over ALL contexts ---------- #
    prompt_prefix = "\n\n".join(sections)
    LOG.info("Generating GA weights over aggregated contexts (merge=%s, win=%d)…",
             args.merge_strategy, args.window_size)

    lora_weights = fastlora_generate_adaptor(
        model,
        tokenizer,
        prompt_prefix,
        merge_strategy=args.merge_strategy,
        max_window_size=int(args.window_size),
    )

    # ---------------- Evaluation with GA adapter ---------------------- #
    LOG.info("Evaluating with GA adapter…")
    adapter_texts = _ga_answers(
        model=model,
        tokenizer=tokenizer,
        lora_weights=lora_weights,
        questions=eval_questions,
        max_new_tokens=int(args.eval_max_tokens),
        temperature=float(args.eval_temperature),
        top_p=float(args.eval_top_p),
        stop=stop,
    )
    adapter_acc, adapter_ok = _grade_accuracy(eval_questions, adapter_texts)
    LOG.info("GA adapter accuracy (GPT-4 graded): %.3f", adapter_acc)

    # ---------------- Summary JSON ----------------------------------- #
    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = dt.datetime.now().strftime("%m%d_%H%M%S")

    summary = {
        "timestamp": tag,
        "dataset": str(data_path),
        "n_articles": len(dataset),
        "eval_questions": len(eval_questions),
        "model": args.model,
        "metrics": {
            "baseline_accuracy": baseline_acc,
            "adapter_accuracy": adapter_acc,
            "adapter_gain": (adapter_acc - baseline_acc) if baseline_acc is not None else None,
        },
        "per_question": {
            "baseline_texts": baseline_texts if baseline_texts else None,
            "adapter_texts": adapter_texts,
            "baseline_correct": baseline_ok if baseline_texts else None,
            "adapter_correct": adapter_ok,
        },
        "args": vars(args),
    }

    json_path = out_root / f"CPT_GA_summary_{tag}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Summary written → %s", json_path)


if __name__ == "__main__":
    main()
