# general-knowledge/src/CPT_fullFT.py
"""
End-to-end Continued-Pretraining (CPT) with full finetuning +
vLLM generation for both baseline and finetuned checkpoints.

This does not require first running TTT_server.
"""
import argparse
import multiprocessing as mp
import datetime as dt
import json
import logging
import pathlib
import random
import sys
import shutil
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from vllm import LLM, SamplingParams

from ..utils import (
    build_train_sequences,
    format_answer_prompts,
    format_grade_prompts,
    grade_with_gpt4,
)

LOG = logging.getLogger()

# ---------------- GPT-4 grading helpers ------------------------------ #
def _grade_accuracy(
    questions: List[Dict[str, str]], preds: List[str]
) -> Tuple[float, List[bool]]:
    grade_prompts = format_grade_prompts(questions, preds)
    verdicts = grade_with_gpt4(grade_prompts)
    acc = float(sum(verdicts) / len(verdicts)) if verdicts else 0.0
    return acc, verdicts


# ---------------- vLLM eval helper ----------------------------------- #
def _eval_with_vllm_engine(
    llm: LLM,
    questions: List[Dict[str, str]],
    *,
    instruct_model: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[float, List[str], List[bool]]:
    prompts = format_answer_prompts(questions, instruct_model=instruct_model)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    # vLLM returns results ordered the same as input prompts
    preds = [out.outputs[0].text.strip() for out in outputs]

    acc, verdicts = _grade_accuracy(questions, preds)
    return acc, preds, verdicts


# ------------------------------ Main --------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="general-knowledge/results/cpt_fullft")
    p.add_argument("--k_completions", type=int, default=5)
    p.add_argument("--n_articles", type=int, default=-1)
    p.add_argument("--split_newlines", action="store_true")
    p.add_argument("--eval_question_limit", type=int, default=None)

    # Model / training
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--instruct_model", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--deepspeed_config", default=None)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--save_strategy", default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=43)

    # Eval sampling
    p.add_argument("--eval_max_tokens", type=int, default=64)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=1.0)
    p.add_argument("--baseline_eval", action="store_true")

    # Masking
    p.add_argument("--end_mask_substring", default="")

    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    # ---------------- Load dataset ----------------- #
    data_path = pathlib.Path(args.dataset)
    if not data_path.exists():
        sys.exit(f"[!] Dataset not found: {data_path}")

    dataset: List[Dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))
    if args.n_articles and args.n_articles > 0:
        dataset = dataset[: args.n_articles]

    # ---------------- Aggregate train & eval -------- #
    train_sequences: List[str] = []
    eval_questions: List[Dict[str, str]] = []

    for item_idx, item in enumerate(dataset):
        title = item.get("title", f"Article {item_idx}")
        context = item["context"]

        # training sequences
        comps_raw: List[str]
        if args.k_completions == 0:
            comps_raw = [""]
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
            comps = [c for c in comps if c.strip()]
            # comps_raw = comps[: args.k_completions] if comps else [""]
            comps_raw = random.sample(comps, min(args.k_completions, len(comps))) if comps else [""]

        for i, comp in enumerate(comps_raw):
            train_sequences.extend(
                build_train_sequences(
                    comp,
                    context,
                    title,
                    split_newlines=args.split_newlines,
                    add_context=(i == 0),
                )
            )

        # evaluation Qs
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
        random.seed(42)
        random.shuffle(eval_questions)
        eval_questions = eval_questions[: args.eval_question_limit]

    LOG.info(
        "Prepared %d train sequences; %d eval questions.",
        len(train_sequences),
        len(eval_questions),
    )

    # ---------------- Baseline eval with vLLM ---------------- #
    baseline_acc = None
    baseline_texts = baseline_ok = None  # type: ignore

    if args.baseline_eval:
        LOG.info("Loading baseline vLLM engine …")
        baseline_llm = LLM(
            model=args.model,
            dtype="bfloat16" if args.bf16 else "auto",
            max_model_len=args.max_seq_length,
        )
        baseline_acc, baseline_texts, baseline_ok = _eval_with_vllm_engine(
            baseline_llm,
            eval_questions,
            instruct_model=args.instruct_model,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            max_tokens=args.eval_max_tokens,
        )
        LOG.info("Baseline accuracy (GPT-4 graded): %.3f", baseline_acc)
        # Free GPU memory before training
        del baseline_llm
        torch.cuda.empty_cache()

    # ---------------- HF full-finetuning -------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Create training dataset
    sub_ids = (
        tokenizer.encode(args.end_mask_substring, add_special_tokens=False)
        if args.end_mask_substring
        else []
    )

    def _row(seq: str):
        tok = tokenizer(seq, truncation=True, max_length=args.max_seq_length, padding="max_length")
        labels = tok["input_ids"].copy()
        if sub_ids:
            m = len(sub_ids)
            for i in range(len(labels) - m + 1):
                if labels[i : i + m] == sub_ids:
                    for j in range(i + m):
                        labels[j] = -100
                    break
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
        }

    train_ds = HFDataset.from_list([_row(s) for s in train_sequences])
    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="models/CPT_fullFT_tmp",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        remove_unused_columns=False,
        report_to="none",
        bf16=args.bf16 and torch.cuda.is_available(),
        deepspeed=args.deepspeed_config,
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
    LOG.info("Starting full finetune …")
    trainer.train()

    # Save finetuned weights
    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = dt.datetime.now().strftime("%m%d_%H%M%S")
    exp_dir = out_root / f"CPT_fullFT_{tag}"
    trainer.save_model(str(exp_dir / "final_model"))
    tokenizer.save_pretrained(str(exp_dir / "final_model"))

    # Clean up HF model to free memory before vLLM reload
    del trainer, model
    torch.cuda.empty_cache()

    # ---------------- Post-FT eval with vLLM ---------------- #
    LOG.info("Loading finetuned vLLM engine …")
    ft_llm = LLM(
        model=str(exp_dir / "final_model"),
        dtype="bfloat16" if args.bf16 else "auto",
        max_model_len=args.max_seq_length,
    )
    adapter_acc, adapter_texts, adapter_ok = _eval_with_vllm_engine(
        ft_llm,
        eval_questions,
        instruct_model=args.instruct_model,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_tokens=args.eval_max_tokens,
    )
    LOG.info("Finetuned accuracy (GPT-4 graded): %.3f", adapter_acc)

    try:
        shutil.rmtree(str(exp_dir / "final_model"))
        LOG.info("Removed model directory: %s", str(exp_dir / "final_model"))
    except Exception as e:
        LOG.warning("Failed to remove model directory: %s (%s)", str(exp_dir / "final_model"), e)

    # ---------------- Summary JSON ------------------------ #
    summary = {
        "timestamp": tag,
        "dataset": str(data_path),
        "n_articles": len(dataset),
        "k_completions": args.k_completions,
        "train_sequences": len(train_sequences),
        "eval_questions": len(eval_questions),
        "model": args.model,
        "metrics": {
            "baseline_accuracy": baseline_acc,
            "adapter_accuracy": adapter_acc,
            "adapter_gain": (adapter_acc - baseline_acc) if baseline_acc is not None else None,
        },
        "per_question": {
            "baseline_texts": baseline_texts,
            "adapter_texts": adapter_texts,
            "baseline_correct": baseline_ok,
            "adapter_correct": adapter_ok,
        },
        "args": vars(args),
    }
    json_path = exp_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOG.info("Summary written → %s", json_path)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
