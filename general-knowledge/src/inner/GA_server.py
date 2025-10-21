# general-knowledge/src/inner/GA_server.py
"""
Inner-loop Generative-Adapter (GA) server used by SEAL's outer-loop drivers
(`query_server.py`, `CPT.py`, `continual_self_edits.py`) to generate a
temporary LoRA adapter from the article passage/context (NOT completions),
and immediately evaluate it on corresponding SQuAD questions.

The server is stateless across requests: every JSON message describes a complete
round consisting of
1. a mini-dataset of train_sequences (ignored here; GA uses passage context),
2. a list of eval_questions (for accuracy measurement),
3. hyperparameters controlling generation and GA details.

Running this requires first installing https://github.com/chentong0/generative-adapter in editable mode (pip install -e).
"""
import argparse, gc, logging, time
from typing import Dict, List, Any, Tuple

import torch
import zmq
import random, numpy as np, time as _time

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

# Monkey patch PEFT registries for FastLoRA
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

# ---------------------------  CONFIG & LOGGING  ----------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger()


# ---------------------------  HELPERS  -------------------------------- #
def _chat_generate_baseline(
    model, tokenizer, question_text: str, max_new_tokens: int, temperature: float, top_p: float, stop: List[str]
) -> str:
    """
    Baseline generation w/o GA weights using standard HF generate.
    Uses chat template when available; falls back to plain text.
    """
    model.eval()
    with torch.no_grad():
        text = default_conditional_generate(
            model,
            tokenizer,
            input_text=question_text,
            use_chat=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ) or ""

        # apply simple string stops
        for s in stop or []:
            if s and s in text:
                text = text.split(s, 1)[0]
                break
        return text.strip()


def _grade(questions: List[Dict[str, str]], preds: List[str]) -> Tuple[float, List[bool]]:
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
    return acc, verdicts


def _answers_with_ga(
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
        # Use FastLoRA GA conditional generation with provided weights
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
        )
        preds.append((out or "").strip())
    return preds


# ---------------------------  MAIN LOOP  ------------------------------ #
def main():
    LOG.info("Starting GA server...")
    p = argparse.ArgumentParser()
    p.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port to listen on")

    # Model / tokenizer (Generative Adapter checkpoint)
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

    # Evaluation sampling
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=1.0)
    p.add_argument("--eval_max_tokens", type=int, default=64)
    p.add_argument("--stop", default="\\n", help="Comma-free string used as stop; default single newline.")

    args = p.parse_args()

    # Dtype resolve
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ----- Monkey patch -------
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
    def patched_set_peft_model_state_dict(*args, **kwargs):
        valid_kwargs = ['adapter_name', 'ignore_mismatched_sizes']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        return orig_set_peft_model_state_dict(*args, **filtered_kwargs)
    peft.peft_model.set_peft_model_state_dict = patched_set_peft_model_state_dict
    # --------------------------

    # ---------------- Load GA model & tokenizer ----------------------- #
    LOG.info("Loading GA checkpoint: %s", args.model)


    # Load the PEFT configuration from the given pretrained model directory.
    peft_config = PeftConfig.from_pretrained(args.model)

    # Get the base model path from the configuration.
    base_model_path = peft_config.base_model_name_or_path

    # Ensure that the base model path is available.
    assert base_model_path is not None, "base_model_name_or_path should not be None"

    # This helper handles both the "new" PEFT path and the fallback path in the repo.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model = FastLoraModelForCausalLM.from_pretrained(
        base_model,
        args.model,
        adapter_name='default',
        is_trainable=False,
        config=peft_config,
    )
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- ZMQ REP socket --------------------------------- #
    ctx, sock = zmq.Context(), None
    try:
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{args.zmq_port}")
        LOG.info("ZMQ listening at tcp://*:%d", args.zmq_port)
        step = 0
        while True:
            LOG.info("Waiting for request...")
            msg = sock.recv_json()
            LOG.info("Received request keys: %s", list(msg.keys()))

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "bye"})
                break

            recv_start = time.time()
            try:
                # Seed for determinism per step
                seed = (int(_time.time() * 1e6) + step) & 0xFFFFFFFF
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                LOG.info("Step %d  using seed %d", step, seed)

                # Inputs
                questions: List[Dict[str, str]] = msg.get("eval_questions", [])

                # Sampling config
                temperature = float(args.eval_temperature)
                top_p = float(args.eval_top_p)
                max_new_tokens = int(args.eval_max_tokens)
                stop = [args.stop.encode("utf-8").decode("unicode_escape")] if args.stop else []

                # ---------------------- Baseline ----------------------- #
                base_texts: List[str] = []
                LOG.info("questions: %s", questions)
                for q in questions:
                    LOG.info("Generating baseline for question: %s", q["question"])
                    base_texts.append(
                        _chat_generate_baseline(
                            model=model,
                            tokenizer=tokenizer,
                            question_text=q["question"],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop,
                        )
                    )
                base_acc, base_ok = _grade(questions, base_texts)

                if not questions:
                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy": round(base_acc, 4),
                        "adapter_gain": 0.0,
                        "baseline_texts": base_texts,
                        "adapter_texts": base_texts,
                        "baseline_correct": base_ok,
                        "adapter_correct": base_ok,
                        "gains": [0] * len(base_ok),
                    }
                    sock.send_json(reply)
                    LOG.info("Step %d  BASE-ONLY  acc %.3f  (%.2fs)", step, base_acc, time.time() - recv_start)
                    step += 1
                    continue

                # ------------- Build GA adapter from PASSAGE ------------ #
                # Use the shared passage/context for the article (from any question)
                context_text = questions[0].get("context", "")
                title_text = questions[0].get("title", "")
                prompt_prefix = f"{title_text}\n{context_text}".strip() or context_text.strip()

                LOG.info("Generating GA weights (merge=%s, win=%d)...",
                         args.merge_strategy, args.window_size)

                lora_weights = fastlora_generate_adaptor(
                    model,
                    tokenizer,
                    prompt_prefix,
                    merge_strategy=args.merge_strategy,
                    max_window_size=int(args.window_size),
                )

                # ------------- Evaluation with GA adapter --------------- #
                adapter_texts = _answers_with_ga(
                    model=model,
                    tokenizer=tokenizer,
                    lora_weights=lora_weights,
                    questions=questions,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                adapter_acc, adapter_ok = _grade(questions, adapter_texts)

                gains = [
                    (1 if a and not b else (-1 if b and not a else 0))
                    for b, a in zip(base_ok, adapter_ok)
                ]

                # Cleanup GPU transient state
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                reply = {
                    "baseline_accuracy": round(base_acc, 4),
                    "adapter_accuracy": round(adapter_acc, 4),
                    "adapter_gain": round(adapter_acc - base_acc, 4),
                    "baseline_texts": base_texts,
                    "adapter_texts": adapter_texts,
                    "baseline_correct": base_ok,
                    "adapter_correct": adapter_ok,
                    "gains": gains,
                }
                LOG.info(
                    "Step %d  base %.3f  adapter %.3f  Δ %.3f  (%.2fs)",
                    step,
                    base_acc,
                    adapter_acc,
                    adapter_acc - base_acc,
                    time.time() - recv_start,
                )

            except Exception as e:
                LOG.exception("Error processing request.")
                reply = {"error": f"{type(e).__name__}: {e}"}
            finally:
                LOG.info("Sending reply...")
                sock.send_json(reply)
                LOG.info("Reply sent, step %d complete.", step)
                step += 1
    finally:
        if sock:
            sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
