# general-knowledge/src/inner/TTT_server.py
"""
Inner-loop Test-Time Training (TTT) server used by SEAL's outer-loop drivers
(`query_server.py`, `CPT.py`, `continual_self_edits.py`) to rapidly fine-tune 
a temporary LoRA adapter on a handful of synthetic sequences and immediately 
evaluate it on corresponding SQuAD questions, without the sequences in context.

The server is stateless across requests: every JSON message describes a complete round consisting of
1. a mini-dataset of train_sequences (for LoRA fine-tuning),
2. a list of eval_questions (for accuracy measurement), and
3. hyperparameters controlling both steps.

It then replies with baseline-vs-adapter accuracies, generated answers, and per-question booleans indicating correctness.

JSON schema
    Request -->
    {
        "train_sequences": [str],
        "eval_questions":  [{title, context, question, answer}],
        "lora_rank": int,
        ...
    }
    Response <--
    {
        "baseline_accuracy": float,
        "adapter_accuracy":  float,
        "adapter_gain":      float,
        ...
    }
"""
import argparse, gc, logging, os, shutil, time
from pathlib import Path
from typing import Dict, List, Any
import torch
import requests
import zmq
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import random, numpy as np, torch, time as _time
from ..utils import (
    set_vllm_api_url,
    load_adapter,
    unload_adapter,
    generate,
    format_answer_prompts,
    format_grade_prompts,
    grade_with_gpt4,
    extract_final_answer,
    score_proxy_with_gpt4,
)

# ---------------------------  CONFIG & LOGGING  ----------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger()


def accuracy_and_texts(
    questions: List[Dict[str, str]],
    answer_model_ref: str,
    sampling: Dict[str, Any],
    stop_ids: List[int],
    instruct_model: bool,
    chain_of_thought: bool = False,
    return_logprobs: bool = False,
) -> tuple:

    # Request log probabilities from the generation API if needed
    if return_logprobs:
        sampling = {**sampling, "logprobs": 1}
    
    # Generate answers for all questions
    ans_out = generate(
        format_answer_prompts(questions, instruct_model=instruct_model, chain_of_thought=chain_of_thought), 
        answer_model_ref, 
        sampling, 
        stop_ids
    ) or []

    
    # Extract prediction texts from generation outputs
    preds = [o.get("text", "") for o in ans_out]
    
    # Initialize return values
    sequence_logprobs = []
    sequence_token_counts = []

    
    if return_logprobs:
        for output in ans_out:
            if "logprobs" in output and output["logprobs"]:
                logprobs_data = output["logprobs"]
                
                # For both bits and KL: calculate sequence probability
                token_logprobs = logprobs_data.get("token_logprobs", [])
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]
                total_logprob = sum(valid_logprobs) if valid_logprobs else -100.0
                sequence_logprobs.append(total_logprob)
                sequence_token_counts.append(len(valid_logprobs))
                

    # Extract final answers from chain-of-thought reasoning if needed
    if chain_of_thought:
        preds = [extract_final_answer(p) for p in preds]
    
    #Grade the answers
    verdicts: List[bool] = [False] * len(preds)
    q_sub, p_sub, idx_sub = [], [], []

    for i, (q, p) in enumerate(zip(questions, preds)):
        if p.strip():
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)

    if q_sub:
        grade_prompts = format_grade_prompts(q_sub, p_sub)
        graded = grade_with_gpt4(grade_prompts)
        for i, v in zip(idx_sub, graded):
            verdicts[i] = v
    
    # Calculate accuracy
    acc = sum(verdicts) / len(questions) if questions else 0.0
    
    # Return log probabilities if requested
    if return_logprobs:
        return acc, preds, verdicts, sequence_logprobs, sequence_token_counts

    return acc, preds, verdicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port to listen on")
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using Qwen Instruct model")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Max training seq len")
    p.add_argument("--eval_temperature", type=float, default=0.0, help="Eval sampling temperature")
    p.add_argument("--eval_top_p", type=float, default=1.0, help="Eval nucleus sampling (top-p)")
    p.add_argument("--eval_max_tokens", type=int, default=64, help="Eval max tokens to generate")
    p.add_argument("--keep_adapter_dir", action="store_true",
                   help="Skip tmp-dir deletion so outer driver can reuse the LoRA. This causes high disk usage and is only used in continual_self_edits.py or for debugging.")
    args = p.parse_args()

    # initialize vLLM API
    set_vllm_api_url(args.vllm_api_url)

    LOG.info("Loading base model %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.instruct_model:
        stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    else:
        stop_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    # ---------- ZMQ REP socket ---------------------------------------- #
    ctx, sock = zmq.Context(), None
    try:
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{args.zmq_port}")
        LOG.info("ZMQ listening at tcp://*:%d", args.zmq_port)
        step = 0
        while True:
            LOG.info("Waiting for request...")
            msg = sock.recv_json()
            LOG.info("Received request: %s", msg)

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "bye"})  # reply
                break  # exit the while-loop

            if msg.get("eval_text_logprobs"):
                try:
                    questions = msg.get("eval_questions", [])
                    texts_to_score = msg.get("texts_to_score", [])
                    
                    LOG.info(f"Scoring {len(texts_to_score)} provided texts")
                    
                    # Use vLLM's /v1/completions with echo=True and max_tokens=0
                    # to get logprobs for the provided text
                    import requests
                    logprobs_list = []
                    token_counts_list = []
                    
                    for question, text in zip(questions, texts_to_score):
                        # Format the prompt + completion
                        prompt = format_answer_prompts([question], instruct_model=args.instruct_model)[0]
                        full_text = prompt + text
                        
                        # Request logprobs via vLLM
                        resp = requests.post(
                            f"{args.vllm_api_url}/v1/completions",
                            json={
                                "model": args.model,
                                "prompt": full_text,
                                "max_tokens": 0,  # Don't generate, just score
                                "echo": True,     # Return logprobs for input
                                "logprobs": 1,
                            },
                            timeout=60,
                        )
                        resp.raise_for_status()
                        result = resp.json()["choices"][0]
                        
                        # Extract logprobs for the completion part only
                        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                        logprobs_data = result.get("logprobs", {})
                        token_logprobs = logprobs_data.get("token_logprobs", [])
                        
                        # Skip prompt tokens, only sum completion tokens
                        completion_logprobs = token_logprobs[prompt_tokens:] if len(token_logprobs) > prompt_tokens else token_logprobs
                        
                        # Sum log probs for the completion
                        total_logprob = sum(lp for lp in completion_logprobs if lp is not None)
                        logprobs_list.append(total_logprob)
                        token_counts_list.append(len(completion_logprobs))
                    
                    reply = {
                        "logprobs": logprobs_list,
                        "token_counts": token_counts_list,
                    }
                    
                    LOG.info(f"Scored {len(texts_to_score)} texts successfully")
                    
                except Exception as e:
                    LOG.exception("Error scoring provided texts")
                    reply = {"error": f"{type(e).__name__}: {e}"}
                
                sock.send_json(reply)
                step += 1
                continue  

            if msg.get("eval_only_with_adapter"):
                try:
                    adapter_path = msg.get("adapter_path")
                    questions = msg.get("eval_questions", [])
                    return_logprobs = msg.get("return_logprobs", False)  
                    chain_of_thought = msg.get("chain_of_thought", False)

                    
                    LOG.info(f"Eval-only with adapter: {adapter_path}, logprobs={return_logprobs}")  
                    
                    # Load the adapter temporarily
                    adapter_name = f"temp_eval_{step}"
                    load_adapter(adapter_path, adapter_name)
                    LOG.debug(f"Loaded adapter as {adapter_name}")  
                    
                    # Evaluate with the adapter
                    sampling_cfg = {
                        "n": 1,
                        "temperature": args.eval_temperature,
                        "top_p": args.eval_top_p,
                        "max_tokens": args.eval_max_tokens,
                    }
                    
                    result = accuracy_and_texts(
                        questions,
                        answer_model_ref=adapter_name,
                        sampling=sampling_cfg,
                        stop_ids=stop_ids,
                        instruct_model=args.instruct_model,
                        chain_of_thought=chain_of_thought,
                        return_logprobs=return_logprobs,
                    )
                    
                    # Unload the adapter
                    unload_adapter(adapter_name)
                    LOG.debug(f"Unloaded adapter {adapter_name}")  
                    
                    if return_logprobs:
                        adapter_acc, adapter_texts, adapter_ok, logprobs, token_counts = result
                        reply = {
                            "adapter_accuracy": round(adapter_acc, 4),
                            "adapter_texts": adapter_texts,
                            "adapter_correct": adapter_ok,
                            "logprobs": logprobs,
                            "token_counts": token_counts,
                        }
                    else:
                        adapter_acc, adapter_texts, adapter_ok = result
                        reply = {
                            "adapter_accuracy": round(adapter_acc, 4),
                            "adapter_texts": adapter_texts,
                            "adapter_correct": adapter_ok,
                        }
                        LOG.info(f"Eval-only result: acc={adapter_acc:.3f}")  
                    
                except Exception as e:
                    LOG.exception("Error during eval-only with adapter.")  
                    reply = {"error": f"{type(e).__name__}: {e}", "adapter_correct": []}

                sock.send_json(reply)
                LOG.info("Reply sent for eval-only request")  
                step += 1
                continue  


            recv_start = time.time()
            try:
                LOG.debug("RX %d %s", step, msg.keys())
                seed = (int(_time.time() * 1e6) + step) & 0xFFFFFFFF
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                LOG.info("Step %d  using seed %d", step, seed)

                train_sequences = msg.get("train_sequences")
                questions = msg.get("eval_questions", [])
                lora_rank = msg.get("lora_rank", 32)
                lora_alpha = msg.get("lora_alpha", 64)
                lora_dropout = msg.get("lora_dropout", 0)
                finetune_epochs = msg.get("finetune_epochs", 10)
                finetune_lr = msg.get("finetune_lr", 1e-3)
                batch_size = msg.get("batch_size", 1)
                gradient_accumulation_steps = msg.get("gradient_accumulation_steps", 1)
                end_mask_substring = msg.get("end_mask_substring")
                baseline_eval = bool(msg.get("baseline_eval", True))
                chain_of_thought = bool(msg.get("chain_of_thought", False))
                return_logprobs = bool(msg.get("return_logprobs", False)) 
                reward_mode = msg.get("reward_mode", "ttt")  
                completion_raw = msg.get("comp_raw", "")

                sampling_cfg = {
                    "n": 1,
                    "temperature": args.eval_temperature,
                    "top_p": args.eval_top_p,
                    "max_tokens": args.eval_max_tokens,
                }

                title = ""
                article_context = ""
                if questions:
                    try:
                        title = questions[0].get("title", "") or ""
                        article_context = questions[0].get("context", "") or ""
                    except Exception:
                        pass

                if reward_mode in ("proxy", "both"):
                    try:
                        proxy_scores = score_proxy_with_gpt4(title=title, context=article_context, completion=completion_raw)
                    except Exception:
                        proxy_scores = {"length": 1, "diversity": 1, "quality": 1, "correctness": 1, "final": 4}

                if reward_mode == "proxy":
                    reply = {
                        "baseline_accuracy": 0.0,
                        "adapter_accuracy": 0.0,
                        "adapter_gain": 0.0,
                        "baseline_texts": [""] * len(questions),
                        "adapter_texts": [""] * len(questions),
                        "baseline_correct": [False] * len(questions),
                        "adapter_correct": [False] * len(questions),
                        "gains": [0] * len(questions),
                        "proxy_scores": proxy_scores,
                    }
                    LOG.info("Proxy scores %s", proxy_scores)
                    continue

                # ---------- baseline ------------------------------------------------ #
                if baseline_eval:
                    result = accuracy_and_texts(
                        questions,
                        answer_model_ref=args.model,
                        sampling=sampling_cfg,
                        stop_ids=stop_ids,
                        instruct_model=args.instruct_model,
                        chain_of_thought=chain_of_thought,
                        return_logprobs=return_logprobs,
                    )
                    # Unpack based on whether logprobs were requested
                    if return_logprobs:
                        base_acc, base_texts, base_ok, base_logprobs, base_token_counts = result
                    else:
                        base_acc, base_texts, base_ok = result
                        base_logprobs = None
                        base_token_counts = None
                else:
                    base_acc, base_texts, base_ok = 0.0, [""] * len(questions), [False] * len(questions)
                    base_logprobs = None
                

                if not train_sequences:
                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy": round(base_acc, 4),
                        "adapter_gain": 0.0,
                        "baseline_texts": base_texts,
                        "adapter_texts": base_texts,
                        "baseline_correct": base_ok,
                        "adapter_correct": base_ok,
                        "gains": [0]*len(base_ok),
                    }

                    if return_logprobs and base_logprobs is not None:
                        reply["logprobs"] = base_logprobs
                        reply["token_counts"] = base_token_counts
                    

                    LOG.info("Step %d  BASE-ONLY  acc %.3f  (%.2fs)", step, base_acc, time.time()-recv_start)
                    continue

                # ---------- prepare LoRA fine-tune dataset -------------------------- #
                tmp_tag = f"inner_TTT_{step}"
                tmp_dir = Path(f"models/tmp_{args.zmq_port}_{tmp_tag}")
                os.makedirs(tmp_dir, exist_ok=True)

                rows = []
                sub_ids = (
                    tokenizer.encode(end_mask_substring, add_special_tokens=False)
                    if end_mask_substring else []
                )

                for idx, seq in enumerate(train_sequences):
                    tok = tokenizer(
                        seq,
                        truncation=True,
                        max_length=args.max_seq_length,
                        padding="max_length",
                    )
                    labels = tok["input_ids"].copy()
                    if sub_ids:
                        M = len(sub_ids)
                        for i in range(len(labels) - M + 1):
                            if labels[i : i + M] == sub_ids:
                                for j in range(i + M):
                                    labels[j] = -100
                                # ---------- DEBUG LOG (first 5 only) ---------------
                                if idx < 5:
                                    # insert a visual marker after the masked span
                                    marker_pos = tokenizer.decode(tok["input_ids"][: i + M])
                                    debug_str = seq.replace(
                                        marker_pos,
                                        marker_pos + "<<<MASK_END>>>",
                                        1
                                    )
                                    LOG.info("TRAIN[%d] %s", idx, debug_str)
                                # ---------------------------------------------------
                                break
                    if idx < 3 and not sub_ids:  # no masking substring given; default
                        LOG.info("TRAIN[%d] %s", idx, seq)

                    rows.append(
                        {
                            "input_ids": tok["input_ids"],
                            "attention_mask": tok["attention_mask"],
                            "labels": labels,
                        }
                    )

                ds = HFDataset.from_list(rows)
                collator = DataCollatorWithPadding(tokenizer)

                lora_cfg = LoraConfig(
                    r=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, task_type="CAUSAL_LM"
                )
                lora_model = get_peft_model(base_model, lora_cfg)

                trainer = Trainer(
                    model=lora_model,
                    args=TrainingArguments(
                        output_dir=str(tmp_dir),
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        num_train_epochs=finetune_epochs,
                        learning_rate=finetune_lr,
                        logging_steps=1, save_strategy="no", report_to="none",
                        remove_unused_columns=False, fp16=False,
                        bf16=torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported(),
                        seed=seed,
                    ),
                    train_dataset=ds,
                    data_collator=collator,
                )
                trainer.train()
                adapter_path = tmp_dir / "final_adapter"
                lora_model.save_pretrained(str(adapter_path))

                # ---------- evaluation with adapter ------------------------------- #
                adapter_name = tmp_tag
                load_adapter(str(adapter_path), adapter_name)

                result = accuracy_and_texts(
                    questions,
                    answer_model_ref=adapter_name,
                    sampling=sampling_cfg,
                    stop_ids=stop_ids,
                    instruct_model=args.instruct_model,
                    chain_of_thought=chain_of_thought,
                    return_logprobs=return_logprobs, 
                )

                # Unpack based on whether logprobs were requested
                if return_logprobs:
                    adapter_acc, adapter_texts, adapter_ok, adapter_logprobs, adapter_token_counts = result
                else:
                    adapter_acc, adapter_texts, adapter_ok = result
                    adapter_logprobs = None
                    adapter_token_counts = None

                gains = [
                    1  if a and not b else
                    -1 if b and not a else
                    0
                    for b, a in zip(base_ok, adapter_ok)
                ]

                unload_adapter(adapter_name)
                if not args.keep_adapter_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                gc.collect();  torch.cuda.empty_cache()

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

                if return_logprobs:
                    if base_logprobs is not None:
                        reply["baseline_logprobs"] = base_logprobs
                        reply["baseline_token_counts"] = base_token_counts
                    if adapter_logprobs is not None:
                        reply["adapter_logprobs"] = adapter_logprobs  
                        reply["adapter_token_counts"] = adapter_token_counts



                if reward_mode in ("proxy", "both"):
                    reply["proxy_scores"] = proxy_scores
                    LOG.info("Step %d  Proxy scores: %s", step, proxy_scores)
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
