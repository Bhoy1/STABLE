# general-knowledge/src/continual/continual_self_edits.py
"""
Incremental self-editing experiment with progressive LoRA merges

This script builds a lower-triangular accuracy matrix for K datapoints
d_0 ... d_{K-1} drawn from a SQuAD-style dataset, repeating the entire
process S times to obtain good estimates of mean and standard-deviation

* Outer loop (S sequences): each repeat draws its own subsequence of
  --n_datapoints items (without replacement, but the full dataset may
  be reused across sequences).
* Inner loop (K steps): at step k we
  1. self-edit on datapoint k (generate implications),
  2. finetune one LoRA adapter on that completion plus the raw
     passage (split_newlines=True),
  3. merge the adapter into the base weights, creating a new base
     for all subsequent steps,
  4. evaluate the freshly merged model on the questions from datapoints
     d_0 ... d_k and collect accuracies.

The final output for each sequence is two (K+1) x K top-row + lower-triangular matrices:
  - A mean-accuracy matrix
  - A standard-deviation matrix

Row 0 corresponds to the not-yet-finetuned base model evaluated on all K
datapoints. Rows 1 through K correspond to evaluations after each
self-edit-and-merge step, with row r containing results on datapoints
d_0 through d_{r-1}.

All artifacts are dumped to --output_dir.
"""
import argparse
import json
import os
import random
import statistics as _stats
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import requests
import torch
import numpy as np
import zmq
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

from ..utils import (
    set_vllm_api_url,
    build_train_sequences,
)
from ..data_generation.make_squad_data import make_prompt

# Silence transformers warning spam inside forked processes
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

###############################################################################
#                               Infra helpers                                 #
###############################################################################

def _banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n", flush=True)

def _spawn_vllm(model: str, host: str, port: int, gpus: str, log_dir: Path, tag: str, lora_rank, max_model_len: int) -> subprocess.Popen:
    """Launch vLLM (LoRA-enabled) and wait for /health."""
    cmd = [
        "vllm",
        "serve",
        model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--enable-lora",
        "--max-lora-rank", str(lora_rank),
        "--max-logprobs", "50",  
        "--trust-remote-code",
    ]
    _banner(f"[vLLM] launching on GPU(s) {gpus} → :{port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{tag}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)

    health = f"http://{host}:{port}/health"
    for _ in range(600):
        if proc.poll() is not None:
            sys.exit(f"[vLLM] crashed (exit {proc.returncode})")
        try:
            if requests.get(health, timeout=1).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    sys.exit("[vLLM] failed to start within timeout")


def _spawn_inner_server(vllm_api: str, model: str, zmq_port: int, gpu: str, log_dir: Path, tag: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "general-knowledge.src.inner.TTT_server",
        "--vllm_api_url",
        vllm_api,
        "--model",
        model,
        "--zmq_port",
        str(zmq_port),
        "--keep_adapter_dir",  # keep the adapter dir for merging later. It will be removed after merging
    ]
    _banner(f"[Inner] launching on GPU {gpu}, ZMQ :{zmq_port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_dir / f"inner_{tag}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    return proc


def _connect_zmq(port: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    return ctx, sock


def _send_round(sock, train_seqs: List[str], questions: List[Dict[str, str]], args, 
                return_logprobs: bool = False):
    sock.send_json(
        {
            "train_sequences": train_seqs,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
            "return_logprobs": return_logprobs,
        }
    )
    return sock.recv_json()


def _send_eval_text_logprobs(sock, questions: List[Dict[str, str]], 
                             texts_to_score: List[str]) -> Dict:
    """Get log probabilities for specific completion texts."""
    sock.send_json({
        "eval_text_logprobs": True,
        "eval_questions": questions,
        "texts_to_score": texts_to_score,
    })
    return sock.recv_json()


###############################################################################
#                          Safety mechanisms                                  #
###############################################################################

def _send_eval_with_adapter(sock, adapter_path: Path, questions: List[Dict[str, str]]) -> Dict:
    """Special request to evaluate with a specific adapter loaded."""
    sock.send_json(
        {
            "eval_only_with_adapter": True,
            "adapter_path": str(adapter_path),
            "eval_questions": questions,
        }
    )
    return sock.recv_json()


def _send_eval_with_adapter_logprobs(sock, adapter_path: Path, questions: List[Dict[str, str]]) -> Dict:
    """Special request to evaluate with adapter and return log probabilities."""
    sock.send_json(
        {
            "eval_only_with_adapter": True,
            "adapter_path": str(adapter_path),
            "eval_questions": questions,
            "return_logprobs": True,
        }
    )
    return sock.recv_json()


def measure_em_drop_with_kl(zmq_sock, anchors: List[Dict], anchor_questions: List[Dict], 
                             adapter_path: Path, args) -> Tuple[float, float]:
    """
    Measure EM accuracy drop and KL divergence on anchor questions.
    Returns: (em_drop, kl_divergence)
    """
    if not anchors or not anchor_questions:
        return 0.0, 0.0
    
    # Get baseline accuracy
    base_response = _send_round(zmq_sock, [], anchor_questions, args, 
                               return_logprobs=True)
    base_correct = base_response["adapter_correct"]
    base_em = sum(base_correct) / len(base_correct) if base_correct else 0.0
    
    # Get adapter accuracy AND generated texts
    adapter_response = _send_eval_with_adapter_logprobs(zmq_sock, adapter_path, 
                                                        anchor_questions)
    adapter_correct = adapter_response["adapter_correct"]
    adapter_em = sum(adapter_correct) / len(adapter_correct) if adapter_correct else 0.0
    adapter_texts = adapter_response["adapter_texts"]
    
    # Calculate EM drop
    em_drop = max(0, base_em - adapter_em)
    
    # Calculate PPO-style KL divergence
    # Get reference model's logprobs on the SAME texts that adapter generated
    base_logprobs_for_adapter_texts = _send_eval_text_logprobs(
        zmq_sock, anchor_questions, adapter_texts
    )
    
    adapter_logprobs = adapter_response.get("logprobs", [])
    base_logprobs = base_logprobs_for_adapter_texts.get("logprobs", [])
    adapter_token_counts = adapter_response.get("token_counts", [])
    
    kl_divergence = 0.0
    if adapter_logprobs and base_logprobs and adapter_token_counts:
        total_kl_nats = 0.0
        total_tokens = 0
        
        for adapter_lp, base_lp, n_tokens in zip(adapter_logprobs, base_logprobs, 
                                                   adapter_token_counts):
            if n_tokens > 0:
                # KL(adapter || base) ≈ log p_adapter - log p_base per token
                kl_nats = adapter_lp - base_lp  # Already sum of log probs
                total_kl_nats += kl_nats
                total_tokens += n_tokens
        
        kl_divergence = (total_kl_nats / max(1, total_tokens)) / np.log(2)
    
    print(f"    [EM+KL] Base EM: {base_em:.3f}, Adapter EM: {adapter_em:.3f}, "
          f"Drop: {em_drop:.3f}, KL: {kl_divergence:.4f} bits/token")
    
    return em_drop, kl_divergence



def measure_bits_increase_with_kl(zmq_sock, anchors: List[Dict], 
                                  anchor_questions: List[Dict], 
                                  adapter_path: Path, args) -> Tuple[float, float]:
    """
    Measure bits increase and KL divergence on anchor questions.
    Returns: (bits_increase, kl_divergence)
    """
    if not anchors or not anchor_questions:
        return 0.0, 0.0
    
    # Get baseline with logprobs
    base_response = _send_round(zmq_sock, [], anchor_questions, args, 
                               return_logprobs=True)
    
    # Get adapter with logprobs AND texts
    adapter_response = _send_eval_with_adapter_logprobs(zmq_sock, adapter_path, 
                                                        anchor_questions)
    
    # Extract data for bits calculation
    base_logprobs = base_response.get("logprobs", [])
    adapter_logprobs = adapter_response.get("logprobs", [])
    base_token_counts = base_response.get("token_counts", [])
    adapter_token_counts = adapter_response.get("token_counts", [])
    
    # Calculate bits increase (existing logic)
    bits_increase = 0.0
    if base_logprobs and adapter_logprobs and base_token_counts and adapter_token_counts:
        ln2 = np.log(2)
        
        if args.bits_mode == "total":
            total_bits_increase = 0.0
            for i in range(len(base_logprobs)):
                base_bits = -base_logprobs[i] / ln2 if base_logprobs[i] > -100 else 100.0
                adapter_bits = -adapter_logprobs[i] / ln2 if adapter_logprobs[i] > -100 else 100.0
                total_bits_increase += max(0, adapter_bits - base_bits)
            bits_increase = total_bits_increase
            
        elif args.bits_mode == "average":
            total_bits_increase = 0.0
            total_tokens = 0
            for i in range(len(base_logprobs)):
                base_tokens = base_token_counts[i]
                adapter_tokens = adapter_token_counts[i]
                base_bits_per_token = (-base_logprobs[i] / ln2) / base_tokens
                adapter_bits_per_token = (-adapter_logprobs[i] / ln2) / adapter_tokens
                increase = max(0, adapter_bits_per_token - base_bits_per_token)
                total_bits_increase += increase * base_tokens
                total_tokens += base_tokens
            bits_increase = total_bits_increase / max(1, total_tokens)
            
    
    # Calculate PPO-style KL divergence
    adapter_texts = adapter_response.get("adapter_texts", [])
    base_logprobs_for_adapter_texts = _send_eval_text_logprobs(
        zmq_sock, anchor_questions, adapter_texts
    )
    
    kl_divergence = 0.0
    base_lp_on_adapter = base_logprobs_for_adapter_texts.get("logprobs", [])
    if adapter_logprobs and base_lp_on_adapter and adapter_token_counts:
        total_kl_nats = 0.0
        total_tokens = 0
        
        for adapter_lp, base_lp, n_tokens in zip(adapter_logprobs, 
                                                   base_lp_on_adapter, 
                                                   adapter_token_counts):
            if n_tokens > 0:
                kl_nats = adapter_lp - base_lp
                total_kl_nats += kl_nats
                total_tokens += n_tokens
        
        kl_divergence = (total_kl_nats / max(1, total_tokens)) / np.log(2)
    
    mode_str = f"Bits-{args.bits_mode.capitalize()}"
    print(f"    [{mode_str}+KL] Bits: {bits_increase:.3f}, "
          f"KL: {kl_divergence:.4f} bits/token")
    
    return bits_increase, kl_divergence


def measure_kl_divergence(zmq_sock, anchors: List[Dict], anchor_questions: List[Dict],
                          adapter_path: Path, args) -> float:
    """
    Measure KL divergence using PPO-style approach (much simpler!).
    Returns: average KL divergence in bits per token
    """
    if not anchors or not anchor_questions:
        return 0.0
    
    print(f"[KL] Computing PPO-style KL divergence")
    
    # Generate with adapter and get its logprobs
    adapter_response = _send_eval_with_adapter_logprobs(zmq_sock, adapter_path, 
                                                        anchor_questions)
    
    # Get base model's logprobs on the SAME generated texts
    adapter_texts = adapter_response.get("adapter_texts", [])
    base_response = _send_eval_text_logprobs(zmq_sock, anchor_questions, adapter_texts)
    
    # Extract data
    adapter_logprobs = adapter_response.get("logprobs", [])
    base_logprobs = base_response.get("logprobs", [])
    adapter_token_counts = adapter_response.get("token_counts", [])

    
    # Calculate KL divergence
    total_kl_nats = 0.0
    total_tokens = 0
    
    for adapter_lp, base_lp, n_tokens in zip(adapter_logprobs, base_logprobs, 
                                               adapter_token_counts):
        if n_tokens > 0:
            # KL(adapter || base) ≈ log p_adapter(x) - log p_base(x)
            kl_nats = adapter_lp - base_lp
            total_kl_nats += kl_nats
            total_tokens += n_tokens
    
    avg_kl_nats = total_kl_nats / max(1, total_tokens)
    avg_kl_bits = avg_kl_nats / np.log(2)
    
    print(f"    [KL] {avg_kl_bits:.4f} bits/token (over {total_tokens} tokens)")
    
    return avg_kl_bits



def scale_lora_adapter(adapter_path: Path, scale: float) -> Path:
    """Scale LoRA weights by a factor and return path to scaled adapter."""
    scaled_path = adapter_path.parent / f"scaled_{scale:.3f}"
    scaled_path.mkdir(exist_ok=True)
    
    # Copy config files
    for config_file in ["adapter_config.json", "adapter_model.json"]:
        src = adapter_path / config_file
        if src.exists():
            shutil.copy(src, scaled_path / config_file)
    
    # Scale the weights
    for weight_file in adapter_path.glob("*.safetensors"):
        weights = load_file(weight_file)
        # Only scale LoRA weights (those with .lora_A or .lora_B in the name)
        scaled_weights = {}
        for k, v in weights.items():
            if "lora_" in k:
                scaled_weights[k] = v * scale
            else:
                scaled_weights[k] = v
        save_file(scaled_weights, scaled_path / weight_file.name)
    
    # If using .bin files instead of safetensors
    for weight_file in adapter_path.glob("*.bin"):
        weights = torch.load(weight_file)
        scaled_weights = {}
        for k, v in weights.items():
            if "lora_" in k:
                scaled_weights[k] = v * scale
            else:
                scaled_weights[k] = v
        torch.save(scaled_weights, scaled_path / weight_file.name)
    
    return scaled_path


def binary_search_scale(zmq_sock, adapter_path: Path, anchors: List[Dict],
                        anchor_questions: List[Dict], budget: float, 
                        metric: str, args) -> Optional[float]:
    """Binary search for LoRA scaling factor that meets budget."""
    left, right = args.clip_min_scale, 1.0
    best_scale = None
    
    for step in range(args.clip_bisect_steps):
        mid = (left + right) / 2
        
        # Scale the adapter
        scaled_adapter_path = scale_lora_adapter(adapter_path, mid)
        
        # Measure forgetting with scaled adapter
        #Using non-Kl variants for EM and bits to save computation as we don't 
        #need secondary metrics for binary search
        if metric == "em":
            forgetting, _= measure_em_drop_with_kl(zmq_sock, anchors, anchor_questions, 
                                        scaled_adapter_path, args)
        elif metric == "bits":
            forgetting, _= measure_bits_increase_with_kl(zmq_sock, anchors, anchor_questions,
                                              scaled_adapter_path, args)
        elif metric == "kl":
            forgetting = measure_kl_divergence(zmq_sock, anchors, anchor_questions,
                                             scaled_adapter_path, args)
        
        print(f"  [Binary search] scale={mid:.3f}, forgetting={forgetting:.4f}, budget={budget:.4f}")
        
        if forgetting <= budget:
            best_scale = mid
            left = mid  # Try larger scale
        else:
            right = mid  # Try smaller scale
        
        # Clean up scaled adapter
        shutil.rmtree(scaled_adapter_path, ignore_errors=True)
    
    return best_scale

###############




###############################################################################
#                             LoRA → merge                                    #
###############################################################################

def _merge_lora(base_path: str, adapter_path: Path, out_dir: Path) -> str:
    """Merge adapter_path into base_path and save to out_dir (returns str)."""
    _banner(f"[Merge] base={base_path} + adapter={adapter_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    torch.cuda.empty_cache()
    print(f"[Merge] saved → {out_dir}\n")
    return str(out_dir)

###############################################################################
#                          One sequence (outer loop)                          #
###############################################################################

def run_one_sequence(seq_idx: int, items: List[Dict[str, Any]], args) -> Tuple[List[List[float]], List[List[float]]]:
    """Run the incremental loop over items and return mean/std matrices."""
    K = len(items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(K)] for _ in range(R)]

    current_model_path = args.model  # evolves after each merge

    anchors = []  # Track datapoints that have been successfully merged
    rejected_count = 0  # Track rejected edits
    safety_log = []  # Track safety decisions

    # -------- 0) Base-model row  (row 0 in mat_vals) ------------------
    base_tag      = f"seq{seq_idx}_base"
    logs_step_dir = Path(args.output_dir) / "logs"

    vllm = _spawn_vllm(current_model_path, "127.0.0.1", args.vllm_port, 
                       args.vllm_gpus, logs_step_dir, base_tag, args.lora_rank, 2048)
    vllm_api = f"http://127.0.0.1:{args.vllm_port}"
    set_vllm_api_url(vllm_api)

    inner  = _spawn_inner_server(vllm_api, current_model_path,
                                 args.zmq_port, args.inner_gpu,
                                 logs_step_dir, base_tag)
    ctx, sock = _connect_zmq(args.zmq_port)

    for i, item in enumerate(items):
        eval_q = [
            {
                "title":    item["title"],
                "context":  item["context"],
                "question": f"Topic: {item['title']}\n{q['question']}",
                "answer":   q["answer"],
            }
            for q in item["questions"]
        ]

        # send with empty train_sequences → no fine-tuning
        rep_out  = _send_round(sock, [], eval_q, args)
        correct  = rep_out["adapter_correct"]
        acc      = sum(correct) / len(correct)
        mat_vals[0][i].append(acc)          # row-0, col-i
        print(f"    [base] d{i}: {acc:.3f}")

    # clean up
    sock.send_json({"cmd": "shutdown"}); sock.recv_json()
    sock.close(); ctx.term()
    inner.terminate(); vllm.terminate()
    vllm.wait()
    torch.cuda.empty_cache()

    # Pre-compute question spans for convenience
    q_spans: List[Tuple[int, int]] = []
    cum = 0
    for it in items:
        n_q = len(it["questions"])
        q_spans.append((cum, cum + n_q))
        cum += n_q
    agg_questions: List[Dict[str, str]] = []

    max_model_len = args.max_tokens + 2048  # for vLLM
    for k, item in enumerate(items):
        print(f"[Seq {seq_idx}] Step {k}/{K-1} - {item['title']}")

        # ---------------- 1) spin up infra --------------------------------
        step_tag = f"seq{seq_idx}_step{k}"
        logs_step_dir = Path(args.output_dir) / "logs"
        vllm = _spawn_vllm(current_model_path, "127.0.0.1", args.vllm_port, args.vllm_gpus, logs_step_dir, step_tag, args.lora_rank, max_model_len)
        vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(vllm_api)
        inner = _spawn_inner_server(vllm_api, current_model_path, args.zmq_port, args.inner_gpu, logs_step_dir, step_tag)
        zmq_ctx, zmq_sock = _connect_zmq(args.zmq_port)

        # ---------------- 2) self-edit completion --------------------------
        prompt = make_prompt(item["title"], item["context"], instruct_model=False, prompt_key="implications")
        comp_resp = requests.post(
            f"{vllm_api}/v1/completions",
            json={
                "model": current_model_path,
                "prompt": [prompt],
                "logprobs": 1,  # Request top-1 logprob for each token
                "echo": True,    # Include prompt in response to get all logprobs
                "n": 1,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            timeout=600,
        )
        comp_resp.raise_for_status()
        completion = comp_resp.json()["choices"][0]["text"].strip()

        train_sequences = build_train_sequences(
            completion or item["context"], item["context"], item["title"], split_newlines=True
        )

        # ---------------- 3) extend evaluation question list --------------
        new_q = [
            {
                "title": item["title"],
                "context": item["context"],
                "question": f"Topic: {item['title']}\n{q['question']}",
                "answer": q["answer"],
            }
            for q in item["questions"]
        ]
        agg_questions.extend(new_q)


        # Prepare anchor questions for safety checks
        anchor_questions = []
        for anchor in anchors:
            for q in anchor["questions"]:
                anchor_questions.append({
                    "title": anchor["title"],
                    "context": anchor["context"],
                    "question": f"Topic: {anchor['title']}\n{q['question']}",
                    "answer": q["answer"],
                })

        

        # ---------------- 4) tune-and-eval  -------------------------------
        rep_out = _send_round(zmq_sock, train_sequences, agg_questions, args)
        correct = rep_out["adapter_correct"]
   

        # ---------------- 5) grab adapter & merge into base ---------------
        adapter_path = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0/final_adapter")
        print("[Merge] adapter path:", adapter_path)
        if not adapter_path.exists():
            print("[!] adapter not found - skipping merge, keeping previous base")
            rejected_count += 1
            safety_log.append({"step": k, "action": "rejected", "reason": "no_adapter"})
        else:
            # ------------ Safety checks before merging --------------
            should_merge = True
            scale_factor = 1.0
            safety_decision = {"step": k, "action": "accept", "scale": 1.0}
            
            if args.safety_mode != "baseline" and anchors:
                # Measure forgetting on unscaled adapter
                if args.forget_metric == "em":
                    em_drop, kl_div = measure_em_drop_with_kl(zmq_sock, anchors, anchor_questions, 
                                                            adapter_path, args)
                    forgetting = em_drop
                    budget = args.em_max_drop
                    safety_decision["em_drop_unscaled"] = em_drop
                    safety_decision["kl_divergence_unscaled"] = kl_div
                    
                elif args.forget_metric == "bits":
                    bits_increase, kl_div = measure_bits_increase_with_kl(zmq_sock, anchors, anchor_questions,
                                                                                adapter_path, args)
                    forgetting = bits_increase
                    if args.bits_mode == "average":
                        budget = args.bits_max_average
                    else:  # total
                        budget = args.bits_max_total
                    safety_decision["bits_increase_unscaled"] = bits_increase
                    safety_decision["bits_mode"] = args.bits_mode
                    safety_decision["kl_divergence_unscaled"] = kl_div
                    
                elif args.forget_metric == "kl":
                    kl_div  = measure_kl_divergence(zmq_sock, anchors, anchor_questions,
                                                        adapter_path, args)
                    forgetting = kl_div
                    budget = args.kl_max_bits
                    safety_decision["kl_divergence_unscaled"] = kl_div

                
                safety_decision["primary_metric"] = args.forget_metric
                safety_decision["forgetting_unscaled"] = forgetting
                safety_decision["budget"] = budget
                
                # Enhanced logging
                if args.forget_metric == "bits":
                    print(f"[Safety] Step {k}: Bits={forgetting:.4f} ({args.bits_mode}), KL={kl_div:.4f} bits/token")
                elif args.forget_metric == "em":
                    print(f"[Safety] Step {k}: EM drop={forgetting:.4f}, KL={kl_div:.4f} bits/token")
                else:  # kl
                    print(f"[Safety] Step {k}: KL={forgetting:.4f} bits/token (primary metric)")
                
                print(f"[Safety] Primary metric ({args.forget_metric}): {forgetting:.4f} vs budget {budget:.4f}")
                
                # Decide on action based on safety mode
                if args.safety_mode == "gate_only":
                    if forgetting > budget:
                        should_merge = False
                        rejected_count += 1
                        safety_decision["action"] = "rejected"
                        safety_decision["reason"] = f"forgetting ({forgetting:.4f}) > budget ({budget:.4f})"
                        print(f"[Safety] REJECTED edit at step {k}")
                    else:
                        # No scaling needed - log the unscaled values as final
                        safety_decision["kl_divergence"] = kl_div
                        safety_decision["forgetting"] = forgetting
                        if args.forget_metric == "em":
                            safety_decision["em_drop"] = em_drop
                        elif args.forget_metric == "bits":
                            safety_decision["bits_increase"] = bits_increase
                
                elif args.safety_mode == "lora_clip":
                    if forgetting > budget:
                        print(f"[Safety] Forgetting exceeds budget, searching for safe scale...")
                        scale_factor = binary_search_scale(
                            zmq_sock, adapter_path, anchors, anchor_questions,
                            budget, args.forget_metric, args
                        )
                        if scale_factor is None or scale_factor < args.clip_min_scale:
                            should_merge = False
                            rejected_count += 1
                            safety_decision["action"] = "rejected"
                            safety_decision["reason"] = "no_safe_scale_found"
                            print(f"[Safety] REJECTED edit at step {k} (couldn't find safe scale)")
                        else:
                            print(f"[Safety] Scaling adapter by {scale_factor:.3f}")
                            adapter_path = scale_lora_adapter(adapter_path, scale_factor)
                            
                            # Re-measure with scaled adapter
                            if args.forget_metric == "em":
                                em_drop, kl_div = measure_em_drop_with_kl(zmq_sock, anchors, anchor_questions, 
                                                                                adapter_path, args)
                                forgetting = em_drop
                                safety_decision["em_drop"] = em_drop
                                safety_decision["kl_divergence"] = kl_div
                            elif args.forget_metric == "bits":
                                bits_increase, kl_div= measure_bits_increase_with_kl(zmq_sock, anchors, anchor_questions,
                                                                                            adapter_path, args)
                                forgetting = bits_increase
                                safety_decision["bits_increase"] = bits_increase
                                safety_decision["kl_divergence"] = kl_div
                            elif args.forget_metric == "kl":
                                kl_div = measure_kl_divergence(zmq_sock, anchors, anchor_questions,
                                                                    adapter_path, args)
                                forgetting = kl_div
                                safety_decision["kl_divergence"] = kl_div
                            
                            safety_decision["action"] = "scaled"
                            safety_decision["scale"] = scale_factor
                            safety_decision["forgetting"] = forgetting  # Post-scaling
                            
                            print(f"[Safety] After scaling: {args.forget_metric}={forgetting:.4f}, KL={kl_div:.4f}")
                    else:
                        # No scaling needed - log the unscaled values as final
                        safety_decision["kl_divergence"] = kl_div
                        safety_decision["forgetting"] = forgetting
                        if args.forget_metric == "em":
                            safety_decision["em_drop"] = em_drop
                        elif args.forget_metric == "bits":
                            safety_decision["bits_increase"] = bits_increase
            
            safety_log.append(safety_decision)
            
            if should_merge:
                merged_dir = Path(args.output_dir) / f"merged_seq{seq_idx}_step{k}"
                prev_model_path = current_model_path
                current_model_path = _merge_lora(current_model_path, adapter_path, merged_dir)
                
                # Clean up previous merge dir if it's not the original base
                if k > 0 and Path(prev_model_path).is_dir() and str(prev_model_path).startswith(str(Path(args.output_dir))):
                    try:
                        shutil.rmtree(prev_model_path)
                        print(f"[Cleanup] removed previous merge dir {prev_model_path}")
                    except Exception as exc:
                        print(f"[Cleanup] failed to remove {prev_model_path}: {exc}")
                
                # Also remove last step of previous sequence, if this is first step of current
                if k == 0 and seq_idx > 0:
                    prev_final_merge = Path(args.output_dir) / f"merged_seq{seq_idx - 1}_step{K - 1}"
                    if prev_final_merge.exists():
                        try:
                            shutil.rmtree(prev_final_merge)
                            print(f"[Cleanup] removed prior sequence's final merge dir {prev_final_merge}")
                        except Exception as exc:
                            print(f"[Cleanup] failed to remove {prev_final_merge}: {exc}")
            else:
                print(f"[Safety] Skipping merge at step {k}, keeping previous model")
        
        # Always add to anchors after evaluation (whether merged or not)
        anchors.append(item)
        
        # Update accuracy matrix (regardless of merge decision)
        for i in range(k + 1):
            s, e = q_spans[i]
            acc = sum(correct[s:e]) / (e - s)
            mat_vals[k+1][i].append(acc)
        print([f"d{i}:{_stats.mean(mat_vals[k+1][i]):.3f}" for i in range(k + 1)])



        # ---------------- 6) graceful shutdown ----------------------------
        try:
            zmq_sock.send_json({"cmd": "shutdown"}); 
            msg = zmq_sock.recv_json()
            print(msg)
        except Exception:
            pass
        zmq_sock.close(); zmq_ctx.term()
        inner.terminate(); vllm.terminate()
        vllm.wait()
        torch.cuda.empty_cache()

    # ---------------- 7) clean up remaining directories ------------------

    # delete the inner-server's "models/tmp_*" folder:
    tmp_dir = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0")
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            print(f"[Cleanup] removed temporary adapter dir {tmp_dir}")
        except Exception as exc:
            print(f"[Cleanup] failed to remove temporary adapter dir {tmp_dir}: {exc}")

# ---------------- 7.5) Measure global drift from original to final ------------------
    global_kl_results = None
    
    # Only measure if we actually did some merges and have anchors
    if current_model_path != args.model and anchors:
        # Rebuild anchor_questions from ALL anchors (not just the last iteration)
        anchor_questions = []
        for anchor in anchors:
            for q in anchor["questions"]:
                anchor_questions.append({
                    "title": anchor["title"],
                    "context": anchor["context"],
                    "question": f"Topic: {anchor['title']}\n{q['question']}",
                    "answer": q["answer"],
                })
        
        print("\n" + "="*60)
        print(f"[Global Drift] Measuring KL divergence from original to final model")
        print(f"[Global Drift] Evaluating on {len(anchor_questions)} anchor questions")
        print("="*60 + "\n")
        
        # Don't delete the final model yet - we need it
        final_model_path = current_model_path
        
        # ---------- Evaluate with FINAL merged model (GENERATE HERE) ----------
        print(f"[Global Drift] Step 1/2: Evaluating final model ({final_model_path})...")
        vllm = _spawn_vllm(final_model_path, "127.0.0.1", args.vllm_port,
                          args.vllm_gpus, logs_step_dir, f"seq{seq_idx}_global_final",
                          args.lora_rank, 2048)
        vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(vllm_api)
        inner = _spawn_inner_server(vllm_api, final_model_path, args.zmq_port,
                                    args.inner_gpu, logs_step_dir, f"seq{seq_idx}_global_final")
        zmq_ctx, zmq_sock = _connect_zmq(args.zmq_port)

        # Generate with FINAL model and get its logprobs
        final_response = _send_round(zmq_sock, [], anchor_questions, args,
                                    return_logprobs=True)
        
        # Get FINAL model's generated texts
        final_texts = final_response.get("adapter_texts", [])
        final_logprobs = final_response.get("logprobs", [])
        final_token_counts = final_response.get("token_counts", [])

        # Cleanup final model evaluation
        try:
            zmq_sock.send_json({"cmd": "shutdown"})
            zmq_sock.recv_json()
        except Exception:
            pass
        zmq_sock.close()
        zmq_ctx.term()
        inner.terminate()
        vllm.terminate()
        vllm.wait()
        torch.cuda.empty_cache()
        
        # ---------- Evaluate with ORIGINAL model (SCORE final's text) ----------
        print(f"[Global Drift] Step 2/2: Scoring final's outputs under original model ({args.model})...")
        vllm = _spawn_vllm(args.model, "127.0.0.1", args.vllm_port, 
                          args.vllm_gpus, logs_step_dir, f"seq{seq_idx}_global_orig", 
                          args.lora_rank, 2048)
        vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(vllm_api)
        inner = _spawn_inner_server(vllm_api, args.model, args.zmq_port, 
                                    args.inner_gpu, logs_step_dir, f"seq{seq_idx}_global_orig")
        zmq_ctx, zmq_sock = _connect_zmq(args.zmq_port)
        
        # Score FINAL's texts under ORIGINAL model
        original_logprobs_for_final_texts = _send_eval_text_logprobs(
            zmq_sock, anchor_questions, final_texts
        )
        
        original_logprobs = original_logprobs_for_final_texts.get("logprobs", [])
        
        # Cleanup original model evaluation
        try:
            zmq_sock.send_json({"cmd": "shutdown"})
            zmq_sock.recv_json()
        except Exception:
            pass
        zmq_sock.close()
        zmq_ctx.term()
        inner.terminate()
        vllm.terminate()
        vllm.wait()
        torch.cuda.empty_cache()

        # ---------- Calculate global KL divergence ----------
        if final_logprobs and original_logprobs and final_token_counts:
            total_kl_nats = 0.0
            total_tokens = 0
            
            for final_lp, orig_lp, n_tokens in zip(final_logprobs, original_logprobs, 
                                                    final_token_counts):
                if n_tokens > 0:
                    # KL(final || original) ≈ log p_final(x) - log p_original(x)
                    # where x is sampled from final model
                    kl_nats = final_lp - orig_lp
                    total_kl_nats += kl_nats
                    total_tokens += n_tokens
            
            global_kl = (total_kl_nats / max(1, total_tokens)) / np.log(2)
            
            global_kl_results = {
                "global_kl_bits_per_token": global_kl,
                "total_tokens_measured": total_tokens,
                "num_anchor_questions": len(anchor_questions),
            }
            
            print(f"\n[Global Drift] KL(final || original) = {global_kl:.4f} bits/token")
            print(f"[Global Drift] Measured over {total_tokens} tokens")
            print(f"[Global Drift] (Final model's outputs scored under both models)")
            print("="*60 + "\n")
        else:
            print("[Global Drift] WARNING: Could not retrieve logprobs, skipping KL calculation")
        
        # Now clean up the final model directory
        if Path(final_model_path).is_dir() and str(final_model_path).startswith(str(Path(args.output_dir))):
            try:
                shutil.rmtree(final_model_path)
                print(f"[Cleanup] removed final merge dir {final_model_path}")
            except Exception as exc:
                print(f"[Cleanup] failed to remove final merge dir {final_model_path}: {exc}")
    else:
        print(f"[Global Drift] Skipping (no merges performed or no anchors)")


    # Add summary statistics
    if args.safety_mode != "baseline":
        print("\n" + "="*60)
        print(f"[Safety Report] Sequence {seq_idx} Complete")
        print(f"  Mode: {args.safety_mode}, Metric: {args.forget_metric}")
        print(f"  Total steps: {K}")
        print(f"  Accepted edits: {K - rejected_count}")
        print(f"  Rejected edits: {rejected_count}")
        if rejected_count > 0:
            print(f"  Rejection rate: {rejected_count/K*100:.1f}%")
        print("="*60 + "\n")
    

    # Write safety log
    print(f"\n[Safety Summary] Mode: {args.safety_mode}, Metric: {args.forget_metric}")
    print(f"[Safety Summary] Rejected {rejected_count}/{K} edits")
    
    # Save safety log
    if args.safety_mode != "baseline":
        safety_log_data = {
            "config": {
                "safety_mode": args.safety_mode,
                "forget_metric": args.forget_metric,
                "total_steps": K,
            },
            "steps": safety_log,
            "summary": {
                "accepted_edits": K - rejected_count,
                "rejected_edits": rejected_count,
                "rejection_rate": rejected_count / K if K > 0 else 0.0,
            },
            "global_drift": global_kl_results
        }
        safety_log_path = Path(args.output_dir) / f"safety_log_seq{seq_idx}.json"
        with safety_log_path.open("w") as f:
            json.dump(safety_log_data, f, indent=2)
        print(f"[Safety] Log with global drift saved to {safety_log_path}")

    # ---------------- 8) aggregate mean/std over reps --------------------
    mean_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    std_mat: List[List[float]]  = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = mat_vals[r][i]
            if vals:                       # base row has all K cells,
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i]  = _stats.stdev(vals) if len(vals) > 1 else 0.0
    print("mean matrix:\n", json.dumps(mean_mat, indent=2))
    print("std matrix:\n", json.dumps(std_mat, indent=2))
    print("finished - matrices computed\n")
    return mean_mat, std_mat

###############################################################################
#                                   CLI                                       #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser()

    # Dataset & sampling
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_sequences", type=int, default=1, help="How many independent sequences")
    p.add_argument("--n_datapoints", type=int, default=8, help="Datapoints per sequence")

    # Model & placement
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--gpus", default="0,1", help="Comma-separated list → first for vLLM, second for inner")
    p.add_argument("--vllm_port", type=int, default=8001)
    p.add_argument("--zmq_port", type=int, default=5555)

    # Generation params for self-edit completion
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=8192)

    # Inner loop hyperparams (pass-through)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")

    # Safety mechanism arguments
    p.add_argument("--safety_mode", choices=["baseline", "gate_only", "lora_clip"],
                   default="baseline", help="Safety mode for forgetting prevention")
    p.add_argument("--forget_metric", choices=["em", "bits", "kl"],
                   default="em", help="Metric to measure forgetting")
    
    # Thresholds for each metric
    p.add_argument("--em_max_drop", type=float, default=0.0,
                   help="Max allowed EM drop on anchors (for forget_metric=em)")
    p.add_argument("--bits_max_total", type=float, default=1.0,
                   help="Max total bits increase on anchors (for forget_metric=bits)")
    p.add_argument("--bits_mode", choices=["total", "average"], 
                default="total", help="How to measure bits increase")
    p.add_argument("--bits_max_average", type=float, default=2.0,
                help="Max bits/token for average mode")
    p.add_argument("--kl_max_bits", type=float, default=0.3,
                   help="Max mean KL divergence in bits/token (for forget_metric=kl)")
    
    # LoRA clipping parameters
    p.add_argument("--clip_bisect_steps", type=int, default=7,
                   help="Binary search steps for LoRA scaling")
    p.add_argument("--clip_min_scale", type=float, default=0.1,
                   help="Minimum scale factor for LoRA clipping")

    p.add_argument("--output_dir", default="general-knowledge/results/continual_self_edits")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

###############################################################################
#                             Top-level driver                                #
###############################################################################

def main():
    args = parse_args()
    _banner("[Args] " + json.dumps(vars(args), indent=2))
    random.seed(args.seed)

    gpus = args.gpus.split(",")
    if len(gpus) < 2:
        sys.exit("[!] --gpus must list at least two IDs (vLLM,inner)")
    args.vllm_gpus, args.inner_gpu = gpus[0], gpus[1]

    full_data: List[Dict[str, Any]] = json.load(Path(args.dataset).open())
    if args.n_datapoints > len(full_data):
        sys.exit("[!] n_datapoints exceeds dataset size")

    seq_matrices_mean: List[List[List[float]]] = []
    seq_matrices_std:  List[List[List[float]]] = []

    for seq_idx in range(args.n_sequences):
        items = random.sample(full_data, args.n_datapoints)
        mean_mat, std_mat = run_one_sequence(seq_idx, items, args)
        seq_matrices_mean.append(mean_mat)
        seq_matrices_std.append(std_mat)

    # -------- aggregate across sequences (simple arithmetic mean) --------
    K = args.n_datapoints
    R = K + 1
    agg_mean = [[0.0] * K for _ in range(R)]
    agg_std  = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = [seq_matrices_mean[s][r][i] for s in range(args.n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "mean_over_sequences": agg_mean,
            "std_over_sequences": agg_std,
            "n_sequences": args.n_sequences,
            "n_datapoints": args.n_datapoints,
            "dataset": args.dataset,
            "base_model": args.model,
        },
        (out_dir / f"summary_{int(time.time())}.json").open("w"),
        indent=2,
    )
    print("\nfinished - summary written to", out_dir)


if __name__ == "__main__":
    main()
