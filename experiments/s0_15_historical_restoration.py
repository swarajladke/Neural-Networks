#!/usr/bin/env python3
"""
experiments/s0_15_historical_restoration.py -- Bounded Historical Runtime Restoration Test

Directive S0-15 Follow-Up:
Executes an unchanged historical G0c test under exact reference code and pinned inputs.
Evaluates whether a restored or partially restored runtime environment reproduces:
- Gate G0c retention: 30/50
- Gate G0c terminal weight SHA: 578ee1cb8557bda2a513572200300e02b7361f28ca8f1a49a6e4720287271aac

Protocol Governance (AGENTS.md):
- Section 0: Prime Directive (mechanisms over promises, pinned provenance).
- Section 2.1-2.4: Gate G0c remains immutable.
- Section 3.7: Raw per-unit outcome vectors serialized.
- Section 5.1-5.4: Explicit environment fingerprints and execution mode assertions.
- Section 7.1: Strictly under 600 lines.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data import sample_200_facts
from experiments.s0_14_audit import (
    load_and_assert_pinned_cache,
    run_g0_bit_reproduction,
    run_audit_seed
)

FACTS_FILE = REPO_ROOT / "b1_facts.json"
CACHE_DIR = str(REPO_ROOT / "experiments" / "cache")
OUTPUT_FILE = REPO_ROOT / "experiments" / "results" / "s0_15_restoration_audit.json"

HISTORICAL_REF_RETENTION = 30
HISTORICAL_REF_WEIGHT_SHA = "578ee1cb8557bda2a513572200300e02b7361f28ca8f1a49a6e4720287271aac"
SESSION_BUDGET_CEILING = 3600.0


def compute_tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def compute_full_model_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(" DIRECTIVE S0-15 FOLLOW-UP: BOUNDED HISTORICAL RUNTIME RESTORATION TEST")
    print("=" * 80)

    # 1. Environment & Determinism Setup
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_ver = torch.__version__
    torch_git = getattr(torch.version, "git_version", "UNKNOWN")
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    cudnn_ver = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "None"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    workspace_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET")
    det_alg_enabled = torch.are_deterministic_algorithms_enabled() if hasattr(torch, "are_deterministic_algorithms_enabled") else False

    print("\n[Complete Runtime Environment Fingerprint]")
    print(f"  Python Version          : {py_ver}")
    print(f"  PyTorch Version         : {torch_ver}")
    print(f"  PyTorch Git Version     : {torch_git}")
    print(f"  CUDA Version            : {cuda_ver}")
    print(f"  cuDNN Version           : {cudnn_ver}")
    print(f"  Device Name             : {dev_name}")
    print(f"  cuBLAS Workspace Config : {workspace_cfg}")
    print(f"  Deterministic Algorithms: {det_alg_enabled}")

    # 2. Pinned Model & Tokenizer Initial Hash Audit
    print("\n[Model Checkpoint & Pinned Artifact Cryptographic Audit]")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    base_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    initial_full_model_sha = compute_full_model_sha256(model)
    initial_scalar_checksum = float(sum(p.sum().item() for p in model.parameters()))
    initial_c_proj_sha = compute_tensor_sha256(model.transformer.h[1].mlp.c_proj.weight.data)
    fresh_c_proj_hashes = {1: initial_c_proj_sha}

    print(f"  Initial Full Model SHA256 : {initial_full_model_sha}")
    print(f"  Initial Scalar Checksum   : {initial_scalar_checksum}")
    print(f"  Layer 1 c_proj Weight SHA : {initial_c_proj_sha}")

    assert FACTS_FILE.exists(), f"Facts file not found: {FACTS_FILE}"
    with open(FACTS_FILE, "rb") as f:
        facts_file_sha = hashlib.sha256(f.read()).hexdigest()
    with open(FACTS_FILE, "r", encoding="utf-8") as f:
        facts_pool = json.load(f)
    facts_seed0, _ = sample_200_facts(facts_pool, seed=0)

    # Cryptographic hash of first 50 encoded prompts
    prompt_hasher = hashlib.sha256()
    for f in facts_seed0[:50]:
        enc_p = tokenizer(f["edit_prompt"]).input_ids
        prompt_hasher.update(bytes(enc_p))
    encoded_prompts_sha = prompt_hasher.hexdigest()

    print(f"  b1_facts.json SHA256      : {facts_file_sha}")
    print(f"  First-50 Encoded Prompts  : {encoded_prompts_sha}")

    # 3. Stage 0 Controls: G0 and G0b Cache Assertion
    print("\n[Executing Stage 0 Baseline Controls]")
    g0_res = run_g0_bit_reproduction(model, tokenizer, base_state_dict, facts_seed0, device=device)
    cache_info = load_and_assert_pinned_cache(model, tokenizer, CACHE_DIR, device=device)
    v_null = cache_info["v_null"]

    # 4. Execute Unchanged Historical Gate G0c
    print("\n" + "=" * 80)
    print(" [Executing Unchanged Historical Gate G0c (Seed 0, Arm F3 Reference)]")
    print("=" * 80)
    t0_g0c = time.time()
    g0c_run = run_audit_seed(
        model, tokenizer, base_state_dict, initial_scalar_checksum, fresh_c_proj_hashes,
        facts_seed0, v_null, use_margin=True, device=device
    )
    t_g0c_elapsed = time.time() - t0_g0c

    g0c_imm = sum(g0c_run["immediate_matches"])
    g0c_ret = sum(g0c_run["canonical_retention"])
    g0c_sha = g0c_run["final_weight_sha"]

    print(f"  G0c Immediate Efficacy    : {g0c_imm}/200 (expected 200)")
    print(f"  G0c Terminal Retention    : {g0c_ret}/50 (expected {HISTORICAL_REF_RETENTION})")
    print(f"  G0c Terminal Weight SHA   : {g0c_sha}")
    print(f"  Historical Target SHA     : {HISTORICAL_REF_WEIGHT_SHA}")
    print(f"  G0c Execution Wall-Clock  : {t_g0c_elapsed:.2f} s")

    # Gate Evaluation
    retention_passed = (g0c_ret == HISTORICAL_REF_RETENTION)
    weight_sha_passed = (g0c_sha == HISTORICAL_REF_WEIGHT_SHA)
    full_g0c_passed = retention_passed and weight_sha_passed

    print("\n[Gate G0c Verification Verdict]")
    print(f"  Retention Gate (==30)     : {'PASSED' if retention_passed else 'FAILED'} ({g0c_ret}/50)")
    print(f"  Weight Hash Gate (bit-rep): {'PASSED' if weight_sha_passed else 'FAILED'}")
    print(f"  Overall G0c Status        : {'PASSED' if full_g0c_passed else 'FAILED'}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total Wall-Clock Time     : {total_elapsed:.2f} s / {SESSION_BUDGET_CEILING:.1f} s")

    # 5. Serialization of Complete Restoration Manifest & Raw Vectors
    output_payload = {
        "metadata": {
            "directive": "S0-15_RESTORATION_TEST",
            "execution_mode": "evaluation",
            "producing_script": "experiments/s0_15_historical_restoration.py",
            "total_elapsed_s": total_elapsed,
            "session_budget_ceiling_s": SESSION_BUDGET_CEILING
        },
        "environment_manifest": {
            "python_version": py_ver,
            "pytorch_version": torch_ver,
            "pytorch_git_version": torch_git,
            "cuda_version": cuda_ver,
            "cudnn_version": cudnn_ver,
            "device_name": dev_name,
            "cublas_workspace_config": workspace_cfg,
            "deterministic_algorithms_enabled": det_alg_enabled
        },
        "artifact_hashes": {
            "initial_full_model_sha": initial_full_model_sha,
            "initial_scalar_checksum": initial_scalar_checksum,
            "initial_c_proj_sha": initial_c_proj_sha,
            "facts_file_sha": facts_file_sha,
            "encoded_prompts_first50_sha": encoded_prompts_sha,
            "cov_cache_sha": cache_info["cov_sha"],
            "v_null_cache_sha": cache_info["v_null_sha"]
        },
        "gate_g0": g0_res,
        "gate_g0c": {
            "immediate_matches_count": g0c_imm,
            "retention_count": g0c_ret,
            "expected_retention": HISTORICAL_REF_RETENTION,
            "terminal_weight_sha": g0c_sha,
            "expected_weight_sha": HISTORICAL_REF_WEIGHT_SHA,
            "retention_passed": retention_passed,
            "weight_sha_passed": weight_sha_passed,
            "overall_passed": full_g0c_passed
        },
        "raw_vectors": {
            "immediate_matches": g0c_run["immediate_matches"],
            "canonical_retention": g0c_run["canonical_retention"],
            "terminal_margins": g0c_run["terminal_margins"],
            "write_margins": g0c_run["write_margins"],
            "steps_per_edit": g0c_run["steps_per_edit"]
        }
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Restoration Audit Serialized to: {OUTPUT_FILE}]")
    print("=" * 80)

    if not full_g0c_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
