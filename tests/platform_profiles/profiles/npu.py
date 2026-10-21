"""Huawei Ascend NPU — 910B3 (A2) and 910C (A3), the primary validation target.

Values follow the recipes already validated by the existing NPU smoke tests
(tests/test_qwen3_4B_npu.py, tests/test_qwen3_30B_A3B_npu.py) and the launch
policy in vime/utils/external_utils/launch.py.
"""

import json

from platform_profiles import PlatformProfile

PROFILE = PlatformProfile(
    name="npu",
    description="Huawei Ascend 910B3 (A2) / 910C (A3), validated on the ascend-a3 CI queue",
    # Paths: NPU CI mounts a warm HF cache; keep the VIME_TEST_ROOT override first.
    asset_root_env_vars=("VIME_TEST_ROOT", "HF_HOME"),
    default_asset_root="/root",
    tmp_under_asset_root=True,
    # Args: unconditional launch flags for every shared smoke test.
    train_args_extra=(
        "--use-flash-attn "
        "--no-gradient-accumulation-fusion "
        f"--vllm-additional-config '{json.dumps({'weight_nz_mode': 0})}' "
    ),
    # Args: per-knob tuning. Keys are semantic; tests render them with
    # P.tune()/P.flag() and keep their GPU-blessed values as defaults.
    tuning={
        # Serving memory bounds for colocated runs (4B-class models).
        "vllm_gpu_memory_utilization": 0.5,
        "vllm_max_num_seqs": 16,
        "vllm_max_cudagraph_capture_size": 8,
        "vllm_max_model_len": 32768,
        # Training memory bounds.
        "max_tokens_per_gpu": 8192,
        "log_probs_chunk_size": 128,
    },
    # Resources: native HF checkpoint loading; no torch_dist conversion step.
    launch_platform="npu",
    torch_dist_convert=False,
    # Capabilities.
    capabilities={
        "vllm_sleep_mode": True,
    },
)
