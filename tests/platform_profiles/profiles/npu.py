"""Huawei Ascend NPU — 910B3 (A2) and 910C (A3), the primary validation target.

Values follow the recipes already validated by the existing NPU smoke tests
(tests/test_qwen3_4B_npu.py, tests/test_qwen3_30B_A3B_npu.py), the delta-disk
test (tests/test_delta_disk_weight_update.py, running on npu-4 today) and the
launch policy in vime/utils/external_utils/launch.py.

Scoping rules (see README §"Topology and tuning scoping"):
  * topology/tuning values without a prefix are the default recipe for
    4B-class colocated smoke tests (the most common shape);
  * scenario-prefixed keys (e.g. "full_disk_*") apply only to the test that
    queries them — use those whenever a platform recipe differs per scenario.
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
        # Training memory bounds (4B-class models).
        "max_tokens_per_gpu": 8192,
        "log_probs_chunk_size": 128,
        # Colocated engines must fit the 4-card budget (GPU tests rely on the
        # framework default, so this renders only on NPU).
        "rollout_num_gpus": 4,
        # Full-disk weight sync (0.8B-class, non-colocate): tighter than the
        # 4B recipe — the 0.8B job OOMs on 910B3 without these bounds.
        "full_disk_max_tokens_per_gpu": 2048,
        "full_disk_vllm_max_model_len": 8192,
    },
    # Resources: native HF checkpoint loading; no torch_dist conversion step.
    launch_platform="npu",
    torch_dist_convert=False,
    # Resources: device visibility + SoC identification (for reports).
    visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
    hardware_env="SOC_VERSION",
    hardware_names={
        "ascend910b3": "ascend910b3 / A2",
        "ascend910b3b": "ascend910b3 / A2",
        "ascend910c": "ascend910c / A3",
    },
    # Topology: the validated 4B-class colocated layout on 4 cards (A2 CI
    # pool). Tests whose layout does not vary by platform never query these.
    topology={
        "num_gpus": 4,
        "tp": 4,
        "cp": 1,
    },
    # Capabilities.
    capabilities={
        "vllm_sleep_mode": True,
        # vllm-ascend does not expose /health_generate yet (404 during fanout
        # validation); tests gate health-monitored fault tolerance on this.
        "health_monitor": False,
    },
)
