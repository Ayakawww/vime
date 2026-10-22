"""E2E smoke test for full checkpoint weight updates through disk.

Runs a tiny Qwen3.5-0.8B job where each weight sync writes a complete HF
checkpoint and rollout engines reload it through ``update_weights_from_disk``.

Shared across platforms through tests/platform_profiles/ (see the ckpt test
for the migration recipe); GPU output is byte-identical to the pre-migration
script, enforced by tests/utils/test_platform_profiles.py.
"""

import os
import sys
import tempfile
from pathlib import Path

if (tests_dir := str(Path(__file__).resolve().parent)) not in sys.path:
    sys.path.insert(0, tests_dir)

from platform_profiles import (
    apply_profile_env,
    current_profile,
    download_dataset,
    download_model,
    ensure_asset_dirs,
    require_devices,
    validation_report,
)

import vime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"


def prepare():
    P = current_profile()
    ensure_asset_dirs()
    download_model(f"Qwen/{MODEL_NAME}")
    download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=P.topo("num_gpus", 4),
        dir_dst="/dev/shm",
    )


def build_train_args(disk_dir: str) -> str:
    P = current_profile()
    # Platforms without torch_dist conversion (see the npu profile) load the HF
    # checkpoint natively instead of a converted reference.
    ref_load = TORCH_DIST_CKPT if P.torch_dist_convert else P.model_path(MODEL_NAME)
    ckpt_args = f"--hf-checkpoint {P.model_path(MODEL_NAME)}/ " f"--ref-load {ref_load} "

    rollout_args = (
        f"--prompt-data {P.dataset_file('zhuzilin/gsm8k', 'train.parquet')} "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {os.environ.get('VIME_TEST_NUM_ROLLOUT', '1')} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 8 "
        "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {P.tune('full_disk_max_tokens_per_gpu', 9216)} "
        f"{P.flag('log-probs-chunk-size', 'log_probs_chunk_size')}"
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.01 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--rollout-num-gpus 3 "
        f"{P.flag('vllm-gpu-memory-utilization', 'vllm_gpu_memory_utilization', 0.7)}"
        f"{P.flag('vllm-max-cudagraph-capture-size', 'vllm_max_cudagraph_capture_size', 32)}"
        f"{P.flag('vllm-max-num-seqs', 'vllm_max_num_seqs')}"
        f"{P.flag('vllm-max-model-len', 'full_disk_vllm_max_model_len')}"
    )

    disk_update_args = (
        "--update-weight-mode full "
        "--update-weight-transport disk "
        f"--update-weight-disk-dir {disk_dir} "
        "--update-weight-disk-keep-files "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--loss-mask-type qwen3_5 "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{disk_update_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{P.train_args_extra}"
    )


def execute():
    P = current_profile()
    # NPU keeps checkpoint scratch under the (writable, sized) asset root;
    # cuda keeps the system default, exactly as before the migration.
    with tempfile.TemporaryDirectory(prefix="vime_full_disk_weight_update_", dir=P.tmp_dir()) as disk_dir:
        U.execute_train(
            train_args=build_train_args(disk_dir),
            num_gpus_per_node=P.topo("num_gpus", 4),
            megatron_model_type=MODEL_TYPE,
        )

        checkpoint_dirs = sorted(Path(disk_dir).glob("weight_v*"))
        assert checkpoint_dirs, f"No disk checkpoint directories were written under {disk_dir}"
        assert any((path / "model.safetensors.index.json").exists() for path in checkpoint_dirs)
        assert any(list(path.glob("*.safetensors")) for path in checkpoint_dirs)


if __name__ == "__main__":
    apply_profile_env()
    P = current_profile()
    print(P.dump())
    require_devices(P.topo("min_gpus", P.topo("num_gpus", 4)))
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    try:
        execute()
    except BaseException:
        print(
            validation_report(model=f"{MODEL_NAME} (full-disk weight update)", result="FAILED", notes="non-colocate")
        )
        raise
    print(validation_report(model=f"{MODEL_NAME} (full-disk weight update)", result="PASSED", notes="non-colocate"))
