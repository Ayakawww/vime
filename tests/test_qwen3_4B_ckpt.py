import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from shlex import quote

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

ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")
parser.add_argument("--save-optimizer", choices=["cpu", "gpu"], default="cpu", help="Optimizer placement for save.")
parser.add_argument("--load-optimizer", choices=["cpu", "gpu"], default="cpu", help="Optimizer placement for load.")
parser.add_argument("--checkpoint-dir", default=None, help="Directory used for the save/load checkpoint roundtrip.")


def default_checkpoint_dir(args):
    save_mode = "async" if args.async_save else "sync"
    name = f"{MODEL_NAME}_vime_{save_mode}_{args.save_optimizer}_save_{args.load_optimizer}_load"
    return current_profile().model_path(name)


def prepare(checkpoint_dir: str):
    P = current_profile()
    ensure_asset_dirs()
    download_model(f"Qwen/{MODEL_NAME}")
    U.exec_command(f"rm -rf {quote(checkpoint_dir)}")
    download_dataset("zhuzilin/dapo-math-17k")
    download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=P.topo("num_gpus", 8),
        dir_dst=f"{P.asset_root()}/models",
    )


def optimizer_args(optimizer: str):
    args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-precision-aware-optimizer "
    )
    if optimizer == "cpu":
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    return args


def execute(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = ""):
    P = current_profile()
    # Platforms without torch_dist conversion (see the npu profile) load the HF
    # checkpoint natively instead of a converted reference.
    ref_load = P.model_path(f"{MODEL_NAME}_torch_dist") if P.torch_dist_convert else P.model_path(MODEL_NAME)
    ckpt_args = f"--hf-checkpoint {P.model_path(MODEL_NAME)}/ " f"--ref-load {ref_load} "
    checkpoint_dir_arg = quote(checkpoint_dir)
    if mode == "save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
    elif mode == "async_save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load {checkpoint_dir_arg} "
        ckpt_args += "--ckpt-step 1 "

    rollout_args = (
        f"--prompt-data {P.dataset_file('zhuzilin/dapo-math-17k', 'dapo-math-17k.jsonl')} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {P.topo('tp', 2)} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {P.topo('cp', 2)} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {P.tune('max_tokens_per_gpu', 16384)} "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    vllm_args = (
        # colocate engines must fit the platform's card budget; GPU tests rely
        # on the framework default, so the flag renders only when a profile
        # declares rollout_num_gpus
        f"{P.flag('rollout-num-gpus', 'rollout_num_gpus')}"
        # colocated engines mirror the training tensor-parallel size
        f"--rollout-num-gpus-per-engine {P.topo('tp', 2)} "
        f"{P.flag('vllm-gpu-memory-utilization', 'vllm_gpu_memory_utilization', 0.8)}"
        f"{P.flag('vllm-max-cudagraph-capture-size', 'vllm_max_cudagraph_capture_size', 16)}"
        f"{P.flag('vllm-max-num-seqs', 'vllm_max_num_seqs')}"
        f"{P.flag('vllm-max-model-len', 'vllm_max_model_len')}"
    )

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {P.topo('num_gpus', 8)} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args(optimizer)} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{P.train_args_extra}"
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=P.topo("num_gpus", 8),
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    apply_profile_env()
    P = current_profile()
    print(P.dump())
    require_devices(P.topo("min_gpus", P.topo("num_gpus", 8)))
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args)
    prepare(checkpoint_dir)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    try:
        execute(
            "save" if not args.async_save else "async_save",
            optimizer=args.save_optimizer,
            checkpoint_dir=checkpoint_dir,
        )
        execute("load", optimizer=args.load_optimizer, checkpoint_dir=checkpoint_dir)
    except BaseException:
        print(validation_report(model=f"{MODEL_NAME} (ckpt test)", result="FAILED", notes="colocate"))
        raise
    print(validation_report(model=f"{MODEL_NAME} (ckpt test)", result="PASSED", notes="colocate"))
