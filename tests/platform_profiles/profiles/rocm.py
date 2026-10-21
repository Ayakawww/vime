"""AMD ROCm — initial support (MI300/MI355 class, gfx942/gfx950).

Mirrors the ROCm variants on the main branch (tests/test_qwen2.5_0.5B_vllm_config-rocm.py
and .buildkite/pipeline-rocm.yaml): ROCm rides the CUDA launch path (the
ray-job-submit race is worked around inside command_utils.execute_train), so
the test layer only needs a device-variant profile selected via
VIME_TEST_DEVICE=rocm (set by the ROCm CI pipeline).
"""

from platform_profiles import PlatformProfile

PROFILE = PlatformProfile(
    name="rocm",
    description="AMD Instinct (MI300/MI355); initial support, validated for the vllm-config scenarios",
    # Resources: same launcher as CUDA; HIP-specific behavior lives in
    # command_utils (torch.version.hip) and vime/utils/rocm_checkpoint_writer.py.
    launch_platform="cuda",
    # Args: flags the -rocm.py variants established on main.
    train_args_extra="--no-gradient-accumulation-fusion ",
    tuning={
        "vllm_gpu_memory_utilization": 0.3,
        # Collective weight transfer is not reliable on ROCm yet; use disk.
        "update_weight_transport": "disk",
    },
)
