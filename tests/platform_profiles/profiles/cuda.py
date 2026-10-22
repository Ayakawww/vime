"""NVIDIA CUDA — the reference profile.

Deliberately *neutral*: args/tuning/topology/env are all empty so that any
shared test renders byte-identical arguments to the pre-migration GPU script
(invariant enforced by tests/utils/test_platform_profiles.py). Pre-migration
GPU behavior IS the profile's default behavior.

`capabilities` is exempt from neutrality: it states feature presence and
never changes rendered arguments. health_monitor=True keeps fault-tolerance
features enabled exactly as in the existing GPU tests.
"""

from platform_profiles import PlatformProfile

PROFILE = PlatformProfile(
    name="cuda",
    description="NVIDIA CUDA (H100/GB200/GB300 CI pools); the neutral reference profile",
    capabilities={
        # /health_generate and friends exist on the vLLM CUDA stack; tests gate
        # --use-fault-tolerance style flags on this (absent on Ascend today).
        "health_monitor": True,
    },
)
