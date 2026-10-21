"""NVIDIA CUDA — the reference profile.

Deliberately *neutral*: every section is empty so that any shared test renders
byte-identical arguments to the pre-migration GPU script (invariant enforced
by tests/utils/test_platform_profiles.py). Pre-migration GPU behavior IS the
profile's default behavior.
"""

from platform_profiles import PlatformProfile

PROFILE = PlatformProfile(
    name="cuda",
    description="NVIDIA CUDA (H100/GB200/GB300 CI pools); the neutral reference profile",
)
