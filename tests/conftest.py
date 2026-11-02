import os

from hypothesis import HealthCheck, settings

# Health checks grade the generators, not the code under test, and they can trip on a loaded runner.
settings.register_profile(
    "ci",
    derandomize=True,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
settings.register_profile(
    "campaign",
    deadline=None,
    database=None,
    suppress_health_check=list(HealthCheck),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
