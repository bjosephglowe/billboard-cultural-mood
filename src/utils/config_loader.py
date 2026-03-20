"""
src/utils/config_loader.py

Re-export shim for src.pipeline.config_loader.

All pipeline modules and test files import configuration utilities from
this path:

    from src.utils.config_loader import load_config, ProjectConfig, config_hash

This shim exists to provide a stable, short import path throughout the
codebase without duplicating any implementation. The canonical source of
truth remains src/pipeline/config_loader.py.

Do not add implementation here. All logic lives in src/pipeline/config_loader.py.
"""

from src.pipeline.config_loader import (  # noqa: F401
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    DecadeBucket,
    ProjectConfig,
    config_hash,
    load_config,
    sentinel_config_matches,
)
