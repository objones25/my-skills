"""Excerpt of a Django settings.py, running on Python 3.13.

The obslog package (from the observability-expert skill, ORIGINAL version) is on the
path. Django calls logging.config.dictConfig(LOGGING) itself during setup.
"""
from obslog.setup import build_config

LOGGING_CONFIG = "logging.config.dictConfig"
LOGGING = build_config(level="INFO", fmt="json", service="shop")
