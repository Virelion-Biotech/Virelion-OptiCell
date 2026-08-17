"""Deprecated compatibility shim.

Use ``opticell.statistics`` for the stable public API. This module intentionally
contains no implementation to avoid shadowing Python's standard-library
``statistics`` module in installed environments.
"""
from opticell.statistics import *  # noqa: F401,F403
