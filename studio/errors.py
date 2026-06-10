"""Typed errors so the UI can show a clear message instead of crashing."""
from __future__ import annotations


class StudioError(Exception):
    """Base class for all expected, user-facing failures."""


class LoadError(StudioError):
    """The file could not be opened / decoded."""


class SolverError(StudioError):
    """The mix solver could not run (e.g. pymixbox is not installed)."""
