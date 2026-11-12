"""Optional Numba JIT compilation.

Numba is a hard dependency of the package, but the decorator falls back to plain
Python when Numba is missing or disabled (``TITRATERL_DISABLE_JIT=1``), which is
useful for debugging the model with a normal debugger.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_DISABLED = os.environ.get("TITRATERL_DISABLE_JIT", "0") == "1"

try:  # pragma: no cover - exercised implicitly
    if _DISABLED:
        raise ImportError
    from numba import njit as _njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _njit = None
    NUMBA_AVAILABLE = False


def njit(*args: Any, **kwargs: Any) -> Any:
    """``numba.njit(cache=True)`` when available, identity decorator otherwise."""
    if _njit is None:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
    kwargs.setdefault("cache", True)
    return _njit(*args, **kwargs)
