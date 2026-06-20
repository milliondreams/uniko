"""uniko — async-first Python SDK for the uniko cognitive memory engine.

The compiled PyO3 extension is imported as ``uniko._uniko`` and its public
surface is re-exported here so callers write ``import uniko`` /
``from uniko import Uniko``.

Almost every method is async (returns an awaitable driven by ``asyncio``);
construction helpers and builder setters are synchronous.
"""

from __future__ import annotations

from . import _uniko
from ._uniko import *  # noqa: F401,F403  (re-export the full native surface)

# The Pydantic IO layer is reachable as ``uniko.models`` (e.g.
# ``from uniko.models import ContextBundle, GoalSpec, TypedUniko``). Importing it
# here only binds the submodule; it does NOT pull model classes into ``uniko.*``.
from . import models  # noqa: F401,E402

# Mirror the native module's __all__ when it defines one; otherwise fall back
# to every public attribute the extension exposes.
__all__ = getattr(
    _uniko, "__all__", [name for name in dir(_uniko) if not name.startswith("_")]
)

__version__ = getattr(_uniko, "__version__", "0.1.0")
