"""
SUMO Backend Adapter — libsumo / TraCI

Provides a unified interface so the rest of the codebase can swap between
libsumo (in-process, fast) and TraCI (socket-based, supports sumo-gui)
with a single config flag.

Usage:
    from envs.sumo_backend import backend as traci

    # All existing traci.vehicle.*, traci.edge.*, traci.simulation.*
    # calls work unchanged through the adapter.

Limitations (libsumo):
    - No sumo-gui support
    - Not thread-safe — one simulation per process
    - In-process execution means a SUMO crash kills the Python process
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------
_backend_name: str = ""   # "libsumo" or "traci"
_sumo = None              # the underlying module (lazily assigned)


def set_backend(name: str) -> None:
    """Select and import the SUMO backend. Call once before any simulation."""
    global _backend_name, _sumo

    name = name.lower().strip()
    if name == _backend_name and _sumo is not None:
        return  # already initialised with this backend

    if name == "libsumo":
        import libsumo as mod
    elif name == "traci":
        import traci as mod
    else:
        raise ValueError(
            f"Unknown SUMO backend '{name}'. Use 'libsumo' or 'traci'."
        )

    _backend_name = name
    _sumo = mod
    logger.info(f"SUMO backend set to '{_backend_name}'")


def is_libsumo() -> bool:
    """Return True if the active backend is libsumo."""
    return _backend_name == "libsumo"


def _ensure_initialised() -> None:
    """Auto-initialise from env var if set_backend() was never called."""
    if _sumo is None:
        default = os.environ.get("SUMO_BACKEND", "libsumo")
        set_backend(default)


# ---------------------------------------------------------------------------
# Backend proxy object
#
# Aliased as `traci` in consumer files so every existing call site
# (traci.vehicle.getSpeed, traci.edge.getTraveltime, etc.) keeps working.
# ---------------------------------------------------------------------------
class _BackendProxy:
    """Thin proxy that forwards attribute access to the active backend module
    and normalises the few API calls that differ between TraCI and libsumo."""

    # -- pass-through sub-modules (vehicle, edge, simulation, etc.) ---------

    def __getattr__(self, name: str):
        _ensure_initialised()
        return getattr(_sumo, name)

    # -- normalised top-level calls -----------------------------------------

    def start(self, cmd: list) -> None:
        """Start a SUMO simulation.

        For libsumo the binary name in cmd[0] is ignored (in-process), but
        we replace 'sumo-gui' with 'sumo' to avoid confusing error messages.
        """
        _ensure_initialised()
        if is_libsumo():
            # libsumo cannot launch sumo-gui
            if cmd and "sumo-gui" in cmd[0]:
                cmd = list(cmd)  # copy before mutating
                cmd[0] = cmd[0].replace("sumo-gui", "sumo")
        _sumo.start(cmd)

    def simulationStep(self, time: float = 0) -> None:
        """Advance the simulation by one step.

        TraCI:   traci.simulationStep()
        libsumo: libsumo.simulation.step()
        """
        _ensure_initialised()
        if is_libsumo():
            _sumo.simulation.step(time)
        else:
            _sumo.simulationStep(time)

    def close(self) -> None:
        """Close the simulation."""
        _ensure_initialised()
        _sumo.close()


# Singleton instance — import this as `traci` in consumer modules
backend = _BackendProxy()
