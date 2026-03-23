"""
PyMARL Environment Module

This module contains multi-agent RL environments compatible with PyMARL.
"""

from .sumo_grid_reroute import SUMOGridRerouteEnv

__all__ = ["SUMOGridRerouteEnv"]


# Environment registry for PyMARL
ENV_REGISTRY = {
    "sumo_grid_reroute": SUMOGridRerouteEnv,
}


def get_env_class(env_name: str):
    """
    Get environment class by name.

    Args:
        env_name: Name of environment

    Returns:
        Environment class
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_REGISTRY.keys())}")

    return ENV_REGISTRY[env_name]
