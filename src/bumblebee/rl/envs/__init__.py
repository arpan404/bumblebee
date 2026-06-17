from .mouse import MouseEnvConfig, MouseImitationEnv, VirtualScreen

__all__ = [
    "GymMouseImitationEnv",
    "MouseEnvConfig",
    "MouseImitationEnv",
    "VirtualScreen",
]


def __getattr__(name: str):
    if name == "GymMouseImitationEnv":
        from .gymnasium import GymMouseImitationEnv

        return GymMouseImitationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
