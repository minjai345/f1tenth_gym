from __future__ import annotations

from enum import IntEnum
from typing import Optional

from .masked_reset import GridResetFn, AllTrackResetFn
from .map_reset import AllMapResetFn
from .reset_fn import ResetFn
from ..track import Track


class ResetStrategy(IntEnum):
    """Enumeration of the supported environment reset strategies."""

    RL_GRID_STATIC = 1
    RL_RANDOM_STATIC = 2
    RL_GRID_RANDOM = 3
    RL_RANDOM_RANDOM = 4
    MAP_RANDOM_STATIC = 5


def make_reset_fn(
    track: Track,
    num_agents: int,
    type: Optional[ResetStrategy] = None,
    **kwargs,
) -> ResetFn:
    """Factory for constructing a reset function based on the chosen strategy."""

    strategy = type or ResetStrategy.RL_GRID_STATIC
    if not isinstance(strategy, ResetStrategy):
        raise TypeError("type must be a ResetStrategy")

    type_token = {
        ResetStrategy.RL_GRID_STATIC: "rl_grid_static",
        ResetStrategy.RL_RANDOM_STATIC: "rl_random_static",
        ResetStrategy.RL_GRID_RANDOM: "rl_grid_random",
        ResetStrategy.RL_RANDOM_RANDOM: "rl_random_random",
        ResetStrategy.MAP_RANDOM_STATIC: "map_random_static",
    }[strategy]

    try:
        refline_token, reset_token, shuffle_token = type_token.split("_")

        if refline_token == "map":
            reset_fn = {"random": AllMapResetFn}[reset_token]
            shuffle = {"static": False, "random": True}[shuffle_token]
            return reset_fn(track=track, num_agents=num_agents, shuffle=shuffle, **kwargs)

        # "cl" or "rl"
        refline = {"cl": track.centerline, "rl": track.raceline}[refline_token]
        reset_fn = {"grid": GridResetFn, "random": AllTrackResetFn}[reset_token]
        shuffle = {"static": False, "random": True}[shuffle_token]
        options = {"cl": {"move_laterally": True}, "rl": {"move_laterally": False}}[refline_token]

    except Exception as ex:
        raise ValueError(
            f"Invalid reset function type: {strategy}. Expected format: <refline>_<resetfn>_<shuffle>"
        ) from ex

    return reset_fn(
        reference_line=refline,
        num_agents=num_agents,
        shuffle=shuffle,
        **options,
        **kwargs,
    )
