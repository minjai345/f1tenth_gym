"""Episode termination policy shared by environment adapters."""

from enum import IntEnum


class AgentTerminationMode(IntEnum):
    """Reduce per-agent terminal status to one environment result.

    ``EGO`` watches only ``EnvConfig.ego_index``. ``ANY`` ends the episode as
    soon as one agent satisfies a terminal condition. ``ALL`` waits until
    every agent has satisfied one; per-agent status is latched without
    freezing the corresponding vehicle.
    """

    EGO = 0
    ANY = 1
    ALL = 2
