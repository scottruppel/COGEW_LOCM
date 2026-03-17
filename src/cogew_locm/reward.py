"""CogEW reward function (PRD §4.3)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cogew_locm.threat_model import ThreatState


def compute_reward(
    threat_state: "ThreatState",
    action: np.ndarray,
    prev_action: np.ndarray | None,
    *,
    power_threshold: float = 0.1,
    energy_penalty: float = 0.01,
    smooth_penalty: float = 0.1,
    effect_bonus: float = 1.0,
    effect_Q_threshold: float = 0.3,
) -> float:
    """
    r_suppress = -Q (keep track quality low)
    r_coverage = penalize if total power < power_threshold
    r_energy = -energy_penalty * power
    r_smooth = -smooth_penalty * sum((action - prev_action)^2)
    r_effect = bonus if Q < effect_Q_threshold
    """
    r_suppress = -threat_state.Q
    power = float(np.sum(action ** 2))
    r_coverage = -max(0.0, power_threshold - power)
    r_energy = -energy_penalty * power
    if prev_action is not None and prev_action.size == action.size:
        r_smooth = -smooth_penalty * float(np.sum((action - prev_action) ** 2))
    else:
        r_smooth = 0.0
    r_effect = effect_bonus if threat_state.Q < effect_Q_threshold else 0.0
    return r_suppress + r_coverage + r_energy + r_smooth + r_effect
