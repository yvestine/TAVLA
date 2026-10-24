"""Force-only real-target prediction, with separate current and increment supervision."""

FORMAT = "tavla_force_predictive_v4"
HISTORY = 32
FUTURE = 5
BLOCK = 16
GAP_S = 0.5
LAYOUT = [f"force_feature_{i}" for i in range(32)]
