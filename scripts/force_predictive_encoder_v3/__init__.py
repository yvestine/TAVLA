"""Force-only, trend-aligned predictive representations; no policy side effects."""

FORMAT = "tavla_force_predictive_v3"
HISTORY = 32
PAST = 16
FUTURE = 5
BLOCK = 16
GAP_S = .5
TAU_S = .3
LAYOUT = [f"trend_{i}" for i in range(16)] + [f"transient_{i}" for i in range(16)]
