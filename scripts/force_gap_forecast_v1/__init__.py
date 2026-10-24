"""Causal real-force forecasting and sim-to-real force adaptation."""

FORMAT = "tavla_real_force_forecast_v1"
HISTORY = 32
PAST = 8
FUTURE = 5
GAP_S = 0.5
LAYOUT = ["real_equivalent_Fx_N", "real_equivalent_Fy_N", "real_equivalent_Fz_N",
          "delta_Fx_N_per_frame", "delta_Fy_N_per_frame", "delta_Fz_N_per_frame"] + [
              f"temporal_{i}" for i in range(26)]
