"""Duplex rule rewards, telemetry, trip-wires and FDB harness for MiniCPM-o 4.5."""

from verl_omni.utils.reward_score.duplex.rule_rewards import (
    DuplexTrajectory,
    TripWireConfig,
    check_trip_wires,
    compute_all_rewards,
    compute_interruption_yield,
    compute_pause_violation,
    compute_telemetry,
    compute_turn_latency,
)
from verl_omni.utils.reward_score.duplex.fdb_harness import (
    FDBResult,
    RegressionGateConfig,
    check_regression_gate,
    evaluate_fdb_timing,
)

__all__ = [
    "DuplexTrajectory",
    "TripWireConfig",
    "check_trip_wires",
    "compute_all_rewards",
    "compute_interruption_yield",
    "compute_pause_violation",
    "compute_telemetry",
    "compute_turn_latency",
    "FDBResult",
    "RegressionGateConfig",
    "check_regression_gate",
    "evaluate_fdb_timing",
]
