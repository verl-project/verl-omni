"""Tests for the duplex event miner, rule rewards, FDB harness, and calibration audit."""

from __future__ import annotations

import unittest

import numpy as np

from verl_omni.utils.dataset.duplex.event_miner import EventMiner, EventSegment, SPEECH_SAMPLE_RATE
from verl_omni.utils.dataset.duplex.calibration import run_calibration_audit
from verl_omni.utils.reward_score.duplex.rule_rewards import (
    DuplexTrajectory,
    compute_all_rewards,
    compute_interruption_yield,
    compute_pause_violation,
    compute_telemetry,
    compute_turn_latency,
    check_trip_wires,
    TripWireConfig,
)
from verl_omni.utils.reward_score.duplex.fdb_harness import (
    evaluate_fdb_timing,
    check_regression_gate,
    RegressionGateConfig,
    FDBResult,
)


def _chunk(text: str, is_listen: bool, current_time: float) -> dict:
    return {"text": text, "is_listen": is_listen, "current_time": current_time, "end_of_turn": False}


def _traj(event_type: str, chunks: list[dict], metadata: dict, preamble: float = 0.0) -> DuplexTrajectory:
    return DuplexTrajectory(chunks=chunks, event_type=event_type, event_metadata=metadata, preamble=preamble)


class TestPauseViolation(unittest.TestCase):
    def test_model_speaks_during_pause(self):
        traj = _traj("pause", [_chunk("Hello", False, 5.0)], {"pause_start": 4.0, "pause_end": 8.0})
        result = compute_pause_violation(traj)
        self.assertEqual(result["pause_violation"], -1.0)

    def test_model_stays_silent(self):
        traj = _traj("pause", [_chunk("", True, 5.0)], {"pause_start": 4.0, "pause_end": 8.0})
        result = compute_pause_violation(traj)
        self.assertEqual(result["pause_violation"], 0.0)

    def test_non_pause_event_returns_zero(self):
        traj = _traj("turn", [_chunk("Hi", False, 5.0)], {"turn_boundary": 4.0})
        result = compute_pause_violation(traj)
        self.assertEqual(result["pause_violation"], 0.0)

    def test_preamble_offset(self):
        # chunk at session t=25, preamble=20 → media t=5 (inside pause [4,8])
        traj = _traj("pause", [_chunk("Hello", False, 25.0)], {"pause_start": 4.0, "pause_end": 8.0}, preamble=20.0)
        result = compute_pause_violation(traj)
        self.assertEqual(result["pause_violation"], -1.0)


class TestTurnLatency(unittest.TestCase):
    def test_immediate_speech(self):
        traj = _traj("turn", [_chunk("", True, 4.0), _chunk("Yes", False, 4.5)], {"turn_boundary": 4.0, "segment_end": 8.0})
        result = compute_turn_latency(traj, full_credit_band=1.0)
        self.assertEqual(result["turn_latency"], 0.0)

    def test_delayed_speech(self):
        traj = _traj(
            "turn",
            [_chunk("", True, 4.0), _chunk("", True, 5.0), _chunk("", True, 6.0), _chunk("Yes", False, 7.0)],
            {"turn_boundary": 4.0, "segment_end": 8.0},
        )
        result = compute_turn_latency(traj, full_credit_band=1.0)
        self.assertLess(result["turn_latency"], 0.0)
        self.assertGreater(result["turn_latency"], -1.0)

    def test_never_takes_floor(self):
        traj = _traj("turn", [_chunk("", True, 4.0), _chunk("", True, 5.0)], {"turn_boundary": 4.0, "segment_end": 8.0})
        result = compute_turn_latency(traj)
        self.assertEqual(result["turn_latency"], -1.0)

    def test_non_turn_event(self):
        traj = _traj("pause", [_chunk("", True, 5.0)], {})
        result = compute_turn_latency(traj)
        self.assertEqual(result["turn_latency"], 0.0)


class TestInterruptionYield(unittest.TestCase):
    def test_immediate_yield(self):
        """Yield within 0.5s of overlap end → full credit."""
        traj = _traj(
            "interruption",
            [_chunk("Speaking", False, 5.0), _chunk("", True, 5.4)],
            {"overlap_end": 5.0},
        )
        result = compute_interruption_yield(traj, deadline=2.0, full_credit_band=0.5)
        self.assertEqual(result["interruption_yield"], 0.0)

    def test_delayed_yield(self):
        """Yield at 1.5s → linear penalty between 0 and -1."""
        traj = _traj(
            "interruption",
            [_chunk("Speaking", False, 5.0), _chunk("Still", False, 6.0), _chunk("", True, 6.5)],
            {"overlap_end": 5.0},
        )
        result = compute_interruption_yield(traj, deadline=3.0, full_credit_band=0.5)
        self.assertLess(result["interruption_yield"], 0.0)
        self.assertGreater(result["interruption_yield"], -1.0)

    def test_never_yields(self):
        traj = _traj("interruption", [_chunk("Speaking", False, 5.0)], {"overlap_end": 5.0})
        result = compute_interruption_yield(traj)
        self.assertEqual(result["interruption_yield"], -1.0)


class TestComputeAllRewards(unittest.TestCase):
    def test_pause_correct(self):
        traj = _traj("pause", [_chunk("", True, 5.0)], {"pause_start": 4.0, "pause_end": 8.0})
        result = compute_all_rewards(traj)
        self.assertEqual(result["reward"], 0.0)

    def test_pause_violated(self):
        traj = _traj("pause", [_chunk("Oops", False, 5.0)], {"pause_start": 4.0, "pause_end": 8.0})
        result = compute_all_rewards(traj)
        self.assertEqual(result["reward"], -1.0)


class TestTelemetry(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(compute_telemetry([]), {})

    def test_basic_telemetry(self):
        trajs = [
            _traj("pause", [_chunk("", True, 1), _chunk("", True, 2)], {}),
            _traj("turn", [_chunk("Hello", False, 5)], {"turn_boundary": 4}),
        ]
        tel = compute_telemetry(trajs)
        self.assertIn("telemetry/speak_ratio", tel)
        self.assertIn("telemetry/silence_ratio", tel)
        self.assertIn("telemetry/duplicate_trigram_share", tel)


class TestTripWires(unittest.TestCase):
    def test_all_clear(self):
        tel = {
            "telemetry/duplicate_trigram_share": 0.1,
            "telemetry/silence_ratio": 0.5,
            "telemetry/speak_ratio": 0.5,
        }
        self.assertEqual(check_trip_wires(tel), [])

    def test_silence_collapse(self):
        tel = {
            "telemetry/duplicate_trigram_share": 0.1,
            "telemetry/silence_ratio": 0.90,
            "telemetry/speak_ratio": 0.1,
        }
        violations = check_trip_wires(tel)
        self.assertTrue(any("silence" in v for v in violations))

    def test_repetition_degeneracy(self):
        tel = {
            "telemetry/duplicate_trigram_share": 0.5,
            "telemetry/silence_ratio": 0.5,
            "telemetry/speak_ratio": 0.5,
        }
        violations = check_trip_wires(tel)
        self.assertTrue(any("duplicate" in v for v in violations))


class TestFDBHarness(unittest.TestCase):
    def test_pause_evaluation(self):
        traj = _traj("pause", [_chunk("", True, 5.0)], {})
        result = evaluate_fdb_timing(traj)
        self.assertFalse(result.spoke_during_pause)

    def test_turn_evaluation(self):
        traj = _traj("turn", [_chunk("Yes", False, 5.0)], {"turn_boundary": 4.0})
        result = evaluate_fdb_timing(traj)
        self.assertIsNotNone(result.response_latency)
        self.assertAlmostEqual(result.response_latency, 1.0)

    def test_interruption_evaluation(self):
        traj = _traj(
            "interruption",
            [_chunk("Hello", False, 5.0), _chunk("", True, 6.0)],
            {"overlap_end": 5.0},
        )
        result = evaluate_fdb_timing(traj)
        self.assertTrue(result.responded_after_overlap)

    def test_regression_gate_pass(self):
        results = [
            FDBResult(event_type="pause", response_latency=None, yield_latency=None, spoke_during_pause=False, responded_after_overlap=False),
            FDBResult(event_type="turn", response_latency=1.5, yield_latency=None, spoke_during_pause=False, responded_after_overlap=True),
            FDBResult(event_type="interruption", response_latency=1.0, yield_latency=1.5, spoke_during_pause=False, responded_after_overlap=True),
        ]
        self.assertEqual(check_regression_gate(results), [])

    def test_regression_gate_fail(self):
        results = [
            FDBResult(event_type="pause", response_latency=None, yield_latency=None, spoke_during_pause=True, responded_after_overlap=False),
        ]
        violations = check_regression_gate(results)
        self.assertTrue(len(violations) > 0)


class TestCalibrationAudit(unittest.TestCase):
    def test_passing_calibration(self):
        trajs = [
            _traj("pause", [_chunk("", True, 5.0)], {"pause_start": 4.0, "pause_end": 8.0}),
            _traj("turn", [_chunk("Hi", False, 4.5)], {"turn_boundary": 4.0, "segment_end": 8.0}),
        ]
        reports = run_calibration_audit(trajs, min_mean_reward=-0.9)
        self.assertTrue(all(r.passes for r in reports))

    def test_floored_axis(self):
        # Model never takes the floor on turns → floored
        trajs = [
            _traj("turn", [_chunk("", True, 4.0), _chunk("", True, 5.0)], {"turn_boundary": 4.0, "segment_end": 8.0}),
        ]
        reports = run_calibration_audit(trajs, min_mean_reward=-0.5)
        turn_report = [r for r in reports if r.axis == "turn_latency"][0]
        self.assertFalse(turn_report.passes)


class TestEventMiner(unittest.TestCase):
    def test_event_segment_creation(self):
        seg = EventSegment(
            dialogue_id="test_0",
            event_type="pause",
            start_time=3.0,
            end_time=7.0,
            user_audio=np.zeros(16000 * 4, dtype=np.float32),
            context_start=0.0,
            context_end=3.0,
            context_audio=np.zeros(16000 * 3, dtype=np.float32),
            metadata={"pause_duration": 2.0},
        )
        self.assertEqual(seg.event_type, "pause")
        self.assertEqual(seg.dialogue_id, "test_0")

    def test_miner_init_defaults(self):
        miner = EventMiner()
        self.assertEqual(miner.min_gap, 1.0)
        self.assertEqual(miner.segment_duration, 4.0)
        self.assertEqual(miner.sample_rate, SPEECH_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
