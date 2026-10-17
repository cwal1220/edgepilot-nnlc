"""edgepilot 마찰 보상 몫(nnlc.friction)과 friction_free 목표를 시험한다.

  .venv/bin/python -m unittest discover -s tests -t .
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from nnlc.dataset import CHOSEN, build
from nnlc.friction import FRICTION, curve_weight, friction_share
from tests.test_dataset import write_csv

N = 300


class FrictionShare(unittest.TestCase):
    def test_follows_the_error_and_saturates(self):
        v = np.full(N, 25.0)  # 90 km/h: deadzone 없음
        des = np.full(N, 0.001)
        np.testing.assert_allclose(friction_share(des, des, v)[100:], 0.0, atol=1e-6)  # 오차·저크 0
        # 실제가 목표보다 0.3 m/s² 덜 돈다(오른쪽 +): 문턱 0.2를 넘어 +마찰 전부
        act = des - 0.3 / 625.0
        np.testing.assert_allclose(friction_share(des, act, v)[100:], FRICTION, rtol=1e-5)
        np.testing.assert_allclose(friction_share(-des, -act, v)[100:], -FRICTION, rtol=1e-5)
        # 문턱 안은 비례(0.1 m/s² → 절반)
        act = des - 0.1 / 625.0
        np.testing.assert_allclose(friction_share(des, act, v)[100:], 0.5 * FRICTION, rtol=1e-4)

    def test_deadzone_only_at_low_speed(self):
        des = np.full(N, 0.0)
        act = np.full(N, -0.01 / 100.0)  # 10 m/s에서 0.01 m/s²: 1° deadzone(약 0.04 m/s²) 안
        self.assertLess(float(np.abs(friction_share(des, act, np.full(N, 10.0))[100:]).max()), 1e-9)
        act = np.full(N, -0.01 / 625.0)  # 25 m/s에서는 deadzone 없음
        self.assertGreater(float(friction_share(des, act, np.full(N, 25.0))[150]), 0.0)

    def test_curve_weight_ramps_from_0_2_to_0_6(self):
        v = np.full(4, 20.0)
        np.testing.assert_allclose(curve_weight(np.array([0.1, 0.4, 0.6, 1.0]) / 400.0, v), [0.0, 0.5, 1.0, 1.0])


class FrictionFreeTarget(unittest.TestCase):
    def test_target_drops_the_weighted_friction(self):
        with tempfile.TemporaryDirectory() as d:
            # 목표 곡률이 실제보다 앞서는 경로: 오차가 있고, 뒤로 갈수록 목표 횡가속이 0.6 m/s²를 넘는다
            write_csv(Path(d) / "r.csv", des_curv=lambda t: 2.5e-4 * (t + 0.5) ** 2 + 2e-4)
            x0, y0, _, t0 = build(Path(d), replace(CHOSEN, friction_free=False))
            x1, y1, _, t1 = build(Path(d), CHOSEN)  # 기본값이 friction_free
        np.testing.assert_array_equal(x0, x1)
        np.testing.assert_array_equal(t0, t1)
        self.assertGreater(float(np.abs(y0 - y1).max()), 0.5 * FRICTION)
        self.assertLessEqual(float(np.abs(y0 - y1).max()), FRICTION + 1e-6)


if __name__ == "__main__":
    unittest.main()
