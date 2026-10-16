"""합성 CSV로 표본 정렬(지연·창 오프셋)과 좌우 증강을 시험한다.

  .venv/bin/python -m unittest discover -s tests
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnlc.csvdata import COLS
from nnlc.dataset import CHOSEN, SP18, build, mirrored
from nnlc.series import segments


def write_csv(path: Path, n: int = 3000, v: float = 20.0) -> None:
    """100 Hz, 일정 속도, 곡률이 시간에 비례하는 경로(la = act_curv·v² = 1e-4·t·v²). 토크는 시각 값 그대로라
    목표 정렬을 바로 확인할 수 있다."""
    t = np.arange(n) * 0.01
    rows = {c: np.zeros(n) for c in COLS}
    rows.update(t=t, v=np.full(n, v), active=1, toi_active=1, lat_valid=1, model_age=0.05, la_p15=0.0,
                act_curv=t * 1e-4, tq_norm=t, angle=-t * 2.0, long=0.5, roll=0.01)
    rows = {c: np.broadcast_to(np.asarray(val, float), (n,)) for c, val in rows.items()}
    header = COLS[:4] + ["block"] + COLS[4:]
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for i in range(n):
            vals = [f"{rows[c][i]:.7g}" for c in COLS]
            f.write(",".join(vals[:4] + ["-"] + vals[4:]) + "\n")


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_csv(cls.dir / "2026-01-01--00-00-00-000.csv")
        segments.cache_clear()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_alignment(self):
        x, y, names, times = build(self.dir, None, SP18)
        feat = SP18.names()
        self.assertEqual(x.shape[1], 18)
        k = 1e-4 * 20.0 ** 2  # la = k·t
        # 목표는 표본 시각 t의 토크(= t), la는 t + lag
        np.testing.assert_allclose(y, times, atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("la")], k * (times + SP18.lag_s), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("la-0.3")], k * (times + SP18.lag_s - 0.3), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("la+1.5")], k * (times + SP18.lag_s + 1.5), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("jerk")], k, atol=1e-3)
        # 창이 구간 안에 들어오는 표본만, 20 Hz
        self.assertGreaterEqual(times.min(), 0.0)
        self.assertLessEqual(times.max() + SP18.lag_s + 1.5, 29.99)
        np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-6)

    def test_chosen_extras(self):
        x, _, _, times = build(self.dir, None, CHOSEN)
        feat = CHOSEN.names()
        self.assertEqual(len(feat), 33)
        np.testing.assert_allclose(x[:, feat.index("angle")], -2.0 * (times + CHOSEN.lag_s), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("angle+1.50")], -2.0 * (times + CHOSEN.lag_s + 1.5), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("long")], 0.5, atol=1e-6)

    def test_mirror_is_an_involution(self):
        rng = np.random.default_rng(0)
        feat = CHOSEN.names()
        x = rng.normal(size=(500, len(feat))).astype(np.float32)
        x[:, feat.index("v")] = rng.uniform(5, 30, 500)
        la = x[:, feat.index("la")]
        x[:, [i for i, f in enumerate(feat) if f.startswith("angle")]] = (-2600 * la / x[:, 0] ** 2 - 1.4)[:, None]
        y = rng.normal(size=500).astype(np.float32)
        xm, ym = mirrored(x, y, feat, torque_bias=0.03)
        self.assertEqual(len(xm), 1000)
        flipped, yf = xm[500:], ym[500:]
        np.testing.assert_allclose(flipped[:, feat.index("la")], -la, atol=1e-6)
        np.testing.assert_allclose(flipped[:, feat.index("v")], x[:, 0], atol=1e-6)
        np.testing.assert_allclose(yf, 0.06 - y, atol=1e-6)
        # 조향각은 영점(-1.4°) 기준으로 뒤집힌다
        np.testing.assert_allclose(flipped[:, feat.index("angle")] + x[:, feat.index("angle")], -2.8, atol=1e-3)
        back, yb = mirrored(flipped, yf, feat, torque_bias=0.03)
        np.testing.assert_allclose(back[500:], x, atol=1e-3)
        np.testing.assert_allclose(yb[500:], y, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
