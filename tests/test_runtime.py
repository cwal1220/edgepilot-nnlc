"""운용 입력 표본(nnlc.runtime): 입력 파일의 행과 경로 CSV의 짝, 이름으로 고르는 입력, 시간축 검사.

  .venv/bin/python -m unittest discover -s tests -t .
"""
import contextlib
import gzip
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnlc.csvdata import route_files
from nnlc.dataset import CHOSEN, NO_ANGLE
from nnlc.runtime import build_runtime, inputs_file
from nnlc.series import segments
from tests.test_dataset import write_csv

ROUTE = "2026-01-01--00-00-00-000"


def long_of(t):
    """종가속: 시간축 검사가 어긋남을 가려낼 수 있게 주기가 서로 다른 사인 둘"""
    return 0.3 * np.sin(2.1 * t) + 0.2 * np.sin(5.3 * t + 1.0)


def write_inputs(path: Path, t: np.ndarray, value_of, shift_s: float = 0.0) -> None:
    """운용 입력 파일(replay_controls --nn-inputs 꼴, CHOSEN의 33개). 입력 열마다 value_of(이름, 시각) 값이고 long은
    CSV와 같은 신호다. shift_s만큼 시각을 늦춰 쓰면 두 시간축이 어긋난다."""
    names = CHOSEN.names()
    with gzip.open(path, "wt") as f:
        f.write(",".join(["t"] + names) + "\n")
        for ti in t:
            row = [long_of(ti) if n == "long" else value_of(n, ti) for n in names]
            f.write(",".join([f"{ti + shift_s:.3f}"] + [f"{v:.6g}" for v in row]) + "\n")


class RuntimeInputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_csv(self.dir / f"{ROUTE}.csv", long=long_of)
        segments.cache_clear()
        # 20 Hz, 경로 CSV의 격자에서 2 ms 비껴 있다(재생 틱과 녹화 틱의 위상이 다르다)
        self.t = np.arange(0.002, 29.9, 0.05)
        # 입력 값은 이름마다 다른 시각의 함수라 짝이 맞았는지 바로 보인다
        self.value = lambda name, t: (len(name) + 0.001 * t) if name != "v" else 20.0

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_pair_with_the_route_and_inputs_are_picked_by_name(self):
        write_inputs(self.dir / f"{ROUTE}.inputs.csv.gz", self.t, self.value)
        self.assertEqual([p.name for p in route_files(self.dir)], [f"{ROUTE}.csv"])  # 입력 파일은 경로가 아니다
        self.assertEqual(inputs_file(self.dir / f"{ROUTE}.csv").name, f"{ROUTE}.inputs.csv.gz")
        x, y, names, times = build_runtime(self.dir, NO_ANGLE)
        feat = NO_ANGLE.names()
        self.assertEqual(x.shape[1], len(feat))
        self.assertEqual(set(names), {ROUTE})
        # 목표는 그 격자 점의 보낸 토크(합성 CSV에서 = 시각), 입력은 그 행(격자 점 + 2 ms)의 값
        np.testing.assert_allclose(y, times, atol=1e-3)
        rows = times + 0.002
        for name in ("la", "la+1.5", "roll-0.3", "long"):
            expected = long_of(rows) if name == "long" else len(name) + 0.001 * rows
            np.testing.assert_allclose(x[:, feat.index(name)], expected, atol=2e-3, err_msg=name)
        # 행마다 한 표본(20 Hz), 창이 구간 안에 드는 것만
        np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-6)
        self.assertLessEqual(times.max() + NO_ANGLE.lag_s + 1.5, 29.99)

    def test_a_route_off_the_clock_is_left_out(self):
        write_inputs(self.dir / f"{ROUTE}.inputs.csv.gz", self.t, self.value, shift_s=0.4)
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(ValueError):
            build_runtime(self.dir, NO_ANGLE)
        self.assertIn("off the route's clock", err.getvalue())

    def test_a_missing_input_is_refused(self):
        names = [n for n in CHOSEN.names() if n != "la+1.5"]
        with gzip.open(self.dir / f"{ROUTE}.inputs.csv.gz", "wt") as f:
            f.write(",".join(["t"] + names) + "\n0.002," + ",".join(["0"] * len(names)) + "\n")
        with self.assertRaises(ValueError):
            build_runtime(self.dir, NO_ANGLE)

    def test_routes_without_inputs_give_no_samples(self):
        with self.assertRaises(ValueError):
            build_runtime(self.dir, NO_ANGLE)


if __name__ == "__main__":
    unittest.main()
