"""합성 CSV로 표본 정렬(지연·창 오프셋)과 좌우 증강, CSV 읽기를 시험한다.

  .venv/bin/python -m unittest discover -s tests -t .
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnlc.csvdata import COLS, HEADER, OLD_HEADER, load, usable_mask
from nnlc.dataset import CHOSEN, SP18, build, mirrored
from nnlc.series import segments

V = 20.0
CURV = 2.5e-4  # act_curv = CURV·t², la = CURV·V²·t² = 0.1·t²: 한 틱 어긋나면 la가 0.002·t 바뀐다


def write_csv(path: Path, n: int = 3000, **columns) -> None:
    """100 Hz, 일정 속도, 곡률이 시간의 제곱인 경로. 토크는 시각 값 그대로라 목표 정렬을 바로 확인할 수 있다. 종가속과
    롤은 시간에 비례해 응답 시각에서 읽는지 볼 수 있다. columns는 열 값(시각 배열을 받는 함수)을 바꾼다."""
    t = np.arange(n) * 0.01
    rows = {c: np.zeros(n) for c in COLS}
    rows.update(t=t, v=np.full(n, V), active=1, toi_active=1, lat_valid=1, model_age=0.05, la_p15=0.0,
                act_curv=CURV * t ** 2, tq_norm=t, angle=-t * 2.0, long=0.01 * t, roll=0.001 * t)
    rows.update({c: f(t) for c, f in columns.items()})
    rows = {c: np.broadcast_to(np.asarray(val, float), (n,)) for c, val in rows.items()}
    with path.open("w") as f:
        f.write(",".join(HEADER) + "\n")
        for i in range(n):
            vals = [f"{rows[c][i]:.9g}" for c in COLS]
            f.write(",".join(vals[:4] + ["-"] + vals[4:]) + "\n")


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_csv(cls.dir / "2026-01-01--00-00-00-000.csv")
        # 행이 없거나 하나뿐인 경로(끊긴 녹화)는 건너뛴다
        (cls.dir / "2026-01-02--00-00-00-000.csv").write_text(",".join(HEADER) + "\n")
        with (cls.dir / "2026-01-03--00-00-00-000.csv").open("w") as f:
            f.write(",".join(HEADER) + "\n" + ",".join(["0"] * 4 + ["-"] + ["0"] * (len(COLS) - 4)) + "\n")
        segments.cache_clear()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_alignment(self):
        x, y, names, times = build(self.dir, SP18)
        feat = SP18.names()
        self.assertEqual(x.shape[1], 18)
        self.assertEqual(set(names), {"2026-01-01--00-00-00-000"})
        r = times + SP18.lag_s  # 응답 시각
        close = dict(rtol=1e-5, atol=1e-4)
        # 목표는 표본 시각 t의 토크(= t), la·저크·롤은 응답 시각 기준
        np.testing.assert_allclose(y, times, atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("la")], 0.1 * r ** 2, **close)
        np.testing.assert_allclose(x[:, feat.index("la-0.3")], 0.1 * (r - 0.3) ** 2, **close)
        np.testing.assert_allclose(x[:, feat.index("la+1.5")], 0.1 * (r + 1.5) ** 2, **close)
        # 저크는 float32 la의 차분이라 2e-4쯤 어긋난다(한 틱 어긋나면 0.002)
        np.testing.assert_allclose(x[:, feat.index("jerk")], 0.2 * r, atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("roll")], 0.001 * r, atol=1e-6)
        np.testing.assert_allclose(x[:, feat.index("roll+1.5")], 0.001 * (r + 1.5), atol=1e-6)
        # 창이 구간 안에 들어오는 표본만, 20 Hz
        self.assertGreaterEqual(times.min(), 0.0)
        self.assertLessEqual(times.max() + SP18.lag_s + 1.5, 29.99)
        np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-6)

    def test_chosen_extras(self):
        x, _, _, times = build(self.dir, CHOSEN)
        feat = CHOSEN.names()
        self.assertEqual(len(feat), 33)
        r = times + CHOSEN.lag_s
        np.testing.assert_allclose(x[:, feat.index("la-0.5")], 0.1 * (r - 0.5) ** 2, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(x[:, feat.index("angle")], -2.0 * r, atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("angle+1.50")], -2.0 * (r + 1.5), atol=1e-3)
        np.testing.assert_allclose(x[:, feat.index("long")], 0.01 * r, atol=1e-5)

    def test_other_columns_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "other.csv"
            path.write_text(",".join(HEADER[:-2]) + "\n")
            with self.assertRaises(ValueError):
                load(path)

    def test_files_before_nn_ff_read_as_linear_drives(self):
        """2026-10-19 전 추출기의 파일(nn_ff 열 없음)은 nn_ff 0으로 읽고, 학습 모델로 달린 행(nn_ff 1)은 학습에서
        뺀다: 곡선 이득이 보낸 토크에 이미 들어 있다."""
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.csv"
            row = ["1"] * 4 + ["-"] + ["1"] * (len(OLD_HEADER) - 5)
            old.write_text(",".join(OLD_HEADER) + "\n" + ",".join(row) + "\n")
            self.assertEqual(load(old)["nn_ff"].tolist(), [0.0])
            new = Path(d) / "new.csv"
            write_csv(new, n=200, nn_ff=lambda t: (t > 1.0).astype(float))
            data = load(new)
            keep = usable_mask(data)
            self.assertTrue(keep[data["t"] < 0.99].all())
            self.assertFalse(keep[data["t"] > 1.01].any())

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




class GzipCsvTest(unittest.TestCase):
    def test_gzipped_csv_reads_like_the_plain_one(self):
        import gzip
        import shutil

        from nnlc.csvdata import load, route_files, route_name
        with tempfile.TemporaryDirectory() as d:
            plain = Path(d) / "plain" / "2026-01-01--00-00-00-000.csv"
            plain.parent.mkdir()
            write_csv(plain)
            packed = Path(d) / "packed" / "2026-01-01--00-00-00-000.csv.gz"
            packed.parent.mkdir()
            with plain.open("rb") as src, gzip.open(packed, "wb") as dst:
                shutil.copyfileobj(src, dst)
            self.assertEqual(route_name(packed), "2026-01-01--00-00-00-000")
            a, b = load(plain), load(packed)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])
            self.assertTrue((packed.parent / "2026-01-01--00-00-00-000.npy").exists())  # 캐시 이름에 .csv가 붙지 않는다
            shutil.copy(plain, packed.parent / plain.name)  # 같은 경로가 둘이면 .csv 하나만
            self.assertEqual([p.name for p in route_files(packed.parent)], [plain.name])
            x0, y0, _, _ = build(plain.parent, CHOSEN)
            (packed.parent / plain.name).unlink()
            x1, y1, _, _ = build(packed.parent, CHOSEN)
            np.testing.assert_array_equal(x0, x1)
            np.testing.assert_array_equal(y0, y1)


if __name__ == "__main__":
    unittest.main()
