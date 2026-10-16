"""내보낸 JSON을 numpy로 계산한 값이 PyTorch 망과 같은지, edgepilot이 읽을 수 있는 파일과 설정만 내보내는지 본다."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nnlc import export
from nnlc.configs import CHOSEN, CONFIGS
from nnlc.dataset import CHOSEN as CHOSEN_SPEC
from nnlc.models import Mlp
from nnlc.training import Trained, evaluate, folds


def strict_json(text: str):
    """edgepilot의 JSON 읽기처럼 NaN·Infinity를 받지 않는다."""
    def refuse(constant):
        raise ValueError(constant)
    return json.loads(text, parse_constant=refuse)


class ExportTest(unittest.TestCase):
    def test_round_trip(self):
        torch.manual_seed(0)
        feat = CHOSEN_SPEC.names()
        nets = [Mlp(len(feat), (32, 32)).eval(), Mlp(len(feat), (32, 32)).eval()]
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, len(feat))).astype(np.float32)
        x[:, feat.index("v")] = rng.uniform(5, 30, 300)
        trained = Trained(nets, x.mean(0), x.std(0) + 1e-6, feat, np.array([0.24, 1.9, 0.03, 0.03]))
        # 교차 검증 지표가 표본이 모자라 NaN이어도 파일은 엄격한 JSON이다(null)
        model = export.to_json(trained, CHOSEN_SPEC, "test", x, {"training": {"cv": {"model_10": float("nan")}}})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            export.save(model, path)
            text = path.read_text()
            loaded = export.load(path)
        self.assertIsNone(strict_json(text)["training"]["cv"]["model_10"])
        ref = np.clip(evaluate(trained, x), -1, 1)
        np.testing.assert_allclose(export.evaluate(loaded, x), ref, atol=1e-5)
        for tv in loaded["test_vectors"]:
            self.assertAlmostEqual(export.evaluate(loaded, np.array([tv["input"]]))[0], tv["output"], places=9)
        self.assertEqual(len(loaded["input_tests"][0]["inputs"]), len(feat))
        self.assertEqual(len(loaded["nets"]), 2)
        self.assertEqual([l["activation"] for l in loaded["nets"][0]["layers"]], ["tanh", "tanh", "linear"])

    def test_rejects_other_formats(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text('{"format": "something-else", "version": 1}')
            with self.assertRaises(ValueError):
                export.load(path)

    def test_only_exportable_configs(self):
        export.check_exportable(CONFIGS[CHOSEN])
        with self.assertRaises(ValueError):
            export.check_exportable(CONFIGS["sp (sunnypilot)"])
        far = CONFIGS[CHOSEN]
        with self.assertRaises(ValueError):  # 이력(1 s)보다 먼 과거
            export.check_exportable(replace(far, spec=replace(far.spec, la_past=(-1.5,))))
        with self.assertRaises(ValueError):
            export.check_exportable(replace(far, spec=replace(far.spec, extra=("driver_tq",))))


class FoldsTest(unittest.TestCase):
    def test_every_route_is_held_out_once(self):
        counts = {"2026-01-01": 100, "2026-01-02": 300, "2026-01-03": 50, "2026-01-04": 200, "2026-01-05": 120,
                  "2026-01-06": 80, "2026-01-07": 400, "2026-01-08": 90}
        names = np.array([r for r, n in counts.items() for _ in range(n)])
        result = folds(names)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(result))
        self.assertEqual(sum(result, []), sorted(counts))  # 날짜 순으로 이어진 묶음, 경로마다 한 번
        with self.assertRaises(ValueError):
            folds(np.array(["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main()
