"""내보낸 JSON을 numpy로 계산한 값이 PyTorch 망과 같은지, 시험 벡터가 맞는지 본다."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nnlc import export
from nnlc.dataset import CHOSEN
from nnlc.models import Mlp
from nnlc.training import Trained, evaluate


class ExportTest(unittest.TestCase):
    def test_round_trip(self):
        torch.manual_seed(0)
        feat = CHOSEN.names()
        nets = [Mlp(len(feat), (32, 32)).eval(), Mlp(len(feat), (32, 32)).eval()]
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, len(feat))).astype(np.float32)
        x[:, feat.index("v")] = rng.uniform(5, 30, 300)
        trained = Trained(nets, x.mean(0), x.std(0) + 1e-6, feat, np.array([0.24, 1.9, 0.03, 0.03]))
        model = export.to_json(trained, CHOSEN, "test", x, {"training": {}})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            export.save(model, path)
            loaded = export.load(path)
        ref = np.clip(evaluate(trained, x), -1, 1)
        np.testing.assert_allclose(export.evaluate(loaded, x), ref, atol=1e-5)
        for tv in loaded["test_vectors"]:
            self.assertAlmostEqual(export.evaluate(loaded, np.array([tv["input"]]))[0], tv["output"], places=9)
        self.assertEqual(len(loaded["nets"]), 2)
        self.assertEqual([l["activation"] for l in loaded["nets"][0]["layers"]], ["tanh", "tanh", "linear"])

    def test_rejects_other_formats(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text('{"format": "something-else", "version": 1}')
            with self.assertRaises(ValueError):
                export.load(path)


if __name__ == "__main__":
    unittest.main()
