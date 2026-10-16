"""학습한 모델을 edgepilot이 읽는 JSON으로 내보내고, 같은 파일을 numpy로 계산한다(C++ 구현과 맞출 기준).

형식 "edgepilot-lateral-nn" 버전 1:
- features, spec: 입력 순서와 창 정의(dataset.Spec). 입력은 원래 단위로 넣는다.
- input_mean, input_std: (x - mean) / std로 정규화한다.
- nets: 같은 입력을 받는 망 여러 개(앙상블). 출력은 평균한 뒤 output_clip으로 자른다.
  layers마다 weight [출력][입력], bias [출력], activation "tanh" | "silu" | "linear".
- linear, angle_model: 비교·시험용 정보(torqued식 선형 계수, 조향각 = a·la/v² + b·la + c).
- test_vectors: 원래 단위 입력과 기대 출력. C++ 단위 시험이 이것과 대조한다.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from torch import nn

from nnlc.dataset import Spec, angle_zero
from nnlc.models import Mlp

FORMAT = "edgepilot-lateral-nn"
VERSION = 1


def _layers(net: Mlp) -> list[dict]:
    if net.linear is not None:
        raise ValueError("wide 모델은 내보내지 않는다")
    out = []
    mods = list(net.mlp)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Linear):
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            act = {nn.Tanh: "tanh", nn.SiLU: "silu"}.get(type(nxt), "linear")
            out.append({"weight": m.weight.detach().numpy().astype(float).tolist(),
                        "bias": m.bias.detach().numpy().astype(float).tolist(), "activation": act})
    return out


def to_json(trained, spec: Spec, name: str, x_train: np.ndarray, meta: dict, n_vectors: int = 16) -> dict:
    rng = np.random.default_rng(0)
    pick = x_train[rng.choice(len(x_train), n_vectors, replace=False)]
    model = {
        "format": FORMAT, "version": VERSION, "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": trained.features, "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()},
        "input_mean": trained.mean.astype(float).tolist(), "input_std": trained.std.astype(float).tolist(),
        "nets": [{"layers": _layers(net)} for net in trained.nets], "output_clip": [-1.0, 1.0],
        "linear": dict(zip("abcd", trained.linear.astype(float).tolist())),
        "angle_model": dict(zip("abc", angle_zero(x_train, trained.features).astype(float).tolist()))
        if "angle" in trained.features else None,
        **meta,
    }
    model["test_vectors"] = [{"input": row.astype(float).tolist(), "output": float(evaluate(model, row[None])[0])}
                             for row in pick]
    return model


def save(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1))


def load(path: Path) -> dict:
    model = json.loads(Path(path).read_text())
    if model.get("format") != FORMAT or model.get("version") != VERSION:
        raise ValueError(f"{path}: not an {FORMAT} v{VERSION} file")
    return model


def evaluate(model: dict, x: np.ndarray) -> np.ndarray:
    """원래 단위 입력 [n, f] → 토크 [n]. C++ 구현이 따라야 하는 계산 그대로다."""
    h0 = (np.asarray(x, np.float64) - np.asarray(model["input_mean"])) / np.asarray(model["input_std"])
    outs = []
    for net in model["nets"]:
        h = h0
        for layer in net["layers"]:
            h = h @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
            if layer["activation"] == "tanh":
                h = np.tanh(h)
            elif layer["activation"] == "silu":
                h = h / (1.0 + np.exp(-h))
        outs.append(h[:, 0])
    lo, hi = model["output_clip"]
    return np.clip(np.mean(outs, 0), lo, hi)
