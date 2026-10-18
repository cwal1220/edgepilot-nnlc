"""학습한 모델을 edgepilot이 읽는 JSON으로 내보내고, 같은 파일을 numpy로 계산한다(C++ 구현과 맞출 기준).

형식 "edgepilot-lateral-nn" 버전 1:
- features, spec: 입력 순서와 창 정의(dataset.Spec). 입력은 원래 단위로 넣는다.
- input_mean, input_std: (x - mean) / std로 정규화한다.
- nets: 같은 입력을 받는 망 여러 개(앙상블). 출력은 평균한 뒤 output_clip으로 자른다.
  layers마다 weight [출력][입력], bias [출력], activation "tanh" | "linear".
- curve_gain: edgepilot이 곡선(|목표 횡가속| 0.2→0.6 m/s², 30→50 km/h)에서 모델 출력을 (1 + 이것)배로 쓴다. 학습에는
  쓰지 않고 폐루프 재생으로 정한 값이다(설정의 Cfg.curve_gain, 0이면 그대로).
- linear, angle_model: 비교·시험용 정보(torqued식 선형 계수, 조향각 = a·la/v² + b·la + c).
- test_vectors: 원래 단위 입력과 기대 출력. edgepilot은 불러올 때 이것을 다시 계산해 맞아야 쓴다.
- input_tests: 합성 궤적(지금 앞뒤 la·조향각, 100 Hz)과 학습 코드(dataset.rows)가 거기서 만든 입력. edgepilot 단위
  시험이 운용 입력(LateralNnInputs)으로 같은 입력이 나오는지 본다(이름은 같고 뜻만 달라진 학습 변경을 잡는다).
JSON에는 NaN을 쓰지 않는다(edgepilot의 JSON 읽기가 받지 않아 모델 전체를 거부한다). 값이 없는 지표는 null이다.
"""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from torch import nn

from nnlc.dataset import Spec, angle_zero, rows
from nnlc.models import Mlp
from nnlc.series import DT, jerk_of

FORMAT = "edgepilot-lateral-nn"
VERSION = 1
# edgepilot이 채울 수 있는 입력 시각(지금부터 초, src/controls/lateral_nn.h kLateralNnEarliestS·LatestS): 과거는 1 s
# 이력, 미래는 계획 2.5 s에서 계획 나이 보정 0.25 s를 뺀 만큼
EARLIEST_S, LATEST_S = -0.99, 2.25


def check_exportable(cfg) -> None:
    """edgepilot이 읽을 수 있는 설정인지 학습 전에 본다: Mlp, 추가 입력은 angle·long, 창은 angle, 입력 시각은 이력과
    계획 안."""
    spec = cfg.spec
    if cfg.model != "mlp":
        raise ValueError(f"model {cfg.model!r} cannot be exported (only mlp)")
    if not set(spec.extra) <= {"angle", "long"} or any(col != "angle" for col, _ in spec.extra_windows):
        raise ValueError("edgepilot reads only angle and long extras and angle windows")
    offsets = list(spec.la_past + spec.la_future) + [o for _, offs in spec.extra_windows for o in offs] + [-0.05, 0.05]
    if not (EARLIEST_S <= spec.lag_s + min(offsets) and spec.lag_s + max(offsets) <= LATEST_S):
        raise ValueError(f"inputs must read between {EARLIEST_S} and {LATEST_S} s from now")


def _layers(net: Mlp) -> list[dict]:
    out = []
    mods = list(net.mlp)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Linear):
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            out.append({"weight": m.weight.detach().numpy().astype(float).tolist(),
                        "bias": m.bias.detach().numpy().astype(float).tolist(),
                        "activation": "tanh" if isinstance(nxt, nn.Tanh) else "linear"})
    return out


def input_tests(spec: Spec) -> list[dict]:
    """운용 입력과 대조할 합성 궤적 하나: 지금(now) 1 s 전부터 2.5 s 뒤까지의 la·조향각(100 Hz)과 일정한 속도·롤·종가속,
    그리고 dataset.rows가 표본 시각 now에서 만든 입력. 롤과 종가속을 일정하게 두는 것은 운용이 지금 값 하나를 모든 창에
    쓰기 때문이다(학습은 응답 시각의 값)."""
    t = np.arange(-1.0, 2.5 + DT / 2, DT)
    now = int(round(1.0 / DT))
    la = 0.4 + 0.6 * np.sin(1.7 * t) + 0.15 * t
    angle = -1.4 - 9.0 * np.sin(1.1 * t + 0.4)
    v, roll, long = 17.0, 0.012, -0.4
    c = {"v": np.full(len(t), v), "la": la, "jerk": jerk_of(la), "roll": np.full(len(t), roll), "angle": angle,
         "long": np.full(len(t), long)}
    x = rows(c, np.array([now]), spec)[0]
    return [{"dt": DT, "now": now, "la": la.tolist(), "angle": angle.tolist(), "v": v, "roll": roll, "long": long,
             "inputs": x.astype(float).tolist()}]


def _finite(value):
    """NaN·무한대를 null로(교차 검증 지표가 표본이 모자라 NaN일 때)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def to_json(trained, spec: Spec, name: str, x_train: np.ndarray, meta: dict, n_vectors: int = 16,
            curve_gain: float = 0.0) -> dict:
    if not 0.0 <= curve_gain <= 1.0:
        raise ValueError(f"curve_gain {curve_gain} outside [0, 1] (edgepilot refuses the file)")
    rng = np.random.default_rng(0)
    pick = x_train[rng.choice(len(x_train), n_vectors, replace=False)]
    model = {
        "format": FORMAT, "version": VERSION, "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": trained.features, "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()},
        "input_mean": trained.mean.astype(float).tolist(), "input_std": trained.std.astype(float).tolist(),
        "nets": [{"layers": _layers(net)} for net in trained.nets], "output_clip": [-1.0, 1.0],
        "curve_gain": float(curve_gain),
        "linear": dict(zip("abcd", trained.linear.astype(float).tolist())),
        "angle_model": dict(zip("abc", angle_zero(x_train, trained.features).astype(float).tolist()))
        if "angle" in trained.features else None,
        **meta,
    }
    model["test_vectors"] = [{"input": row.astype(float).tolist(), "output": float(evaluate(model, row[None])[0])}
                             for row in pick]
    model["input_tests"] = input_tests(spec)
    return model


def save(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(model), indent=1, allow_nan=False))


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
        outs.append(h[:, 0])
    lo, hi = model["output_clip"]
    return np.clip(np.mean(outs, 0), lo, hi)
