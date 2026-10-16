"""학습 표본(입력 X, 목표 토크 y)을 만든다.

- 목표: 표본 시각 t에 보낸 토크 tq_norm(부호 보정, steer_max로 나눈 값).
- 입력은 응답 시각 t + lag 기준이다(토크가 조향각 횡가속으로 나타나는 지연). 창(과거·미래 오프셋)도 t + lag에서 센다.
  학습은 실제로 일어난 궤적을 쓰고, 운용 때는 같은 자리에 목표 궤적을 넣는다(역동역학: 이 궤적을 만든 토크).
- 표본은 t와 t + lag가 모두 usable이고, 창 전체가 한 연속 구간 안에 있고, 창 안에 차선 변경·깜빡이가 없을 때만
  쓴다. 100 Hz 격자에서 stride마다 하나(기본 5 = 20 Hz).
- Spec이 창과 추가 입력을 정한다. SP18은 sunnypilot NNLC 입력 18개와 같고, CHOSEN은 비교 실험에서 고른 33개다.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nnlc.series import DT, segments


@dataclass(frozen=True)
class Spec:
    lag_s: float = 0.3
    la_past: tuple = (-0.3, -0.2, -0.1)
    la_future: tuple = (0.3, 0.6, 1.0, 1.5)
    roll_past: tuple = (-0.3, -0.2, -0.1)
    roll_future: tuple = (0.3, 0.6, 1.0, 1.5)
    extra: tuple = ()          # 응답 시각의 열 값: long(종가속), angle(조향각), driver_tq(운전자 토크) 등
    extra_windows: tuple = ()  # (열 이름, 오프셋들), 예: (("angle", (-0.3, 0.3, 1.0)),)
    stride: int = 5
    driver_thr: float = 100.0

    def names(self) -> list[str]:
        return (["v", "la", "jerk", "roll"] + [f"la{p:+.1f}" for p in self.la_past + self.la_future] +
                [f"roll{p:+.1f}" for p in self.roll_past + self.roll_future] + list(self.extra) +
                [f"{col}{p:+.2f}" for col, offs in self.extra_windows for p in offs])


SP18 = Spec()
CHOSEN = Spec(la_past=(-0.5, -0.4, -0.3, -0.2, -0.1), la_future=(0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5),
              extra=("angle", "long"), extra_windows=(("angle", (-0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5)),))
assert len(SP18.names()) == 18 and len(CHOSEN.names()) == 33


def _steps(offsets) -> list[int]:
    return [int(round(p / DT)) for p in offsets]


def build(csv_dir: Path, routes: list[str] | None, spec: Spec = SP18):
    """(X [n, f] float32, y [n], 경로 이름 [n], 표본 시각 [n]). routes가 None이면 전부."""
    lag = int(round(spec.lag_s / DT))
    lp, lf, rp, rf = _steps(spec.la_past), _steps(spec.la_future), _steps(spec.roll_past), _steps(spec.roll_future)
    ew = [(col, _steps(offs)) for col, offs in spec.extra_windows]
    every = lp + lf + rp + rf + [o for _, offs in ew for o in offs] + [0]
    lo, hi = min(every), max(every)
    xs, ys, names, times = [], [], [], []
    for csv in sorted(Path(csv_dir).glob("*.csv")):
        if routes is not None and csv.stem not in routes:
            continue
        for sg in segments(csv, spec.driver_thr):
            c = sg.cols
            n = len(sg.t)
            usable = c["usable"] > 0
            ev_cum = np.r_[0, np.cumsum((c["desire"] != 0) | (c["blinker"] != 0))]
            idx = np.arange(max(0, -(lag + lo)), n - (lag + hi), spec.stride)
            if len(idx) == 0:
                continue
            r = idx + lag
            ok = usable[idx] & usable[r] & (ev_cum[r + hi + 1] - ev_cum[r + lo] == 0)
            idx, r = idx[ok], r[ok]
            if len(idx) == 0:
                continue
            la, roll = c["la"], c["roll"]
            cols = [c["v"][idx], la[r], c["jerk"][r], roll[r]]
            cols += [la[r + p] for p in lp + lf] + [roll[r + p] for p in rp + rf]
            cols += [c[e][r] for e in spec.extra]
            cols += [c[col][r + o] for col, offs in ew for o in offs]
            x = np.stack(cols, 1).astype(np.float32)
            good = np.isfinite(x).all(1) & np.isfinite(c["tq_norm"][idx])
            xs.append(x[good])
            ys.append(c["tq_norm"][idx][good].astype(np.float32))
            names += [sg.route] * int(good.sum())
            times.append(sg.t[idx][good])
    return np.concatenate(xs), np.concatenate(ys), np.array(names), np.concatenate(times)


def angle_zero(x: np.ndarray, features: list[str]) -> np.ndarray:
    """조향각 ≈ a·la/v² + b·la + c(차량 모델 꼴)를 맞춘 (a, b, c). c가 la = 0일 때의 각도(영점)다."""
    v, la = x[:, features.index("v")], x[:, features.index("la")]
    a = np.stack([la / np.maximum(v, 3.0) ** 2, la, np.ones(len(la))], 1)
    return np.linalg.lstsq(a, x[:, features.index("angle")], rcond=None)[0]


def mirrored(x: np.ndarray, y: np.ndarray, features: list[str], torque_bias: float):
    """좌우를 뒤집은 표본을 붙인다(차의 조향 반응은 좌우가 같다). la·jerk·roll은 부호를 바꾸고, 조향각은 영점 기준으로,
    토크는 영점 바이어스(직진에 드는 토크, 도로 횡단 경사 등) 기준으로 뒤집는다. 속도·종가속은 그대로."""
    xm = x.copy()
    neg = [i for i, f in enumerate(features) if f.startswith(("la", "jerk", "roll"))]
    xm[:, neg] *= -1
    ang = [i for i, f in enumerate(features) if f.startswith("angle")]
    if ang:
        zero = angle_zero(x, features)[2]
        xm[:, ang] = 2 * zero - x[:, ang]
    return np.concatenate([x, xm]), np.concatenate([y, 2 * torque_bias - y])
