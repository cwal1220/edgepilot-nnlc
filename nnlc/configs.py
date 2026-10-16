"""비교 실험의 설정 목록(2026-10-16 실험 순서대로). 결과는 README의 표에 있다.

CHOSEN은 최종 후보: 입력 33개(조향각 창, 촘촘한 횡가속 창, 종가속), 32×32 tanh, 좌우 대칭 증강.
"""
from dataclasses import replace

from nnlc.dataset import CHOSEN as CHOSEN_SPEC
from nnlc.dataset import Spec
from nnlc.training import Cfg

BASE = Cfg()
_DENSE = dict(la_past=(-0.5, -0.4, -0.3, -0.2, -0.1), la_future=(0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5))
_ANGLE = replace(BASE.spec, extra=("angle",))
_ANGLE_WIN = (("angle", (-0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5)),)
_LONG_T = tuple(round(0.1 * k, 1) for k in range(-5, 16) if k != 0)  # -0.5..+1.5 s, 0.1 s 간격


def _spec(**kw) -> Spec:
    return replace(BASE.spec, **kw)


A = replace(BASE, spec=CHOSEN_SPEC)
_B_SPEC = replace(CHOSEN_SPEC, lag_s=0.2, la_past=tuple(t for t in _LONG_T if t < 0),
                  la_future=tuple(t for t in _LONG_T if t > 0), extra_windows=(("angle", _LONG_T),))
B = replace(A, spec=_B_SPEC, hidden=(64, 64, 64))

CONFIGS = {
    # 1차: 모델·손실·데이터·입력 하나씩
    "base mlp32 tanh": BASE,
    "sp (sunnypilot)": replace(BASE, model="sp", epochs=60),
    "mlp64": replace(BASE, hidden=(64, 64)),
    "mlp64x3": replace(BASE, hidden=(64, 64, 64)),
    "silu": replace(BASE, act="silu"),
    "wide (linear+mlp)": replace(BASE, model="wide"),
    "huber": replace(BASE, loss="huber"),
    "weights none": replace(BASE, weights="none"),
    "weights sqrt": replace(BASE, weights="sqrt"),
    "wd 1e-3": replace(BASE, wd=1e-3),
    "ensemble 5": replace(BASE, ensemble=5),
    "lag 0.15": replace(BASE, spec=_spec(lag_s=0.15)),
    "stride 2 (50 Hz)": replace(BASE, spec=_spec(stride=2)),
    "driver thr 50": replace(BASE, spec=_spec(driver_thr=50.0)),
    "driver thr 200": replace(BASE, spec=_spec(driver_thr=200.0)),
    "window dense": replace(BASE, spec=_spec(**_DENSE)),
    "no past la": replace(BASE, spec=_spec(la_past=())),
    "+ long accel": replace(BASE, spec=_spec(extra=("long",))),
    "+ angle": replace(BASE, spec=_ANGLE),
    "+ driver_tq (0 at eval)": replace(BASE, spec=_spec(extra=("driver_tq",)), zero=("driver_tq",)),
    # 2차: 조향각 정보 쌓기, 깨끗한 데이터로 학습
    "thr50 train, thr100 eval": replace(BASE, spec=_spec(driver_thr=50.0), eval_thr=100.0),
    "+ angle window": replace(BASE, spec=replace(_ANGLE, extra_windows=_ANGLE_WIN)),
    "+ angle + long": replace(BASE, spec=replace(_ANGLE, extra=("angle", "long"))),
    "+ angle, dense": replace(BASE, spec=replace(_ANGLE, **_DENSE)),
    "+ angle + long, dense": replace(BASE, spec=replace(_ANGLE, extra=("angle", "long"), **_DENSE)),
    "A": A,  # = + angle win + long, dense
    "+ angle, thr50/100": replace(BASE, spec=replace(_ANGLE, driver_thr=50.0), eval_thr=100.0),
    # 3차: A를 바탕으로
    "A lag 0.2": replace(A, spec=replace(CHOSEN_SPEC, lag_s=0.2)),
    "A lag 0.4": replace(A, spec=replace(CHOSEN_SPEC, lag_s=0.4)),
    "A long windows": replace(A, spec=replace(CHOSEN_SPEC, la_past=tuple(t for t in _LONG_T if t < 0),
                                              la_future=tuple(t for t in _LONG_T if t > 0),
                                              extra_windows=(("angle", _LONG_T),))),
    "A mlp64x3": replace(A, hidden=(64, 64, 64)),
    "A ensemble 5": replace(A, ensemble=5),
    "A + jerk window": replace(A, spec=replace(CHOSEN_SPEC, extra_windows=_ANGLE_WIN + (("jerk", (-0.2, 0.2, 0.5)),))),
    # 4차: 묶기와 좌우 대칭 증강
    "B (lag .2, long win, 64x3)": B,
    "B ensemble 5": replace(B, ensemble=5),
    "B 60 epochs": replace(B, epochs=60),
    "base + mirror": replace(BASE, mirror=True),
    "A + mirror": replace(A, mirror=True),
    "A mlp64x3 + mirror": replace(A, hidden=(64, 64, 64), mirror=True),
}
CHOSEN = "A + mirror"
