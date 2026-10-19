"""CSV를 100 Hz 고른 격자의 연속 구간으로 나눈다.

- 행 간격이 0.1 s를 넘으면 구간을 끊는다. 구간 안에서는 열마다 선형 보간한다(플래그 열은 격자 시각 또는 그 뒤의 첫
  행 값: 최대 한 행(10 ms) 앞의 값을 쓴다).
- la = act_curv · v²: 제어기의 실제 횡가속. 제어기는 조향각을 차량 모델로 곡률로 바꾼다(상류 latcontrol_torque와 같다).
- jerk = la의 ±0.05 s 중앙차분.
- fric, curve_w: edgepilot이 그 순간 모델에 더할 마찰 보상 몫과 곡선 가중(nnlc.friction). friction_free 학습이 쓴다.
- 경로·운전자 토크 문턱마다 결과를 기억해 둔다(실험에서 같은 경로를 여러 번 쓴다). 26개 경로에 약 0.6 GB.
"""
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nnlc.csvdata import load, route_name, usable_mask
from nnlc.friction import curve_weight, friction_share

DT = 0.01
GAP_S = 0.1
FLAG_COLS = ("active", "desire", "toi_active", "steer_fault", "blinker", "lat_valid", "usable")


@dataclass
class Segment:
    route: str
    t: np.ndarray
    cols: dict[str, np.ndarray]


@functools.lru_cache(maxsize=None)
def segments(csv: Path, driver_thr: float = 100.0, min_len_s: float = 3.0) -> list[Segment]:
    d = load(csv)
    if len(d["t"]) < 2:
        return []
    t = d["t"].astype(np.float64)
    order = np.argsort(t, kind="stable")
    d = {k: v[order] for k, v in d.items()}
    t = t[order]
    d["usable"] = usable_mask(d, driver_thr).astype(np.float32)
    cuts = np.flatnonzero(np.diff(t) > GAP_S) + 1
    out = []
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if t[e - 1] - t[s] < min_len_s:
            continue
        grid = np.arange(t[s], t[e - 1], DT)
        cols = {}
        for k, v in d.items():
            seg = v[s:e]
            if k in FLAG_COLS:
                idx = np.clip(np.searchsorted(t[s:e], grid), 0, e - s - 1)
                cols[k] = seg[idx]
            else:
                ok = np.isfinite(seg)
                cols[k] = (np.interp(grid, t[s:e][ok], seg[ok]) if ok.sum() > 1
                           else np.full(len(grid), np.nan)).astype(np.float32)
        v = cols["v"]
        cols["la"] = cols["act_curv"] * v * v
        cols["jerk"] = jerk_of(cols["la"])
        cols["fric"] = friction_share(cols["des_curv"], cols["act_curv"], v)
        cols["curve_w"] = curve_weight(cols["des_curv"], v)
        out.append(Segment(route_name(csv), grid, cols))
    return out


def jerk_of(la: np.ndarray) -> np.ndarray:
    """la의 ±0.05 s 중앙차분(양 끝 5점은 0). edgepilot 운용 입력도 같은 식이다."""
    jerk = np.zeros_like(la)
    jerk[5:-5] = (la[10:] - la[:-10]) / (10 * DT)
    return jerk
