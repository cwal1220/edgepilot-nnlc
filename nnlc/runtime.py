"""계획 조건 학습 표본(k7_hev_v4부터): 입력은 edgepilot이 그 순간 학습 모델에 넣었을 값(운용 입력), 목표는 그 틱에
보낸 토크다.

- 실제 궤적 학습(dataset.build)은 차가 실제로 간 궤적을 입력으로 쓰고, 운용에서는 그 자리에 목표 궤적(계획과 지난
  목표)이 들어간다. 둘은 곡선에서 다르다: 진입에서 계획 창은 실제보다 평평하고(가까운 쪽 12 % 크고 먼 쪽 5~7 % 작다),
  저크 입력은 실제와 상관이 0.17뿐이다. 그래서 실제 궤적으로 배운 모델이 운용에서 보는 입력은 학습에 없던 꼴이다. 이
  표본은 운용과 같은 입력으로 배운다(되먹임 오차 학습: 목표인 보낸 토크에 그때 PID가 더한 몫이 들어 있다).
- 운용 입력은 경로마다 <경로>.inputs.csv(.gz)다(scripts/dump_runtime_inputs.sh, edgepilot replay_controls
  --nn-inputs): t(초, 경로 CSV의 t와 같은 기준)와 입력 이름 열(k7_hev_v3의 33개, 20 Hz). 설정의 입력은 이름으로
  고른다.
- 목표와 표본 조건은 dataset.build와 같다: 보낸 토크(friction_free면 운용이 더할 마찰 몫을 뺀 것), 표본 시각과 응답
  시각이 usable이고 창 안에 차선 변경·깜빡이가 없을 것. 운용 입력 행마다 경로 CSV의 가장 가까운 100 Hz 격자 점(6 ms
  안)을 쓴다.
- 경로마다 두 시간축이 맞는지 본다: 두 파일에 같은 ESP12 종가속(입력 long, CSV long)이 있어 상관이 가장 큰 어긋남이
  0이어야 한다. ALIGN_LIMIT_S를 넘으면 그 경로를 빼고 알린다(녹화의 첫 기록 시각을 두 도구가 다르게 잡으면 초 단위로
  어긋난다). K7 데이터(2026-10-19 확인)는 행이 있는 25경로 모두 0~10 ms였다.
"""
import gzip
import sys
import warnings
from pathlib import Path

import numpy as np

from nnlc.csvdata import route_files, route_name
from nnlc.dataset import Spec, _steps
from nnlc.series import DT, segments

INPUTS_SUFFIX = ".inputs.csv"
MATCH_S = 0.006
ALIGN_LIMIT_S = 0.03
ALIGN_SEARCH_S = 1.5
ALIGN_COARSE_S = 0.05


def inputs_file(csv: Path) -> Path | None:
    """경로 CSV 옆의 운용 입력 파일(.inputs.csv를 .inputs.csv.gz보다 먼저). 없으면 None."""
    for suffix in (INPUTS_SUFFIX, INPUTS_SUFFIX + ".gz"):
        path = csv.with_name(route_name(csv) + suffix)
        if path.exists():
            return path
    return None


def load_inputs(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(t [n] float64, 입력 이름, 입력 [n, f] float32). 처음 읽을 때 <경로>.inputs.npy로 저장해 다음부터 빨리 읽는다."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        header = f.readline().strip().split(",")
    if not header or header[0] != "t" or len(header) < 2:
        raise ValueError(f"{path}: not a runtime input file (first column must be t)")
    name = path.name[:-len(".gz")] if path.name.endswith(".gz") else path.name
    cache = path.with_name(name[:-len(".csv")] + ".npy")
    a = np.load(cache) if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime else None
    if a is None or a.ndim != 2 or a.shape[1] != len(header):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 행이 없는 파일(모델을 한 번도 계산하지 않은 경로)
            a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        if a.shape[1] != len(header):
            a = np.zeros((0, len(header)))
        np.save(cache, a)
    return a[:, 0], header[1:], a[:, 1:].astype(np.float32)


def alignment_s(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """구간마다 (CSV 열의 100 Hz 격자, 입력 행의 격자 번호, 그 행의 같은 신호)에서 상관이 가장 큰 어긋남(초, +면
    CSV 쪽이 그만큼 뒤에서 가장 닮았다). ALIGN_COARSE_S 간격으로 찾은 뒤 그 둘레를 10 ms로 다시 본다. 표본이
    모자라면 0."""
    def corr(shift: int) -> float:
        xs, ys = [], []
        for column, k, value in pairs:
            j = k + shift
            ok = (j >= 0) & (j < len(column))
            xs.append(value[ok])
            ys.append(column[j[ok]])
        x, y = np.concatenate(xs), np.concatenate(ys)
        good = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[good], y[good])[0, 1]) if good.sum() >= 500 else np.nan

    if not pairs or sum(len(k) for _, k, _ in pairs) < 500:
        return 0.0
    reach, step = int(round(ALIGN_SEARCH_S / DT)), int(round(ALIGN_COARSE_S / DT))
    coarse = max(range(-reach, reach + 1, step), key=lambda s: np.nan_to_num(corr(s), nan=-2.0))
    fine = max(range(coarse - step, coarse + step + 1), key=lambda s: np.nan_to_num(corr(s), nan=-2.0))
    return fine * DT


def build_runtime(csv_dir: Path, spec: Spec):
    """폴더의 경로마다 운용 입력 파일이 있으면 (X [n, f] float32, y [n], 경로 이름 [n], 표본 시각 [n]). 입력 파일이
    없거나 시간축이 어긋난 경로는 뺀다."""
    lag = int(round(spec.lag_s / DT))
    every = (_steps(spec.la_past) + _steps(spec.la_future) + _steps(spec.roll_past) + _steps(spec.roll_future) +
             [o for _, offs in spec.extra_windows for o in _steps(offs)] + [0])
    lo, hi = min(every), max(every)
    wanted = spec.names()
    xs, ys, names, times = [], [], [], []
    for csv in route_files(Path(csv_dir)):
        path = inputs_file(csv)
        if path is None:
            continue
        t_in, columns, x_in = load_inputs(path)
        missing = [n for n in wanted + ["long"] if n not in columns]
        if missing:
            raise ValueError(f"{path}: no inputs {missing}")
        long_in = x_in[:, columns.index("long")]
        x_in = x_in[:, [columns.index(n) for n in wanted]]
        found = []  # 구간마다 (구간, 격자 번호, 입력 행)
        for sg in segments(csv, spec.driver_thr):
            k = np.round((t_in - sg.t[0]) / DT).astype(np.int64)
            inside = (k >= 0) & (k < len(sg.t))
            rows = np.flatnonzero(inside)
            rows = rows[np.abs(sg.t[k[rows]] - t_in[rows]) <= MATCH_S]
            if len(rows):
                found.append((sg, k[rows], rows))
        if not found:
            continue
        shift = alignment_s([(sg.cols["long"], k, long_in[rows]) for sg, k, rows in found])
        if abs(shift) > ALIGN_LIMIT_S:
            print(f"{path.name}: runtime inputs are {shift:+.2f} s off the route's clock, skipped", file=sys.stderr)
            continue
        for sg, k, rows in found:
            c = sg.cols
            n = len(sg.t)
            usable = c["usable"] > 0
            ev_cum = np.r_[0, np.cumsum((c["desire"] != 0) | (c["blinker"] != 0))]
            keep = (k >= max(0, -(lag + lo))) & (k < n - (lag + hi))
            k, rows = k[keep], rows[keep]
            r = k + lag
            ok = usable[k] & usable[r] & (ev_cum[r + hi + 1] - ev_cum[r + lo] == 0)
            k, rows = k[ok], rows[ok]
            if len(k) == 0:
                continue
            x = x_in[rows]
            y = c["tq_norm"][k] - (c["curve_w"][k] * c["fric"][k] if spec.friction_free else 0.0)
            good = np.isfinite(x).all(1) & np.isfinite(y)
            xs.append(x[good])
            ys.append(y[good].astype(np.float32))
            names += [sg.route] * int(good.sum())
            times.append(sg.t[k][good])
    if not xs:
        raise ValueError(f"{csv_dir}: no usable samples with runtime inputs (scripts/dump_runtime_inputs.sh)")
    return np.concatenate(xs), np.concatenate(ys), np.array(names), np.concatenate(times)
