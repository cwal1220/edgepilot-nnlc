"""edgepilot `extract_lateral_dataset` CSV를 읽는다(경로마다 하나, 행 = ControlState 하나, 약 100 Hz). 열 정의는
docs/data-format.md. gzip으로 묶은 <경로>.csv.gz도 그대로 읽는다(저장소의 dataset/).

처음 읽을 때 <경로>.npy(문자열 열 block을 뺀 float32)로 저장해 다음부터 빨리 읽는다. 머리줄이 COLS와 다르면
(추출기의 열이 바뀌었으면) 다시 추출하라고 멈춘다. 열을 자리로 읽으므로 이름이 밀리면 값이 엉뚱한 열로 간다.
"""
import gzip
from pathlib import Path

import numpy as np

# CSV 열(4번 block은 문자열이라 읽지 않는다)
COLS = ["t", "v", "active", "desire", "angle", "driver_tq", "apply_tq", "tq_norm", "des_tq_norm", "lat", "lat_valid",
        "yaw", "long", "bank", "roll", "des_curv", "act_curv", "des_la", "la_p03", "la_p06", "la_p10", "la_p15",
        "roll_p03", "roll_p06", "roll_p10", "roll_p15", "model_age", "toi_active", "steer_fault", "blinker",
        "eps_out_tq"]
HEADER = COLS[:4] + ["block"] + COLS[4:]
USECOLS = [i for i in range(len(HEADER)) if i != 4]


def route_name(csv: Path) -> str:
    """<경로>.csv 또는 <경로>.csv.gz의 경로 이름."""
    name = csv.name
    return name[:-len(".csv.gz")] if name.endswith(".csv.gz") else csv.stem


def route_files(csv_dir: Path) -> list[Path]:
    """폴더의 경로 CSV(.csv, .csv.gz), 이름순. 같은 경로가 둘 다 있으면 .csv를 쓴다. 옆의 운용 입력 파일
    (<경로>.inputs.csv, nnlc.runtime)은 경로가 아니다."""
    files = {}
    for path in sorted(Path(csv_dir).glob("*.csv.gz")) + sorted(Path(csv_dir).glob("*.csv")):
        if route_name(path).endswith(".inputs"):
            continue
        files[route_name(path)] = path
    return [files[k] for k in sorted(files)]


def load(csv: Path) -> dict[str, np.ndarray]:
    opener = gzip.open if csv.name.endswith(".gz") else open
    with opener(csv, "rt") as f:
        header = f.readline().strip().split(",")
    if header != HEADER:
        raise ValueError(f"{csv}: columns differ from nnlc.csvdata.COLS, re-extract with the current extractor")
    cache = csv.with_name(route_name(csv) + ".npy")
    a = np.load(cache) if cache.exists() and cache.stat().st_mtime > csv.stat().st_mtime else None
    if a is None or a.ndim != 2 or a.shape[1] != len(COLS):
        a = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=USECOLS, dtype=np.float32, ndmin=2)
        if a.shape[1] != len(COLS):  # 행이 없는 파일(머리줄만)
            a = np.zeros((0, len(COLS)), np.float32)
        np.save(cache, a)
    return {name: a[:, i] for i, name in enumerate(COLS)}


def usable_mask(d: dict[str, np.ndarray], driver_thr: float = 100.0) -> np.ndarray:
    """학습에 쓸 행: 조향 활성 + EPS가 LKAS 토크를 받는 중 + EPS 고장 아님 + 차선 변경·깜빡이 없음 +
    운전자 토크 |driver_tq| < 문턱 + 속도 > 3 m/s + ESP12 횡가속 유효 + 모델 계획이 0.2 s 안에 있음."""
    return ((d["active"] == 1) & (d["toi_active"] == 1) & (d["steer_fault"] == 0) & (d["desire"] == 0) &
            (d["blinker"] == 0) & (np.abs(d["driver_tq"]) < driver_thr) & (d["v"] > 3.0) & (d["lat_valid"] == 1) &
            (d["model_age"] >= 0) & (d["model_age"] < 0.2) & np.isfinite(d["la_p15"]))
