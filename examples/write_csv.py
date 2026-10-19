"""다른 기록에서 nnlc CSV 만들기(docs/data-format.md, docs/other-cars.md).

edgepilot 녹화가 아니면 extract_lateral_dataset을 쓸 수 없다. 자기 로거나 openpilot rlog에서 열을 뽑아 이 형식으로
쓰면 같은 학습을 돌릴 수 있다. write_drive_csv가 형식(머리줄, 부호, 기본값)을 맞추고 검사한다.

  python examples/write_csv.py --demo <출력 dir>   # 합성 주행 하나를 쓰고 nnlc로 표본까지 만들어 본다
"""
import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnlc.csvdata import COLS, HEADER  # noqa: E402

# 꼭 있어야 하는 열(단위·부호는 docs/data-format.md). 곡률과 횡가속, 롤은 오른쪽이 +, 조향각과 원시 토크는 왼쪽이 +.
REQUIRED = ("t", "v", "active", "angle", "driver_tq", "apply_tq", "des_curv", "act_curv", "long", "roll", "model_age")
# 없으면 채우는 값: 학습 조건을 막지 않는 값(지시등·desire 없음, EPS가 토크를 받음, 고장 없음, 횡가속 유효, 학습
# 모델 없이 달림)
DEFAULTS = {"desire": 0, "blinker": 0, "toi_active": 1, "steer_fault": 0, "lat_valid": 1, "lat": 0.0, "yaw": 0.0,
            "eps_out_tq": 0, "nn_ff": 0, "bank": None, "des_la": None, "des_tq_norm": None, "tq_norm": None}
FUTURE = ((0.3, "03"), (0.6, "06"), (1.0, "10"), (1.5, "15"))


def write_drive_csv(path: Path, cols: dict[str, np.ndarray], steer_max: float, block: str = "-") -> int:
    """cols(같은 길이의 배열, 행 = 제어 틱)로 주행 하나의 CSV를 쓴다. path가 .gz로 끝나면 gzip으로. 쓴 행 수."""
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    n = len(cols["t"])
    out = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    if any(len(v) != n for v in out.values()):
        raise ValueError("all columns need the same length")
    if np.any(np.diff(out["t"]) <= 0):
        raise ValueError("t must increase")
    v = out["v"]
    for key, value in DEFAULTS.items():
        if key not in out and value is not None:
            out[key] = np.full(n, float(value))
    out.setdefault("tq_norm", -out["apply_tq"] / steer_max)  # 원시 토크는 왼쪽 +, tq_norm은 오른쪽 +
    out.setdefault("des_tq_norm", out["tq_norm"])
    out.setdefault("bank", out["roll"] * 9.81)
    out.setdefault("des_la", out["des_curv"] * v * v)
    for dt, tag in FUTURE:  # 계획이 없으면 목표 곡률을 그 시각만큼 앞에서 읽는다
        out.setdefault(f"la_p{tag}", np.interp(out["t"] + dt, out["t"], out["des_curv"] * v * v))
        out.setdefault(f"roll_p{tag}", out["roll"])
    ints = {"active", "desire", "lat_valid", "toi_active", "steer_fault", "blinker", "driver_tq", "apply_tq",
            "eps_out_tq", "nn_ff"}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        f.write(",".join(HEADER) + "\n")
        for i in range(n):
            vals = [str(int(round(out[c][i]))) if c in ints else f"{out[c][i]:.7g}" for c in COLS]
            f.write(",".join(vals[:4] + [block] + vals[4:]) + "\n")
    return n


def demo_drive(seconds: float = 120.0, steer_max: float = 384.0) -> dict[str, np.ndarray]:
    """합성 주행: 18 m/s, 좌우 곡선이 번갈아 오고, 토크 = la/4 + 작은 마찰, 차는 0.3 s 늦게 따른다."""
    t = np.arange(0.0, seconds, 0.01)
    v = np.full(len(t), 18.0)
    des_curv = 0.004 * np.sin(2 * np.pi * t / 20.0) * (np.sin(2 * np.pi * t / 7.0) > -0.3)
    act_curv = np.interp(t - 0.3, t, des_curv)
    la = act_curv * v * v
    torque = la / 4.0 + 0.03 * np.tanh(np.gradient(la, t) / 0.3)  # 오른쪽 +, steer_max로 나눈 값
    return {"t": t, "v": v, "active": np.ones(len(t)), "angle": -np.degrees(act_curv * 16.3 * 2.855),
            "driver_tq": np.zeros(len(t)), "apply_tq": -torque * steer_max, "des_curv": des_curv,
            "act_curv": act_curv, "long": np.zeros(len(t)), "roll": np.zeros(len(t)),
            "model_age": np.full(len(t), 0.05), "lat": la}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, required=True, help="folder to write the synthetic drive CSVs into")
    args = ap.parse_args()
    args.demo.mkdir(parents=True, exist_ok=True)
    path = args.demo / "2026-01-01--00-00-00-000.csv.gz"
    rows = write_drive_csv(path, demo_drive(), steer_max=384.0)
    from nnlc.csvdata import load, usable_mask
    from nnlc.dataset import CHOSEN, build
    d = load(path)
    x, y, _, _ = build(args.demo, CHOSEN)
    print(f"{path}: {rows} rows, {int(usable_mask(d).sum())} usable, {len(y)} samples of {x.shape[1]} inputs")


if __name__ == "__main__":
    main()
