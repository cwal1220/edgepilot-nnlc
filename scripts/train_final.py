"""고른 설정을 전체 주행으로 학습해 edgepilot 모델 파일로 내보내고, 정상 상태 곡선을 보인다.

  python scripts/train_final.py <csv dir> <모델 json> [설정 이름(기본 nnlc.configs.CHOSEN)] [--no-cv]

- 망은 설정의 ensemble 수만큼 시드마다 하나씩 학습해 앙상블로 넣는다. 교차 검증도 같은 수로 한다.
- 운용 입력으로 배우는 설정(Cfg.inputs "runtime")은 경로 CSV 옆에 <경로>.inputs.csv.gz가 있어야 한다
  (scripts/dump_runtime_inputs.sh). 설정의 curve_gain을 모델 파일에 쓴다.
- 내보낼 수 없는 설정(sunnypilot 꼴 등)은 학습 전에 멈춘다.
- 정상 상태 곡선: 일정한 커브(창 전체가 같은 횡가속, 저크·롤·종가속 0)에서 필요한 토크. 조향각 입력은 데이터에서 맞춘
  조향각 = a·la/v² + b·la + c로 만든다. 좌우 대칭과 단조 증가를 눈으로 확인한다.
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnlc import export  # noqa: E402
from nnlc.configs import CHOSEN, CONFIGS  # noqa: E402
from nnlc.models import torqued_like  # noqa: E402
from nnlc.training import DEVICE, cross_validate, dataset, evaluate, train  # noqa: E402

PROBE_LA = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]


def steady_inputs(features: list[str], v: float, la: float, angle_model: dict | None) -> list[float]:
    row = dict.fromkeys(features, 0.0)
    row["v"] = v
    for f in features:
        if f.startswith("la"):
            row[f] = la
        if f.startswith("angle") and angle_model is not None:
            row[f] = angle_model["a"] * la / v ** 2 + angle_model["b"] * la + angle_model["c"]
    return [row[f] for f in features]


def environment() -> dict:
    """모델을 만든 환경(모델 파일 training.environment): 같은 결과를 다시 내려면 필요하다."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"nnlc": commit, "device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("config", nargs="?", default=CHOSEN)
    ap.add_argument("--no-cv", action="store_true", help="do not recompute the cross-validation figures")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    export.check_exportable(cfg)
    x, y, names, _ = dataset(str(args.csv_dir), cfg.spec, cfg.inputs)
    trained = train(cfg, x, y)
    cv = None if args.no_cv else cross_validate(args.csv_dir, cfg)
    meta = {"training": {"config": args.config, "inputs": cfg.inputs, "samples": int(len(x)),
                         "routes": sorted(set(names.tolist())), "nets": cfg.ensemble, "cv": cv,
                         "environment": environment()}}
    model = export.to_json(trained, cfg.spec, args.out.stem, x, meta, curve_gain=cfg.curve_gain)
    export.save(model, args.out)
    # 내보낸 파일이 학습한 망과 같은 값을 내는지
    reloaded = export.load(args.out)
    probe_x = x[:: max(1, len(x) // 2000)]
    diff = np.abs(export.evaluate(reloaded, probe_x) - np.clip(evaluate(trained, probe_x), -1, 1)).max()
    print(f"{args.out}: {len(x)} samples, {cfg.ensemble} nets, export max diff {diff:.2e}")
    if cv:
        print(f"CV RMSE {cv['model']:.4f} ({cv['gain']:+.1%} vs linear), |la|>=1 {cv['model_10']:.4f} "
              f"({cv['gain_10']:+.1%})")
    print("steady-curve torque (normalised), columns: lateral accel m/s²")
    print(f"{'':12s}" + "".join(f"{l:>7}" for l in PROBE_LA))
    for v in (10, 20, 30):
        rows = np.array([steady_inputs(trained.features, v, la, model["angle_model"]) for la in PROBE_LA])
        nn_out = export.evaluate(reloaded, rows)
        lin_out = torqued_like(rows) @ trained.linear
        print(f"nn  v={v:2d}    " + "".join(f"{q:7.3f}" for q in nn_out))
        print(f"lin v={v:2d}    " + "".join(f"{q:7.3f}" for q in lin_out))


if __name__ == "__main__":
    main()
