"""설정마다 주행 단위 5-fold 교차 검증을 돌리고 결과를 JSON에 모은다.

  python scripts/experiments.py <csv dir> <결과 json> [설정 이름 ...]   (이름 없으면 nnlc.configs.CONFIGS 전부)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnlc.configs import CONFIGS  # noqa: E402
from nnlc.training import cross_validate  # noqa: E402


def main():
    csv_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(CONFIGS)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"{'config':28s} {'RMSE':>7s} {'gain':>6s} {'|la|>=.5':>8s} {'gain':>6s} {'|la|>=1':>8s} {'gain':>6s} "
          f"{'sec':>5s}")
    for name in names:
        t0 = time.time()
        r = cross_validate(csv_dir, CONFIGS[name])
        r["seconds"] = round(time.time() - t0, 1)
        results[name] = r
        print(f"{name:28s} {r['model']:7.4f} {r['gain']:+6.1%} {r['model_05']:8.4f} {r['gain_05']:+6.1%} "
              f"{r['model_10']:8.4f} {r['gain_10']:+6.1%} {r['seconds']:5.0f}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
