"""경로마다 시간과, 학습에 쓸 만한 표본이 속도 × |횡가속| 격자에 얼마나 있는지 센다.

  python scripts/coverage.py <csv dir> [운전자 토크 문턱(기본 100)]

부호 확인도 한다: 보낸 토크와 횡가속(쓸 수 있는 행), 보낸 토크와 MDPS가 실제로 낸 토크(EPS가 받는 행)의 상관은
양수여야 한다. 음수면 추출기의 부호나 MDPS12 필드가 틀린 것이라 경고한다. 횡가속은 제어 부호(오른쪽이 양수)다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nnlc.csvdata import load, usable_mask  # noqa: E402

V_EDGES = [3, 10, 15, 20, 25, 30, 40]  # m/s
LA_EDGES = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 99]  # m/s²
TQ_EDGES = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.01]


def main():
    root = Path(sys.argv[1])
    driver_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    grid = np.zeros((len(V_EDGES) - 1, len(LA_EDGES) - 1))
    side, tq_hist = np.zeros(2), np.zeros(len(TQ_EDGES) - 1)
    total = np.zeros(4)
    print(f"{'route':26s} {'hours':>6s} {'active':>7s} {'EPS on':>7s} {'usable':>7s} {'override%':>9s} "
          f"{'tq~la':>6s} {'tq~eps':>6s}")
    suspicious = []
    for csv in sorted(root.glob("*.csv")):
        d = load(csv)
        t = d["t"]
        if len(t) < 100:
            continue
        dt = np.diff(t, prepend=t[0])
        dt = np.where((dt > 0) & (dt < 0.2), dt, 0.0)  # 끊긴 곳은 세지 않는다
        act = d["active"] == 1
        eps = act & (d["toi_active"] == 1)
        use = usable_mask(d, driver_thr)
        override = act & (np.abs(d["driver_tq"]) >= driver_thr)
        hours = np.array([dt.sum(), dt[act].sum(), dt[eps].sum(), dt[use].sum()]) / 3600
        total += hours
        la = d["act_curv"] * d["v"] ** 2
        out_ok = eps & (d["eps_out_tq"] != 0)  # 0은 MDPS12가 낡았을 때
        corr = [np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan
                for a, b, m in ((d["tq_norm"], la, use), (d["apply_tq"], d["eps_out_tq"], out_ok))]
        if any(c <= 0 for c in corr):
            suspicious.append(csv.stem)
        print(f"{csv.stem:26s} {hours[0]:6.2f} {hours[1]:7.2f} {hours[2]:7.2f} {hours[3]:7.2f} "
              f"{100 * dt[override].sum() / max(dt[act].sum(), 1e-9):9.1f} {corr[0]:+6.2f} {corr[1]:+6.2f}")
        vi, li = np.digitize(d["v"], V_EDGES) - 1, np.digitize(np.abs(la), LA_EDGES) - 1
        ok = use & (vi >= 0) & (vi < grid.shape[0]) & (li >= 0) & (li < grid.shape[1])
        np.add.at(grid, (vi[ok], li[ok]), dt[ok])
        side += [dt[use & (la > 0.5)].sum(), dt[use & (la < -0.5)].sum()]
        ti = np.clip(np.digitize(np.abs(d["tq_norm"]), TQ_EDGES) - 1, 0, len(tq_hist) - 1)
        np.add.at(tq_hist, ti[use], dt[use])
    print(f"{'total':26s} {total[0]:6.2f} {total[1]:7.2f} {total[2]:7.2f} {total[3]:7.2f}")
    print(f"\n쓸 수 있는 표본(분), 속도(m/s) × |횡가속|(m/s²), 운전자 토크 문턱 {driver_thr:.0f}:")
    print(f"{'v / |la|':>9s} " + " ".join(f"{f'{a}-{b}' if b < 99 else f'{a}+':>8s}" for a, b in zip(LA_EDGES, LA_EDGES[1:])))
    for i, (a, b) in enumerate(zip(V_EDGES, V_EDGES[1:])):
        print(f"{f'{a}-{b}':>9s} " + " ".join(f"{m:8.1f}" for m in grid[i] / 60))
    print(f"\n|횡가속| > 0.5에서 오른쪽 {side[0] / 60:.0f}분, 왼쪽 {side[1] / 60:.0f}분")
    print("|토크 명령| 분포(분):", " ".join(f"{a}-{b}: {m / 60:.0f}" for a, b, m in zip(TQ_EDGES, TQ_EDGES[1:], tq_hist)))
    if suspicious:
        print("\n경고: 토크 부호 상관이 0 이하인 경로:", ", ".join(suspicious))


if __name__ == "__main__":
    main()
