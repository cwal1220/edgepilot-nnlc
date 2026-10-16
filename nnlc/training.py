"""학습과 주행 단위 교차 검증.

- Cfg 하나가 입력(Spec), 모델, 학습 길이, 앙상블 크기, 증강을 정한다.
- 교차 검증은 경로를 날짜순으로 이어 표본 수가 비슷한 다섯 묶음(folds)으로 나누고 차례로 빼서 학습한다. epoch 수를
  고정하고 검증 fold로 고르지 않는다. 정규화, 가중치, 선형 기준 모델은 학습 fold에서만 구한다. 망 수는 내보내는 모델과
  같다(Cfg.ensemble).
- 지표(fold 평균): 전체 RMSE와 |la| ≥ 0.5, ≥ 1.0의 RMSE, 그리고 같은 fold의 torqued식 선형 대비 개선.
"""
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnlc.dataset import Spec, build, mirrored
from nnlc.models import Mlp, SpNet, fit_linear, torqued_like

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LA_EDGES = [0, 0.25, 0.5, 1.0, 1.5, 99]
N_FOLDS = 5


@dataclass(frozen=True)
class Cfg:
    spec: Spec = Spec()
    model: str = "mlp"          # mlp | sp(sunnypilot 꼴, 비교용이라 내보내지 않는다)
    hidden: tuple = (32, 32)
    epochs: int = 40
    lr: float = 3e-3
    ensemble: int = 1           # 시드마다 하나씩 학습해 평균하는 망 수
    mirror: bool = False        # 좌우 대칭 증강(학습 fold에만)


def make_weights(la: np.ndarray) -> np.ndarray:
    """|la| 구간마다 드문 구간일수록 큰 표본 가중치(최대 20배). 평균 1로 맞춘다."""
    b = np.digitize(np.abs(la), LA_EDGES) - 1
    counts = np.bincount(b, minlength=len(LA_EDGES) - 1).astype(np.float64)
    w = np.minimum(counts.max() / np.maximum(counts[b], 1.0), 20.0)
    return (w / w.mean()).astype(np.float32)


def make_net(cfg: Cfg, n_in: int) -> nn.Module:
    return SpNet() if cfg.model == "sp" else Mlp(n_in, cfg.hidden)


def fit(cfg: Cfg, x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> nn.Module:
    """정규화된 x로 학습한다(AdamW, 코사인 학습률, 배치 8192, 고정 epoch)."""
    torch.manual_seed(seed)
    net = make_net(cfg, x.shape[1]).to(DEVICE)
    xd, yd, wd = (torch.from_numpy(a).to(DEVICE) for a in (x, y, w))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    n, batch = len(x), 8192
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            err = net(xd[j])[:, 0] - yd[j]
            loss = (wd[j] * err ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net.eval().cpu()


def predict(nets: list[nn.Module], x: np.ndarray) -> np.ndarray:
    xd = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    with torch.no_grad():
        return np.mean([net(xd)[:, 0].numpy() for net in nets], 0)


def rmse(la: np.ndarray, y: np.ndarray, p: np.ndarray, lo: float = 0.0) -> float:
    m = np.abs(la) >= lo
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.sum() >= 50 else float("nan")


@functools.lru_cache(maxsize=None)
def dataset(csv_dir: str, spec: Spec):
    return build(Path(csv_dir), spec)


@dataclass
class Trained:
    nets: list
    mean: np.ndarray
    std: np.ndarray
    features: list
    linear: np.ndarray  # torqued식 선형 계수(a, b, c, d); d가 직진 토크 바이어스


def train(cfg: Cfg, x: np.ndarray, y: np.ndarray) -> Trained:
    """x, y(원래 단위)로 학습한다. 가중치와 선형 기준은 증강 전 원본에서, 정규화는 증강한 학습 데이터에서 구한다."""
    features = cfg.spec.names()
    w = make_weights(x[:, 1])
    lin = fit_linear(torqued_like(x), y, w)
    xs, ys, ws = x, y, w
    if cfg.mirror:
        xs, ys = mirrored(x, y, features, float(lin[3]))
        ws = np.concatenate([w, w])
    mean, std = xs.mean(0), xs.std(0) + 1e-6
    xn = ((xs - mean) / std).astype(np.float32)
    nets = [fit(cfg, xn, ys, ws, seed) for seed in range(cfg.ensemble)]
    return Trained(nets, mean, std, features, lin)


def evaluate(t: Trained, x: np.ndarray) -> np.ndarray:
    return predict(t.nets, (x - t.mean) / t.std)


def folds(names: np.ndarray, n: int = N_FOLDS) -> list[list[str]]:
    """경로를 이름(날짜) 순으로 이어 표본 수가 비슷한 n 묶음으로 나눈다. 경로마다 표본 누적 비율의 가운데가 들어가는
    구간이 그 묶음이다. 묶음이 비면 멈춘다(경로가 모자라다)."""
    routes, counts = np.unique(names, return_counts=True)
    middle = (np.cumsum(counts) - counts / 2) / counts.sum()
    group = np.minimum((middle * n).astype(int), n - 1)
    out = [routes[group == k].tolist() for k in range(n)]
    if any(not fold for fold in out):
        raise ValueError(f"{len(routes)} routes are too few for {n} folds")
    return out


def cross_validate(csv_dir: Path, cfg: Cfg) -> dict:
    x, y, names, _ = dataset(str(csv_dir), cfg.spec)
    res = {k: [] for k in ("base", "model", "base_05", "model_05", "base_10", "model_10")}
    for val in folds(names):
        vm = np.isin(names, val)
        t = train(cfg, x[~vm], y[~vm])
        xv, yv = x[vm], y[vm]
        pb = torqued_like(xv) @ t.linear
        pm = evaluate(t, xv)
        for lo, sfx in ((0.0, ""), (0.5, "_05"), (1.0, "_10")):
            res["base" + sfx].append(rmse(xv[:, 1], yv, pb, lo))
            res["model" + sfx].append(rmse(xv[:, 1], yv, pm, lo))
    out = {k: float(np.nanmean(v)) for k, v in res.items()}
    for sfx in ("", "_05", "_10"):
        out["gain" + sfx] = 1 - out["model" + sfx] / out["base" + sfx]
    return out
