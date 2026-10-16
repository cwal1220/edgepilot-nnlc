"""학습과 주행 단위 교차 검증.

- Cfg 하나가 입력(Spec), 모델, 손실, 표본 가중치, 증강을 정한다.
- 교차 검증은 FOLDS의 주행 묶음을 차례로 빼고 학습한다. epoch 수를 고정하고 검증 fold로 고르지 않는다. 정규화,
  가중치, 선형 기준 모델은 학습 fold에서만 구한다.
- 지표(fold 평균): 전체 RMSE와 |la| ≥ 0.5, ≥ 1.0의 RMSE, 그리고 같은 fold의 torqued식 선형 대비 개선.
"""
import functools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnlc.dataset import Spec, build, mirrored
from nnlc.models import Mlp, SpNet, fit_linear, torqued_like

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LA_EDGES = [0, 0.25, 0.5, 1.0, 1.5, 99]
FOLDS = {
    "10-12..10-15": ["2026-10-12--22-37-07-989", "2026-10-14--19-09-44-760", "2026-10-15--03-14-59-306"],
    "10-10 night": ["2026-10-10--18-34-37-481"],
    "10-10 long": ["2026-10-10--19-32-19-062"],
    "10-04..10-06": ["2026-10-04--12-32-26-070", "2026-10-05--16-28-44-410", "2026-10-06--01-05-37-636"],
    "09-25..10-01": ["2026-09-25--22-23-58-022", "2026-09-27--21-39-39-397", "2026-10-01--02-52-53-517"],
}


@dataclass(frozen=True)
class Cfg:
    spec: Spec = Spec()
    model: str = "mlp"          # mlp | wide | sp
    hidden: tuple = (32, 32)
    act: str = "tanh"           # tanh | silu
    loss: str = "mse"           # mse | huber
    weights: str = "bin20"      # bin20 | sqrt | none
    epochs: int = 40
    lr: float = 3e-3
    wd: float = 0.0
    ensemble: int = 1
    zero: tuple = ()            # 운용 때 0으로 둘 방해 입력(평가 때 0을 넣는다)
    eval_thr: float | None = None  # 평가 데이터의 운전자 토크 문턱(None이면 학습과 같다)
    mirror: bool = False        # 좌우 대칭 증강(학습 fold에만)


def make_weights(la: np.ndarray, kind: str) -> np.ndarray:
    """|la| 구간마다 드문 구간일수록 큰 표본 가중치(bin20은 최대 20배). 평균 1로 맞춘다."""
    if kind == "none":
        return np.ones(len(la), np.float32)
    b = np.digitize(np.abs(la), LA_EDGES) - 1
    counts = np.bincount(b, minlength=len(LA_EDGES) - 1).astype(np.float64)
    w = counts.max() / np.maximum(counts[b], 1.0)
    w = np.sqrt(w) if kind == "sqrt" else np.minimum(w, 20.0)
    return (w / w.mean()).astype(np.float32)


def make_net(cfg: Cfg, n_in: int) -> nn.Module:
    if cfg.model == "sp":
        return SpNet()
    return Mlp(n_in, cfg.hidden, cfg.act, wide=cfg.model == "wide")


def fit(cfg: Cfg, x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> nn.Module:
    """정규화된 x로 학습한다(AdamW, 코사인 학습률, 배치 8192, 고정 epoch)."""
    torch.manual_seed(seed)
    net = make_net(cfg, x.shape[1]).to(DEVICE)
    xd, yd, wd = (torch.from_numpy(a).to(DEVICE) for a in (x, y, w))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    n, batch = len(x), 8192
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            err = net(xd[j])[:, 0] - yd[j]
            if cfg.loss == "huber":
                loss = (wd[j] * nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.05)).mean()
            else:
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
    return build(Path(csv_dir), None, spec)


@dataclass
class Trained:
    nets: list
    mean: np.ndarray
    std: np.ndarray
    features: list
    linear: np.ndarray  # torqued식 선형 계수(a, b, c, d); d가 직진 토크 바이어스


def train(cfg: Cfg, x: np.ndarray, y: np.ndarray, n_seeds: int | None = None) -> Trained:
    """x, y(원래 단위)로 학습한다. 가중치와 선형 기준은 증강 전 원본에서, 정규화는 증강한 학습 데이터에서 구한다."""
    features = cfg.spec.names()
    w = make_weights(x[:, 1], cfg.weights)
    lin = fit_linear(torqued_like(x), y, w)
    xs, ys, ws = x, y, w
    if cfg.mirror:
        xs, ys = mirrored(x, y, features, float(lin[3]))
        ws = np.concatenate([w, w])
    mean, std = xs.mean(0), xs.std(0) + 1e-6
    xn = ((xs - mean) / std).astype(np.float32)
    nets = [fit(cfg, xn, ys, ws, seed) for seed in range(n_seeds or cfg.ensemble)]
    return Trained(nets, mean, std, features, lin)


def evaluate(t: Trained, x: np.ndarray, zero: tuple = ()) -> np.ndarray:
    x = x.copy()
    x[:, [t.features.index(z) for z in zero]] = 0.0
    return predict(t.nets, (x - t.mean) / t.std)


def cross_validate(csv_dir: Path, cfg: Cfg) -> dict:
    x, y, names, _ = dataset(str(csv_dir), cfg.spec)
    if cfg.eval_thr is not None:
        xe, ye, names_e, _ = dataset(str(csv_dir), replace(cfg.spec, driver_thr=cfg.eval_thr))
    else:
        xe, ye, names_e = x, y, names
    res = {k: [] for k in ("base", "model", "base_05", "model_05", "base_10", "model_10")}
    for val in FOLDS.values():
        tm, vm = ~np.isin(names, val), np.isin(names_e, val)
        t = train(cfg, x[tm], y[tm])
        xv, yv = xe[vm], ye[vm]
        pb = torqued_like(xv) @ t.linear
        pm = evaluate(t, xv, cfg.zero)
        for lo, sfx in ((0.0, ""), (0.5, "_05"), (1.0, "_10")):
            res["base" + sfx].append(rmse(xv[:, 1], yv, pb, lo))
            res["model" + sfx].append(rmse(xv[:, 1], yv, pm, lo))
    out = {k: float(np.nanmean(v)) for k, v in res.items()}
    for sfx in ("", "_05", "_10"):
        out["gain" + sfx] = 1 - out["model" + sfx] / out["base" + sfx]
    return out
