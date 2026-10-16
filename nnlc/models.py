"""토크 모델: 신경망 두 가지와 비교용 선형 모델.

- Mlp: 은닉층 tanh MLP. edgepilot이 읽는 모델(export)은 이것뿐이다.
- SpNet: sunnypilot NNTorqueModel 꼴(18→7→13→3→1, σ, σ, 항등, 항등). 비교용.
- torqued식 선형: y = a·la + b·roll + c·clip(jerk/0.5, ±1) + d. 지금 제어기의 피드포워드 꼴이다.
"""
import numpy as np
import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, n_in: int, hidden=(32, 32)):
        super().__init__()
        layers, k = [], n_in
        for h in hidden:
            layers += [nn.Linear(k, h), nn.Tanh()]
            k = h
        layers.append(nn.Linear(k, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4 = nn.Linear(18, 7), nn.Linear(7, 13), nn.Linear(13, 3), nn.Linear(3, 1)

    def forward(self, x):
        return self.l4(self.l3(torch.sigmoid(self.l2(torch.sigmoid(self.l1(x))))))


def torqued_like(x: np.ndarray) -> np.ndarray:
    """선형 모델의 설계 행렬. x의 열 1 = la, 2 = jerk, 3 = roll(Spec.names 순서)."""
    return np.stack([x[:, 1], x[:, 3], np.clip(x[:, 2] / 0.5, -1, 1), np.ones(len(x))], 1)


def fit_linear(a: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(a * sw, y * sw[:, 0], rcond=None)[0]
