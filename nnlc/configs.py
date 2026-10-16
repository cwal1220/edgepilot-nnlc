"""README의 표에 있는 설정과, 고른 설정 CHOSEN(edgepilot에 내보내는 것).

2026-10-16 비교 실험의 나머지 설정(망 크기, SiLU, Huber, 표본 가중치, 지연, 창 등)과 그 코드는 커밋 46f08f5에 있다.
"""
from dataclasses import replace

from nnlc.dataset import CHOSEN as CHOSEN_SPEC
from nnlc.training import Cfg

BASE = Cfg()                       # sunnypilot NNLC 입력 18개, 32×32 tanh
A = replace(BASE, spec=CHOSEN_SPEC)  # + 조향각 창, 촘촘한 la 창, 종가속(입력 33개)

CONFIGS = {
    "sp (sunnypilot)": replace(BASE, model="sp", epochs=60),
    "base mlp32 tanh": BASE,
    "A": A,
    "A + mirror": replace(A, mirror=True),
    "A + mirror, 3 nets": replace(A, mirror=True, ensemble=3),
}
CHOSEN = "A + mirror, 3 nets"
