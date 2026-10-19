"""README의 표에 있는 설정과, 고른 설정 CHOSEN(edgepilot에 내보내는 것).

표(2026-10-16)는 보낸 토크 그대로를 목표로 잰 것이라 그 설정들은 friction_free=False로 둔다(k7_hev_v2까지). k7_hev_v3은
기본값인 friction_free 목표를 쓴다. 고른 설정(k7_hev_v4, 2026-10-18)은 실제 궤적 대신 edgepilot의 운용 입력으로 배우고
(nnlc.runtime) 조향각 입력이 없으며, 폐루프 재생으로 정한 곡선 이득을 모델 파일에 둔다(README "Runtime inputs").

2026-10-16 비교 실험의 나머지 설정(망 크기, SiLU, Huber, 표본 가중치, 지연, 창 등)과 그 코드는 커밋 46f08f5에 있다.
"""
from dataclasses import replace

from nnlc.dataset import CHOSEN as CHOSEN_SPEC
from nnlc.dataset import NO_ANGLE
from nnlc.training import Cfg

BASE = Cfg(spec=replace(Cfg().spec, friction_free=False))       # sunnypilot NNLC 입력 18개, 32×32 tanh
A = replace(BASE, spec=replace(CHOSEN_SPEC, friction_free=False))  # + 조향각 창, 촘촘한 la 창, 종가속(입력 33개)
V3 = replace(A, spec=CHOSEN_SPEC, mirror=True, ensemble=3)

CONFIGS = {
    "sp (sunnypilot)": replace(BASE, model="sp", epochs=60),
    "base mlp32 tanh": BASE,
    "A": A,
    "A + mirror": replace(A, mirror=True),
    "A + mirror, 3 nets": replace(A, mirror=True, ensemble=3),  # k7_hev_v2
    # k7_hev_v3. 배포한 파일의 curve_gain 0.3은 v4와 같이 폐루프 재생으로 정했다(2026-10-18, 학습과 상관없는 값)
    "A + mirror, 3 nets, friction-free": replace(V3, curve_gain=0.3),
    # k7_hev_v4: 운용 입력, 조향각 입력 없음. curve_gain 0.3은 edgepilot 폐루프 재생(주행 여섯)에서 곡선
    # 진입·유지·탈출이 선형 피드포워드와 같아지는 값이다(0.4는 탈출에서 더 돈다)
    "runtime inputs, no angle": replace(V3, spec=NO_ANGLE, inputs="runtime", curve_gain=0.3),
}
CHOSEN = "runtime inputs, no angle"
