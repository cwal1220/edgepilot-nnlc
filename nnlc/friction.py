"""edgepilot 토크 컨트롤러의 마찰 보상 몫을 녹화 값으로 다시 계산한다(src/controls/lateral_torque.cc
TorqueController::update와 같은 식).

edgepilot은 학습한 토크 모델에 곡선(|목표 횡가속| 0.2→0.6 m/s²)에서 이 몫을 더한다. 모델은 추종 오차를 모르기
때문이다. 그래서 friction_free 학습은 정답 토크에서 같은 몫을 빼고, 모델은 나머지만 배운다(운용에서 더해도 겹치지
않는다). 녹화마다 달린 컨트롤러는 달랐지만, 빼는 것은 지금 운용이 그 상황에서 더할 몫이다.

- 요청 = ControlState 목표 곡률(지연 보정된 값) × v². 목표점 = 지연(lagd 추정 0.55 s)만큼 전의 요청, 오차 = 목표점 -
  실제 곡률 × v². 저크 = 요청 버퍼에서 지연 - 19틱 앞의 중앙차분을 1차 저역 필터(1.2 Hz)로 거른 값.
- 마찰 = interp(deadzone(오차 + 0.3·저크), ±0.2 m/s², ±friction). deadzone은 1° 조향각의 차량 모델 횡가속(40 km/h까지
  전부, 60 km/h에서 0). 결과는 정규화 토크(제어 부호, tq_norm과 같은 단위)다.
"""
import math

import numpy as np

DT = 0.01
FRICTION_THRESHOLD = 0.2      # kFrictionThreshold
JERK_GAIN = 0.3               # kJerkGain
JERK_LOOKAHEAD_FRAMES = 19    # kJerkLookaheadFrames
JERK_FILTER_ALPHA = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * 1.2 * DT))
DEADZONE_FULL_MPS, DEADZONE_OFF_MPS = 40.0 / 3.6, 60.0 / 3.6
CURVE_START, CURVE_FULL = 0.2, 0.6  # kNnCurveStartLatAccel, kNnCurveFullLatAccel

# 운용 값(2026-10-16 보드): lagd 지연, torqued 마찰, steering.json deadzone과 K7 차량 모델
DELAY_S = 0.55
FRICTION = 0.11
DEADZONE_DEG = 1.0
STEER_RATIO, WHEELBASE, MASS, CENTER_TO_FRONT_RATIO, STIFFNESS_FACTOR = 16.3, 2.855, 1816.0, 0.4, 1.0


def slip_factor(mass=MASS, wheelbase=WHEELBASE, ratio=CENTER_TO_FRONT_RATIO,
                stiffness_factor=STIFFNESS_FACTOR) -> float:
    """상류 VehicleModel의 슬립 계수(scale_tire_stiffness, 혼다 시빅 기준값)."""
    civic_mass, civic_wb = 1326.0 + 136.0, 2.70
    civic_front = civic_wb * 0.4
    civic_rear = civic_wb - civic_front
    front = wheelbase * ratio
    rear = wheelbase - front
    cf = 192150.0 * stiffness_factor * mass / civic_mass * (rear / wheelbase) / (civic_rear / civic_wb)
    cr = 202500.0 * stiffness_factor * mass / civic_mass * (front / wheelbase) / (civic_front / civic_wb)
    return mass * (cf * front - cr * rear) / (wheelbase ** 2 * cf * cr)


def deadzone_lat_accel(v: np.ndarray, deadzone_deg: float = DEADZONE_DEG) -> np.ndarray:
    deg = deadzone_deg * np.interp(v, [DEADZONE_FULL_MPS, DEADZONE_OFF_MPS], [1.0, 0.0])
    denom = 1.0 - slip_factor() * v * v
    curvature = np.abs(np.radians(deg) / STEER_RATIO / WHEELBASE / denom)
    return curvature * v * v


def friction_share(des_curv: np.ndarray, act_curv: np.ndarray, v: np.ndarray, friction: float = FRICTION,
                   delay_s: float = DELAY_S, deadzone_deg: float = DEADZONE_DEG) -> np.ndarray:
    """100 Hz 연속 구간 하나의 마찰 몫(정규화 토크). 구간 앞의 지연만큼은 첫 요청으로 채운다."""
    req = np.nan_to_num(des_curv * v * v)
    act = np.nan_to_num(act_curv * v * v)
    n = len(req)
    delay_frames = int(np.clip(int(delay_s / DT) + 1, 1, 100))
    k = np.arange(n)
    expected = req[np.maximum(k - (delay_frames - 1), 0)]
    back = int(np.clip(delay_frames - 1 - JERK_LOOKAHEAD_FRAMES, 1, 98))
    raw_jerk = (req[np.maximum(k - (back - 1), 0)] - req[np.maximum(k - (back + 1), 0)]) / (2 * DT)
    jerk = np.empty(n)
    acc = 0.0
    for i in range(n):  # jerk_filtered_ += alpha * (raw - jerk_filtered_)
        acc += JERK_FILTER_ALPHA * (raw_jerk[i] - acc)
        jerk[i] = acc
    x = (expected - act) + JERK_GAIN * jerk
    dz = deadzone_lat_accel(v, deadzone_deg)
    x = np.where(x > dz, x - dz, np.where(x < -dz, x + dz, 0.0))
    return np.interp(x, [-FRICTION_THRESHOLD, FRICTION_THRESHOLD], [-friction, friction]).astype(np.float32)


def curve_weight(des_curv: np.ndarray, v: np.ndarray) -> np.ndarray:
    """운용이 모델에 마찰을 더하는 몫(|목표 횡가속| 0.2→0.6 m/s²에서 0→1)."""
    return np.interp(np.abs(des_curv * v * v), [CURVE_START, CURVE_FULL], [0.0, 1.0]).astype(np.float32)
