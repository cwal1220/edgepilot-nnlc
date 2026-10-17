# Data format

[← README](../README.md)

nnlc trains on one CSV per drive. edgepilot's `extract_lateral_dataset` writes
it from a recording: one row per `ControlState` record, which is one row per
10 ms control tick on current recordings (about 62 Hz on recordings before
2026-10). The files may be gzipped (`<drive>.csv.gz`); nnlc reads both.

The header must be exactly the column list below, in this order.
`nnlc/csvdata.py` refuses a file whose header differs, because it reads the
columns by position.

An excerpt to look at: [`examples/sample_k7_hev_30s.csv`](../examples/sample_k7_hev_30s.csv)
(30 s of a drive with curves).

## Sign convention

nnlc only needs the columns to agree with each other. For a model that
edgepilot can use, keep edgepilot's control sign:

- **Right is positive:** the curvatures, every lateral-acceleration column,
  `roll`, `bank` and `tq_norm`. A right curve has positive curvature,
  positive lateral acceleration and positive `tq_norm`.
- **Left is positive:** the raw CAN values `angle`, `yaw`, `driver_tq`,
  `apply_tq` and `eps_out_tq`, as the car reports them.

`scripts/coverage.py` prints the correlation of `tq_norm` with the lateral
acceleration and with the EPS torque for every drive. A negative value means a
column has the wrong sign.

## Columns

"Used" says what nnlc reads the column for. The others are kept for analysis.

| Column | Unit | Meaning | Used |
| --- | --- | --- | --- |
| `t` | s | time since the first record of the drive | resampling, gaps over 0.1 s split a drive |
| `v` | m/s | wheel speed | input; rows below 3 m/s are dropped |
| `active` | 0/1 | lateral control was steering | row filter |
| `desire` | enum | the model's desire: 0 none, 1/2 turn, 3/4 lane change, 5/6 keep | row filter (any nonzero drops the window) |
| `block` | text | why control was not active, `-` when it was | no |
| `angle` | ° | steering wheel angle, left + | input (steering-angle window); also fits the angle model |
| `driver_tq` | raw | driver torque on the column, left + (K7 MDPS units) | row filter: \|driver_tq\| < `Spec.driver_thr` (100) |
| `apply_tq` | raw | torque sent to the EPS, left + (`steer_max` 384) | no (`coverage.py`) |
| `tq_norm` | 1 | sent torque ÷ `steer_max`, right + | **target** |
| `des_tq_norm` | 1 | requested torque before the rate limits ÷ `steer_max`, right + | no |
| `lat` | m/s² | ESP lateral acceleration, right + | no |
| `lat_valid` | 0/1 | `lat` was valid | row filter |
| `yaw` | rad/s | ESP yaw rate, left + | no |
| `long` | m/s² | longitudinal acceleration, forward + | input |
| `bank` | m/s² | road bank as lateral acceleration (`lat + yaw·v` on straights, held in curves) | no |
| `roll` | rad | `bank / g` | input (roll and its window) |
| `des_curv` | 1/m | the controller's desired curvature (delay-compensated), right + | friction share |
| `act_curv` | 1/m | the controller's actual curvature, from the steering angle through the vehicle model, right + | **input** `la = act_curv·v²`; friction share |
| `des_la` | m/s² | `des_curv·v²` | no |
| `la_p03` … `la_p15` | m/s² | lateral acceleration in the model's plan at +0.3, 0.6, 1.0, 1.5 s | `la_p15` must be finite |
| `roll_p03` … `roll_p15` | rad | roll at the same times (edgepilot uses the current roll) | no |
| `model_age` | s | time since the latest model frame, −1 before the first | row filter: 0 ≤ age < 0.2 |
| `toi_active` | 0/1 | the EPS was taking the LKAS torque (K7: MDPS12 ToiActive) | row filter |
| `steer_fault` | 0/1 | temporary EPS fault | row filter |
| `blinker` | 0/1 | either turn signal on | row filter |
| `eps_out_tq` | raw | torque the EPS applied, left + (K7: MDPS12 OutTq − 2048); 0 when unknown | no (`coverage.py` sign check) |

## From rows to samples

`nnlc/csvdata.py` keeps a row when lateral control is active, the EPS takes
the torque, there is no EPS fault, lane change or blinker, the driver torque
is under the threshold, the speed is over 3 m/s, the lateral acceleration is
valid and the model frame is fresh.

`nnlc/series.py` resamples each run of rows to a 100 Hz grid and computes:

- `la = act_curv·v²` and its jerk (±0.05 s central difference);
- `fric` and `curve_w`, the friction share edgepilot adds to the model in
  curves (`nnlc/friction.py`).

`nnlc/dataset.py` then builds the samples: the target at time t
(`tq_norm − curve_w·fric` with the default `friction_free`), and the inputs
around the response time t + 0.3 s, taken at 20 Hz. See the README's
"Samples" section.
