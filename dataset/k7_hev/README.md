# K7 HEV dataset

[← README](../../README.md)

Drives of a Kia K7 (YG) HEV under edgepilot, 2026-09-25 to 2026-10-15, one
gzipped CSV per drive ([Data format](../../docs/data-format.md)). Drive names
are the local start times (KST). The files hold vehicle dynamics, steering and
torque only, with no position.

- **Recording:** edgepilot recordings (format versions 5–13), extracted with
  edgepilot's `extract_lateral_dataset` as of commit `aa8613a`, which reads
  the event logs in time order.
- **Car values:** `steer_max` 384, driver torque in K7 MDPS units. The
  vehicle model of `nnlc/friction.py`: steer ratio 16.3, wheelbase 2.855 m,
  mass 1816 kg.
- **Controller that drove:** edgepilot's torque controller with the linear
  feed-forward. Its tuning changed over the period: delay 0.42 → 0.55 s, and
  torqued learning from scratch several times.
- **Size:** 11.53 h recorded, 9.03 h with lateral control active, 7.84 h
  usable (the rows nnlc trains on), 550,675 samples at 20 Hz.

Usable minutes by speed and |lateral acceleration| (m/s²):

| v (m/s) | 0–0.25 | 0.25–0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–3 | 3+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-10 | 62.0 | 2.3 | 1.0 | 0.3 | 0.0 | 0.0 | 0.0 |
| 10-15 | 82.2 | 10.5 | 6.1 | 1.1 | 0.4 | 0.4 | 0.0 |
| 15-20 | 81.6 | 11.3 | 11.2 | 3.4 | 0.5 | 0.0 | 0.0 |
| 20-25 | 84.3 | 19.5 | 10.4 | 2.1 | 0.3 | 0.0 | 0.0 |
| 25-30 | 53.7 | 11.2 | 7.4 | 0.6 | 0.1 | 0.0 | 0.0 |
| 30-40 | 2.3 | 1.7 | 1.8 | 0.3 | 0.1 | 0.0 | 0.0 |

| Drive | Hours | Active | Usable | MB (gz) |
| --- | ---: | ---: | ---: | ---: |
| 2026-09-25--22-23-58-022 | 0.26 | 0.11 | 0.07 | 1.8 |
| 2026-09-25--23-05-37-557 | 0.22 | 0.15 | 0.11 | 2.0 |
| 2026-09-27--02-44-40-679 | 0.29 | 0.24 | 0.22 | 3.0 |
| 2026-09-27--21-39-39-397 | 0.65 | 0.63 | 0.56 | 7.6 |
| 2026-10-01--02-52-53-517 | 0.34 | 0.26 | 0.23 | 3.4 |
| 2026-10-02--00-21-22-650 | 0.11 | 0.09 | 0.06 | 1.1 |
| 2026-10-02--00-28-11-285 | 0.09 | 0.08 | 0.07 | 1.0 |
| 2026-10-02--23-59-43-880 | 0.10 | 0.01 | 0.01 | 0.6 |
| 2026-10-03--00-05-56-917 | 0.09 | 0.08 | 0.07 | 1.0 |
| 2026-10-03--00-11-23-005 | 0.15 | 0.12 | 0.09 | 1.6 |
| 2026-10-04--02-29-44-034 | 0.36 | 0.21 | 0.17 | 3.0 |
| 2026-10-04--02-54-42-100 | 0.42 | 0.29 | 0.23 | 3.9 |
| 2026-10-04--08-59-31-512 | 0.02 | 0.00 | 0.00 | 0.0 |
| 2026-10-04--12-32-26-070 | 0.64 | 0.59 | 0.46 | 6.8 |
| 2026-10-05--00-07-29-810 | 0.23 | 0.22 | 0.20 | 2.6 |
| 2026-10-05--00-32-44-758 | 0.59 | 0.23 | 0.19 | 3.2 |
| 2026-10-05--16-28-44-410 | 0.37 | 0.29 | 0.24 | 3.5 |
| 2026-10-06--01-05-37-636 | 0.22 | 0.15 | 0.12 | 2.0 |
| 2026-10-10--11-16-58-871 | 0.33 | 0.33 | 0.29 | 4.2 |
| 2026-10-10--17-34-10-953 | 0.35 | 0.33 | 0.28 | 4.2 |
| 2026-10-10--17-52-49-391 | 0.46 | 0.05 | 0.04 | 1.7 |
| 2026-10-10--18-34-37-481 | 1.35 | 1.32 | 1.20 | 16.5 |
| 2026-10-10--19-32-19-062 | 2.46 | 2.30 | 2.16 | 27.9 |
| 2026-10-12--22-37-07-989 | 0.36 | 0.22 | 0.17 | 3.1 |
| 2026-10-14--19-09-44-760 | 0.85 | 0.58 | 0.46 | 8.7 |
| 2026-10-15--03-14-59-306 | 0.22 | 0.16 | 0.13 | 2.3 |

Train on it directly:

```sh
.venv/bin/python scripts/train_final.py dataset/k7_hev models/my_k7.json
```
