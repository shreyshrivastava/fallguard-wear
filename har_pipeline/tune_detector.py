#!/usr/bin/env python3
"""Retune the fall detector's thresholds for the wrist.

Tuning data (the model saw both during training, so these numbers are optimistic):
  - WEDA-FALL falls -> fall recall
  - the wearer's longest "Normal day" -> false alarms per 16 h day
Final check, run once on the chosen settings (never used for choosing):
  - UMAFall falls and activities (outside dataset)
  - the held-out "Normal day" from train_wrist.py

Rule: among settings that catch at least MIN_RECALL of the tuning falls, pick
the fewest tuning false alarms, then the highest recall.

Usage (from the repo root, after train_wrist.py):
  .venv/bin/python har_pipeline/tune_detector.py
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.neural_net import HARNet  # noqa: E402
from simulate_detector import auto_calls, machine, model_probs, preset_params, steps  # noqa: E402
from train_wrist import HZ, OUT, OWN, UMA, WEDA, read_own_stream, read_uma_stream, read_weda_stream  # noqa: E402

MIN_RECALL = 0.90
N_SAMPLES = 600
GRID = dict(
    spike=[20.0, 26.0, 32.0, 38.0],       # m/s²; data is capped at 39.2 (4 g)
    req_dip=[False, True],
    freefall=[5.5, 7.0],
    gyr=[1.8, 4.0, 8.0],                  # rad/s
    conf=[0.9, 0.97, 0.99],
    still_std=[0.5, 1.2, 2.0],
    min_tilt=[0.0, 30.0, 45.0],
    still_needed=[2, 3, 5],
    post_timeout=[4.5, 8.0],
    lying_from_gravity=[True, False],     # the app labels "watch face up" as lying
)


def prep(files, reader, model, keep_stream=False):
    out = []
    for f in files:
        x = reader(f)
        if x is not None:
            e, p = model_probs(model, x)
            out.append((f, steps(x, e, p), len(x) / HZ / 3600, x if keep_stream else None))
    return out


def recall(data, P):
    return sum(bool(machine(r, P)) for _, r, _, _ in data) / max(len(data), 1)


def per_day(day, P):
    _, rows, hours, _ = day
    return len(machine(rows, P)) / hours * 16


def main():
    t0 = time.time()
    model = HARNet(n_classes=2)
    model.load_state_dict(torch.load(f"{OUT}/wrist_model.pt"))
    model.eval()
    held_name = json.load(open(f"{OUT}/wrist_report.json"))["held_out_file"]
    normal = sorted(glob.glob(f"{OWN}/session_*_normal-day.csv"), key=os.path.getsize)
    tune_day_file = normal[-1]
    assert os.path.basename(tune_day_file) != held_name, "tuning day must not be the held-out day"

    weda_falls = prep(sorted(glob.glob(f"{WEDA}/F*/U*_R*_accel.csv")), read_weda_stream, model)
    tune_day = prep([tune_day_file], read_own_stream, model)[0]
    print(f"Prepared {len(weda_falls)} WEDA falls and {tune_day[2]:.1f} h tuning day in {time.time() - t0:.0f}s", flush=True)

    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    random.Random(0).shuffle(combos)
    base = preset_params("desk_safe")
    tried = []
    for k, combo in enumerate([tuple(base[key] for key in keys)] + combos[:N_SAMPLES]):
        P = dict(zip(keys, combo))
        r = recall(weda_falls, P)
        fa = per_day(tune_day, P) if r >= MIN_RECALL or k == 0 else None
        tried.append(dict(params=P, tune_recall=r, tune_alarms_per_day=fa))
        if k % 100 == 0:
            print(f"  {k}/{N_SAMPLES} settings tried", flush=True)

    ok = [t for t in tried if t["tune_alarms_per_day"] is not None and t["tune_recall"] >= MIN_RECALL]
    assert ok, f"no setting reached {MIN_RECALL:.0%} recall on the tuning falls"
    ok.sort(key=lambda t: (t["tune_alarms_per_day"], -t["tune_recall"]))
    best = ok[0]

    # Final check, once, on data never used for choosing.
    held = prep([os.path.join(OWN, held_name)], read_own_stream, model, keep_stream=True)[0]
    uma = prep(sorted(glob.glob(f"{UMA}/*.csv")), read_uma_stream, model)
    uma_f = [u for u in uma if "_Fall_" in u[0]]
    uma_a = [u for u in uma if "_Fall_" not in u[0]]

    def final(P):
        calls = auto_calls(held[3], machine(held[1], P))
        return dict(held_out_alarms_per_day=round(per_day(held, P), 1),
                    held_out_auto_calls=len(calls),
                    held_out_auto_calls_per_day=round(len(calls) / held[2] * 16, 1),
                    held_out_alarms=len(machine(held[1], P)), held_out_hours=round(held[2], 2),
                    umafall_fall_recall=round(recall(uma_f, P), 3),
                    umafall_activity_false_alarm=round(recall(uma_a, P), 3))

    report = dict(rule=f"fewest tuning-day alarms with >= {MIN_RECALL:.0%} tuning fall recall",
                  tuning_day=os.path.basename(tune_day_file), held_out_day=held_name,
                  current_app_default=dict(params=base, tune_recall=tried[0]["tune_recall"],
                                           tune_alarms_per_day=tried[0]["tune_alarms_per_day"], final=final(base)),
                  chosen=dict(**best, final=final(best["params"])),
                  runners_up=ok[1:6], settings_tried=len(tried), settings_meeting_recall=len(ok))
    with open(f"{OUT}/tuned_detector.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    def line(name, t):
        fin = t["final"]
        print(f"{name:22} tuning: {t['tune_recall']:.1%} falls, {t['tune_alarms_per_day']:.1f} alarms/day | "
              f"held-out day: {fin['held_out_alarms']} alarms in {fin['held_out_hours']} h "
              f"({fin['held_out_alarms_per_day']}/day), auto-calls {fin['held_out_auto_calls']} "
              f"({fin['held_out_auto_calls_per_day']}/day) | UMAFall: {fin['umafall_fall_recall']:.1%} falls, "
              f"{fin['umafall_activity_false_alarm']:.1%} activity false alarms")
    print(f"\n{len(ok)} of {len(tried)} settings reached {MIN_RECALL:.0%} recall on the tuning falls")
    line("App default (desk_safe)", report["current_app_default"])
    line("Chosen for wrist", report["chosen"])
    print("Chosen settings:", json.dumps(best["params"]))
    print(f"Saved {OUT}/tuned_detector.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
