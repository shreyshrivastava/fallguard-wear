#!/usr/bin/env python3
"""Replay recordings through a Python port of the app's full fall pipeline.

Ports InferenceEngine (400 ms cadence, the 1 Hz "stationary" shortcut that
labels posture from gravity) and FallDetector (impact -> rotation -> model ->
post-fall stillness) from activity_collector/core, driven by the wrist model
from train_wrist.py. Reports confirmed alarms, which is what the wearer would
actually see, rather than raw model hits.

One deliberate difference from the current app: the 2-class wrist model has
no "lying" output, so the detector's "probs[1] >= 0.40 means lying" shortcut
is not applied (in the app it would fire constantly). Phase 4 must make the
same change in FallDetector.kt.

Usage (from the repo root, after train_wrist.py):
  .venv/bin/python har_pipeline/simulate_detector.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.neural_net import HARNet  # noqa: E402
from train_wrist import (HZ, WIN, OUT, OWN, UMA, WEDA, LABELS,  # noqa: E402
                         read_own_stream, read_uma_stream, read_weda_stream)

TICK = 10                      # precompute model output every 0.2 s
NORMAL_STEP, REST_STEP = 20, 50   # InferenceEngine: 400 ms, or 1000 ms when stationary
ALERT_HOLD_S = 30              # after an alarm the user dismisses it; detector resets

# FallDetector.applySensitivity presets (spike, freefall, requireDip, conf, minTilt, stillWindows)
PRESETS = {
    "desk_safe": (26.0, 5.5, True, 0.45, 32.0, 3),   # the app's default
    "balanced": (22.0, 6.5, False, 0.35, 28.0, 2),
    "high": (18.0, 7.5, False, 0.30, 20.0, 2),
}
GYR_SPIKE, STILL_STD = 1.8, 1.2
PHASE_TIMEOUT, POST_FALL_TIMEOUT = 2.5, 4.5


def model_probs(model, x):
    """P(fall) for the 2 s window ending at every TICK-th sample."""
    ends = np.arange(WIN, len(x) + 1, TICK)
    if len(ends) == 0:
        return ends, np.empty(0)
    out = []
    with torch.no_grad():
        for i in range(0, len(ends), 2048):
            batch = np.stack([x[e - WIN:e] for e in ends[i:i + 2048]])
            out.append(torch.softmax(model(torch.from_numpy(batch)), 1)[:, 0].numpy())
    return ends, np.concatenate(out)


def is_stationary(w):
    sub = w[::2]
    mag = np.linalg.norm(sub[:, :3], axis=1)
    var = (mag ** 2).mean() - mag.mean() ** 2
    return var < 0.05 and np.abs(sub[:, 3:]).sum(1).max() < 0.25


def unit(v):
    m = np.linalg.norm(v)
    return v / m if m > 0.001 else np.array([0.0, 1.0, 0.0])


def steps(x, ends, probs):
    """Everything the detector sees at each inference step. Depends only on the
    data (cadence comes from the stationary shortcut), so compute it once."""
    idx_of_end = {e: i for i, e in enumerate(ends)}
    rows, stationary_run, end = [], 0, WIN
    while end <= len(x):
        w = x[end - WIN:end]
        stationary_run = stationary_run + 1 if is_stationary(w) else 0
        if stationary_run > 2:
            az, ay = w[-1, 2], w[-1, 1]
            gravity_label = "lying" if abs(az) > 7 else "standing" if abs(ay) > 6 else "sitting"
            p_fall, step = 0.0, REST_STEP
        else:
            gravity_label, p_fall, step = None, probs[idx_of_end[end]], NORMAL_STEP
        mag = np.linalg.norm(w[:, :3], axis=1)
        rows.append((end / HZ, p_fall, gravity_label, mag.max(), mag.min(), mag.std(),
                     np.linalg.norm(w[:, 3:], axis=1).max(), unit(w[:, :3].mean(0))))
        end += step
    return rows


def preset_params(preset, conf_override=None):
    spike, freefall, req_dip, conf, min_tilt, still_needed = PRESETS[preset]
    return dict(spike=spike, freefall=freefall, req_dip=req_dip,
                conf=conf if conf_override is None else conf_override,
                min_tilt=min_tilt, still_needed=still_needed, gyr=GYR_SPIKE, still_std=STILL_STD,
                post_timeout=POST_FALL_TIMEOUT, lying_from_gravity=True)


def machine(rows, P):
    """FallDetector.onInferenceResult over precomputed steps; returns alarm times (s)."""
    phase, entered, still, pre = "IDLE", 0.0, 0, np.array([0.0, 9.8, 0.0])
    alarms, mute_until = [], -1.0
    for now, p_fall, g_label, acc_peak, acc_min, acc_std, gyr_peak, orient in rows:
        if now < mute_until:
            continue
        label = g_label or (LABELS[0] if p_fall >= 0.5 else LABELS[1])
        fall_pred = label == "falling" or p_fall >= P["conf"]
        lying = P["lying_from_gravity"] and label == "lying"   # no probs[1] shortcut, see docstring
        active = label in ("walking", "running")
        ff = acc_min < P["freefall"]
        timed_out = lambda limit: entered > 0 and now - entered > limit
        if phase == "IDLE":
            if not fall_pred and acc_std < 2.0:
                pre = orient
            spike = P["spike"]
            started = (ff and acc_peak >= spike * 0.75) if P["req_dip"] else (acc_peak > spike or (ff and gyr_peak > 1.2))
            if started:
                phase, entered, still = "P1", now, 0
        elif phase == "P1":
            if timed_out(PHASE_TIMEOUT):
                phase, entered = "IDLE", 0.0
            elif gyr_peak > P["gyr"]:
                phase, entered, still = "P2", now, 0
            elif acc_peak > P["spike"]:
                entered = now
        elif phase == "P2":
            if timed_out(PHASE_TIMEOUT):
                phase, entered = "IDLE", 0.0
            elif fall_pred and acc_peak > P["spike"] * 0.75:
                phase, entered, still = "P3", now, 0
        elif phase == "P3":
            if timed_out(P["post_timeout"]) or (active and acc_std > 2.0):
                phase, entered = "IDLE", 0.0
                continue
            tilt = np.degrees(np.arccos(np.clip(np.dot(unit(pre), orient), -1, 1)))
            if lying or (acc_std < P["still_std"] and tilt >= P["min_tilt"]):
                still += 1
                if still >= P["still_needed"]:
                    alarms.append(now)
                    phase, entered, still = "IDLE", 0.0, 0
                    mute_until = now + ALERT_HOLD_S
            elif acc_std > 2.0:
                still = 0
    return alarms


# Step 2 of the two-step alert: after a detected fall the watch asks "Are you OK?";
# it only auto-calls if the wearer then stays near-still. Commercial watches wait
# about a minute. shortcut: values chosen from one wearer's data; recall impact is
# untested because WEDA recordings end ~5 s after each fall.
IMMOBILE_S = 60          # seconds watched after the alarm
IMMOBILE_BLOCK_STD = 0.5  # m/s², std of |a| within a 1 s block counted as still
IMMOBILE_FRACTION = 0.8   # share of 1 s blocks that must be still


def auto_calls(x, alarm_times):
    """Alarms followed by IMMOBILE_S of near-stillness, i.e. those that would auto-call."""
    mag = np.linalg.norm(x[:, :3], axis=1)
    out = []
    for t in alarm_times:
        a = int(t * HZ)
        blocks = [mag[a + i * HZ:a + (i + 1) * HZ] for i in range(IMMOBILE_S)]
        blocks = [b for b in blocks if len(b) == HZ]
        if len(blocks) == IMMOBILE_S and np.mean([b.std() < IMMOBILE_BLOCK_STD for b in blocks]) >= IMMOBILE_FRACTION:
            out.append(t)
    return out


def run(x, ends, probs, preset, conf_override=None):
    """Return the times (s) of confirmed alarms in one stream."""
    return machine(steps(x, ends, probs), preset_params(preset, conf_override))


def self_check():
    """A quiet stream must never alarm; the state machine must reach an alarm for
    a scripted fall (still -> drop -> impact with rotation -> still, tilted)."""
    quiet = np.tile(np.array([0, 0, 9.8, 0, 0, 0], np.float32), (500, 1))
    e, p = np.arange(WIN, 501, TICK), np.zeros(len(np.arange(WIN, 501, TICK)))
    assert run(quiet, e, p, "balanced") == []
    fall = np.concatenate([
        np.tile([0, 0, 9.8, 0, 0, 0], (150, 1)),
        np.tile([0, 0, 2.0, 3, 3, 0], (10, 1)),        # weightless, rotating
        np.tile([25, 25, 10, 3, 3, 0], (5, 1)),        # impact
        np.tile([9.8, 0, 0, 0, 0, 0], (300, 1)),       # lying still, tilted 90 deg
    ]).astype(np.float32)
    e = np.arange(WIN, len(fall) + 1, TICK)
    alarms = run(fall, e, np.ones(len(e)), "balanced")
    assert len(alarms) == 1, "scripted fall must alarm once"
    still_after = np.concatenate([fall, np.tile(fall[-1], (IMMOBILE_S * HZ, 1))])
    assert auto_calls(still_after, alarms) == alarms, "60 s of stillness after a fall must auto-call"
    moving = still_after.copy()
    moving[len(fall):, 0] += np.sin(np.arange(IMMOBILE_S * HZ) / 3) * 3
    assert auto_calls(moving, alarms) == [], "moving after the alarm must not auto-call"


def main():
    self_check()
    model = HARNet(n_classes=2)
    model.load_state_dict(torch.load(f"{OUT}/wrist_model.pt"))
    model.eval()
    report = json.load(open(f"{OUT}/wrist_report.json"))
    held = os.path.join(OWN, report["held_out_file"])

    x = read_own_stream(held)
    ends, probs = model_probs(model, x)
    hours = len(x) / HZ / 3600
    variants = [(pr, None) for pr in PRESETS] + [(pr, 0.9) for pr in PRESETS]

    def streams(files, reader):
        out = []
        for f in files:
            s = reader(f)
            if s is not None:
                e, p = model_probs(model, s)
                out.append((f, s, e, p))
        return out

    uma = streams(sorted(glob.glob(f"{UMA}/*.csv")), read_uma_stream)
    weda = streams(sorted(glob.glob(f"{WEDA}/*/U*_R*_accel.csv")), read_weda_stream)

    def rates(data, is_fall, preset, conf):
        hit = {True: [0, 0], False: [0, 0]}
        for f, s, e, p in data:
            k = is_fall(f)
            hit[k][1] += 1
            hit[k][0] += bool(run(s, e, p, preset, conf))
        return hit[True][0] / max(hit[True][1], 1), hit[False][0] / max(hit[False][1], 1)

    results = {}
    print(f"Held-out day: {os.path.basename(held)}, {hours:.2f} h\n")
    print(f"{'preset':10} {'model conf':>10} | {'your day':>16} | {'UMAFall caught / false alarm':>28} | {'WEDA caught / false alarm (seen in training)':>44}")
    for preset, conf in variants:
        a = run(x, ends, probs, preset, conf)
        u = rates(uma, lambda f: "_Fall_" in f, preset, conf)
        wd = rates(weda, lambda f: os.path.basename(os.path.dirname(f)).startswith("F"), preset, conf)
        c = conf if conf is not None else PRESETS[preset][3]
        key = f"{preset}@{c}"
        results[key] = dict(own_alarms=len(a), own_hours=round(hours, 2), own_per_day_16h=round(len(a) / hours * 16, 1),
                            own_alarm_times_s=[round(t, 1) for t in a],
                            umafall_fall_recall=u[0], umafall_adl_false_alarm=u[1],
                            weda_fall_recall_seen=wd[0], weda_adl_false_alarm_seen=wd[1])
        print(f"{preset:10} {c:>10} | {len(a):3d} alarms ({len(a) / hours * 16:4.1f}/day) | "
              f"{u[0]:12.1%} / {u[1]:12.1%} | {wd[0]:21.1%} / {wd[1]:20.1%}")
    with open(f"{OUT}/detector_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}/detector_report.json")


if __name__ == "__main__":
    main()
