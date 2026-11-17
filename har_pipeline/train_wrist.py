#!/usr/bin/env python3
"""Train and evaluate the wrist fall model for the Wear OS app (Phase 3).

Data
- WEDA-FALL (Fitbit Sense, wrist): training and cross-person evaluation.
  Fall timing comes from the dataset's fall_timestamps.csv.
- Your watch recordings (tools/pull_recordings.sh): not-fall training data,
  including the longest "Normal day" (what normal looks like for you),
  except the shortest "Normal day", held out to measure false alarms per hour.
- UMAFall wrist: outside test only. Its sensor clips at 250 deg/s and 8 g,
  so it is too distorted to train on.

All streams are resampled to 50 Hz on a shared timeline, acceleration is
capped at 4 g (WEDA's Fitbit limit) and gyroscope at 2000 deg/s, then cut
into 2 s windows. The watch app must apply the same caps before inference.

Usage (from the repo root):
  .venv/bin/python har_pipeline/train_wrist.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from models.neural_net import HARNet  # noqa: E402

HZ = 50
WIN = 100                 # 2 s
STEP_PUBLIC = 25          # 0.5 s
STEP_OWN_TRAIN = 100      # 2 s, own data is plentiful
STEP_OWN_TEST = 50        # 1 s
ACC_CAP = 4 * 9.80665     # WEDA's Fitbit saturates at 4 g
GYR_CAP = np.deg2rad(2000)
MIN_FALL_OVERLAP_S = 0.5
THRESHOLDS = (0.5, 0.7, 0.9)
LABELS = ["falling", "not_falling"]   # index 0 = falling, as FallDetector expects

WEDA = "datasets/weda-fall/repo/dataset/50Hz"
UMA = "datasets/umafall/data"
OWN = "recordings"
OUT = "har_pipeline/outputs/wrist"

torch.manual_seed(0)
np.random.seed(0)


# ── Resampling and windows ─────────────────────────────────────────────────────

def resample(ta, acc, tg, gyr):
    """Interpolate both sensors onto one 50 Hz grid over their shared span."""
    def clean(t, v):
        t, idx = np.unique(t, return_index=True)   # drop duplicate timestamps
        return t, v[idx]
    ta, acc = clean(ta, acc)
    tg, gyr = clean(tg, gyr)
    start, end = max(ta[0], tg[0]), min(ta[-1], tg[-1])
    if end - start < WIN / HZ:
        return None, None
    grid = np.arange(start, end, 1 / HZ)
    x = np.column_stack([np.interp(grid, ta, acc[:, i]) for i in range(3)] +
                        [np.interp(grid, tg, gyr[:, i]) for i in range(3)])
    x[:, :3] = np.clip(x[:, :3], -ACC_CAP, ACC_CAP)
    x[:, 3:] = np.clip(x[:, 3:], -GYR_CAP, GYR_CAP)
    return grid - grid[0], x.astype(np.float32)


def windows(x, step):
    starts = np.arange(0, len(x) - WIN + 1, step)
    return np.stack([x[s:s + WIN] for s in starts]) if len(starts) else np.empty((0, WIN, 6), np.float32), starts / HZ


# ── Stream readers: one file in, a 50 Hz (n, 6) array out (or None) ────────────

def read_weda_stream(accel_file):
    gf = accel_file.replace("_accel.csv", "_gyro.csv")
    if not os.path.exists(gf):
        return None
    a = pd.read_csv(accel_file, encoding="utf-8-sig").to_numpy(float)
    g = pd.read_csv(gf, encoding="utf-8-sig").to_numpy(float)
    return resample(a[:, 0], a[:, 1:4], g[:, 0], g[:, 1:4])[1]


def read_uma_stream(path):
    rows = {0: [], 1: []}
    for line in open(path, errors="ignore"):
        if line.startswith("%") or not line.strip():
            continue
        p = line.strip().split(";")
        if len(p) >= 7 and p[6].strip() == "3" and p[5].strip() in ("0", "1"):   # sensor 3 = wrist
            rows[int(p[5])].append([float(p[0]) / 1000, float(p[2]), float(p[3]), float(p[4])])
    if len(rows[0]) < 20 or len(rows[1]) < 20:
        return None
    a, g = np.array(rows[0]), np.array(rows[1])
    return resample(a[:, 0], a[:, 1:] * 9.80665, g[:, 0], np.deg2rad(g[:, 1:]))[1]


def read_own_stream(path):
    df = pd.read_csv(path, comment="#")
    a = df[df.sensor == "A"].to_numpy()
    g = df[df.sensor == "G"].to_numpy()
    if len(a) < WIN or len(g) < WIN:
        return None
    return resample(a[:, 1].astype(float) / 1e9, a[:, 2:5].astype(float),
                    g[:, 1].astype(float) / 1e9, g[:, 2:5].astype(float))[1]


# ── Loaders: each returns a list of recordings ────────────────────────────────
# recording = dict(source, subject, kind, X (n,100,6), y (n,), t (window starts s))

def load_weda():
    falls = {}
    ts = pd.read_csv(os.path.join(os.path.dirname(WEDA), "fall_timestamps.csv"), encoding="utf-8-sig")
    for _, r in ts.iterrows():
        falls[r["filename"]] = (float(r["start_time"]), float(r["end_time"]))
    recs = []
    for af in sorted(glob.glob(f"{WEDA}/*/U*_R*_accel.csv")):
        key = os.path.relpath(af, WEDA)[: -len("_accel.csv")]
        x = read_weda_stream(af)
        if x is None:
            continue
        X, t = windows(x, STEP_PUBLIC)
        kind = "fall" if key.startswith("F") else "adl"
        y = np.ones(len(X), int)   # not_falling
        keep = np.ones(len(X), bool)
        if kind == "fall":
            if key not in falls:
                continue
            s, e = falls[key]
            overlap = np.clip(np.minimum(t + WIN / HZ, e) - np.maximum(t, s), 0, None)
            y[overlap >= MIN_FALL_OVERLAP_S] = 0
            keep = (overlap >= MIN_FALL_OVERLAP_S) | (overlap == 0)   # drop windows that only graze the fall
        recs.append(dict(source="weda", subject="weda_" + os.path.basename(key).split("_")[0],
                         kind=kind, X=X[keep], y=y[keep], t=t[keep]))
    return recs


def load_umafall():
    recs = []
    for f in sorted(glob.glob(f"{UMA}/*.csv")):
        x = read_uma_stream(f)
        if x is None:
            continue
        X, t = windows(x, STEP_PUBLIC)
        kind = "fall" if "_Fall_" in f else "adl"
        y = np.ones(len(X), int)
        if kind == "fall":
            # No fall timing in UMAFall: label the windows around the largest impact.
            peak = np.argmax(np.linalg.norm(x[:, :3], axis=1)) / HZ
            near = np.abs(t + WIN / HZ / 2 - peak) <= 0.75
            X, t, y = X[near], t[near], np.zeros(near.sum(), int)
        recs.append(dict(source="umafall", subject="uma_" + os.path.basename(f).split("_")[2],
                         kind=kind, X=X, y=y, t=t))
    return recs


def load_own():
    """Your watch sessions. All are not-fall. Returns (train, held_out)."""
    files = sorted(glob.glob(f"{OWN}/session_*.csv"))
    normal = [f for f in files if f.endswith("_normal-day.csv")]
    # The longest day trains the model on what normal looks like for this wearer;
    # the shortest is held out so false alarms are still measured on unseen data.
    held_out = min(normal, key=os.path.getsize) if len(normal) > 1 else None
    train, test = [], None
    for f in files:
        x = read_own_stream(f)
        if x is None:
            continue
        step = STEP_OWN_TEST if f == held_out else STEP_OWN_TRAIN
        X, t = windows(x, step)
        rec = dict(source="own", subject="own", kind=os.path.basename(f), X=X, y=np.ones(len(X), int), t=t)
        if f != held_out:
            rec["pool"] = windows(x, STEP_PUBLIC)[0]   # denser windows to search for hard negatives
        if f == held_out:
            test = rec
        else:
            train.append(rec)
    return train, test


# ── Training and prediction ────────────────────────────────────────────────────

def stack(recs):
    return np.concatenate([r["X"] for r in recs]), np.concatenate([r["y"] for r in recs])


def train(X, y, epochs=15):
    model = HARNet(n_classes=2)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    # Square-root balancing: full balancing (~10x for falls) made the model alarm too eagerly.
    weights = torch.tensor(np.sqrt(counts.sum() / (2 * np.maximum(counts, 1))))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep + 1:2d}/{epochs}  loss {total / len(Xt):.4f}", flush=True)
    model.eval()
    return model


def train_with_hard_negatives(recs, own_pool, rounds_copies=3, th=0.5):
    """Train, then add the own-data windows the model wrongly flags as falls
    (from a denser pool of the same training sessions) and train again."""
    X, y = stack(recs)
    model = train(X, y)
    hard = own_pool[p_fall(model, own_pool) >= th]
    print(f"    hard negatives found: {len(hard)} of {len(own_pool)} own windows", flush=True)
    if len(hard) == 0:
        return model
    X = np.concatenate([X] + [hard] * rounds_copies)
    y = np.concatenate([y, np.ones(len(hard) * rounds_copies, int)])
    return train(X, y)


@torch.no_grad()
def p_fall(model, X):
    if len(X) == 0:
        return np.empty(0)
    out = [torch.softmax(model(torch.from_numpy(X[i:i + 2048])), 1)[:, 0] for i in range(0, len(X), 2048)]
    return torch.cat(out).numpy()


def event_metrics(model, recs, th):
    """Per recording: a fall recording counts as caught if any fall window fires;
    an activity recording counts as a false alarm if any window fires."""
    caught = falls = alarms = adls = 0
    for r in recs:
        p = p_fall(model, r["X"])
        if r["kind"] == "fall":
            falls += 1
            caught += bool((p[r["y"] == 0] >= th).any())
        else:
            adls += 1
            alarms += bool((p >= th).any())
    return dict(fall_recall=caught / max(falls, 1), falls=falls,
                adl_false_alarm_rate=alarms / max(adls, 1), adl_recordings=adls)


def alarms_per_hour(model, rec, th, merge_gap_s=3.0):
    p = p_fall(model, rec["X"])
    hits = rec["t"][p >= th]
    episodes = int(len(hits) > 0) + int((np.diff(hits) > merge_gap_s).sum()) if len(hits) else 0
    hours = (rec["t"][-1] + WIN / HZ) / 3600
    return dict(alarms=episodes, hours=round(hours, 2), per_hour=round(episodes / hours, 3))


# ── Self-check ─────────────────────────────────────────────────────────────────

def self_check():
    t = np.arange(0, 10, 0.025)   # 40 Hz input
    v = np.column_stack([np.sin(t), np.cos(t), np.full_like(t, 100.0)])
    grid, x = resample(t, v, t, v)
    assert abs(np.median(np.diff(grid)) - 1 / HZ) < 1e-9, "resample must output 50 Hz"
    assert np.allclose(x[:, 2], ACC_CAP), "acceleration must be capped at 4 g"
    X, starts = windows(x, STEP_PUBLIC)
    assert X.shape[1:] == (WIN, 6) and starts[1] == STEP_PUBLIC / HZ


def main():
    self_check()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    weda = load_weda()
    own_train, own_test = load_own()
    uma = load_umafall()
    assert own_test is not None, "need two Normal day recordings: one to train on, one to hold out"
    n = lambda rs: sum(len(r["y"]) for r in rs)
    print(f"Loaded in {time.time() - t0:.0f}s: WEDA {len(weda)} recordings / {n(weda)} windows "
          f"({sum(int((r['y'] == 0).sum()) for r in weda)} fall windows), "
          f"own train {n(own_train)} windows, own held-out {len(own_test['y'])} windows, "
          f"UMAFall {len(uma)} recordings")

    own_pool = np.concatenate([r["pool"] for r in own_train])

    # A. Cross-person check on WEDA: 5 folds, each person is only ever in one fold.
    subjects = np.array([r["subject"] for r in weda])
    report = {"folds": []}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(weda, groups=subjects)):
        print(f"Fold {k + 1}/5: testing on {len(set(subjects[te]))} unseen people")
        model = train_with_hard_negatives([weda[i] for i in tr] + own_train, own_pool)
        report["folds"].append({str(th): event_metrics(model, [weda[i] for i in te], th) for th in THRESHOLDS})

    # B/C. Final model on all WEDA + own training data.
    print("Final model")
    model = train_with_hard_negatives(weda + own_train, own_pool)
    report["umafall_outside_test"] = {str(th): event_metrics(model, uma, th) for th in THRESHOLDS}
    report["own_held_out_normal_day"] = {str(th): alarms_per_hour(model, own_test, th) for th in THRESHOLDS}
    report["held_out_file"] = own_test["kind"]
    report["labels"] = LABELS
    report["preprocessing"] = dict(hz=HZ, window=WIN, acc_cap_ms2=round(ACC_CAP, 3), gyr_cap_rads=round(float(GYR_CAP), 3))

    torch.save(model.state_dict(), f"{OUT}/wrist_model.pt")
    with open(f"{OUT}/wrist_labels.txt", "w") as f:
        f.write("\n".join(LABELS) + "\n")
    with open(f"{OUT}/wrist_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n== Results ==")
    for th in map(str, THRESHOLDS):
        folds = [fo[th] for fo in report["folds"]]
        rec = np.mean([fo["fall_recall"] for fo in folds])
        fa = np.mean([fo["adl_false_alarm_rate"] for fo in folds])
        u, o = report["umafall_outside_test"][th], report["own_held_out_normal_day"][th]
        print(f"threshold {th}: WEDA unseen people fall recall {rec:.1%}, activity false alarms {fa:.1%} | "
              f"UMAFall recall {u['fall_recall']:.1%}, false alarms {u['adl_false_alarm_rate']:.1%} | "
              f"your normal day {o['alarms']} alarms in {o['hours']} h ({o['per_hour']}/h)")
    print(f"Saved model and report to {OUT}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
