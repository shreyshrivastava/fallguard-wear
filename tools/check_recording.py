#!/usr/bin/env python3
"""Check FallGuard watch recordings for completeness.

Reports, per file, the measured sample rate and any gaps for each sensor.
Every number comes from the sensor timestamps in the file.

Usage: python3 tools/check_recording.py recordings/*.csv
"""
import sys
from statistics import median

GAP_FACTOR = 3  # a gap is an interval longer than 3x the median interval


def check(path):
    meta, ts = {}, {"A": [], "G": []}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k.strip()] = v.strip()
                continue
            if line.startswith("sensor,"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) == 5 and parts[0] in ts:
                ts[parts[0]].append(int(parts[1]))

    print(f"\n{path}")
    print(f"  label: {meta.get('label', '?')}   device: {meta.get('device', '?')}")
    ok = True
    for name, key in (("accelerometer", "A"), ("gyroscope", "G")):
        t = ts[key]
        if len(t) < 3:
            print(f"  {name}: {len(t)} samples (too few to check)")
            ok = ok and key == "G"
            continue
        out_of_order = sum(1 for a, b in zip(t, t[1:]) if b <= a)
        t = sorted(t)
        d = [(b - a) / 1e6 for a, b in zip(t, t[1:])]  # ms
        med = median(d)
        duration_s = (t[-1] - t[0]) / 1e9
        gaps = [x for x in d if x > GAP_FACTOR * med]
        missing_ms = sum(x - med for x in gaps)
        print(
            f"  {name}: {len(t)} samples over {duration_s:.1f} s, "
            f"median interval {med:.1f} ms ({1000 / med:.1f} Hz), "
            f"{len(gaps)} gap(s), longest {max(d):.0f} ms, "
            f"~{missing_ms / 1000:.1f} s missing"
            + (f", {out_of_order} out of order" if out_of_order else "")
        )
        if key == "A" and (gaps or out_of_order):
            ok = False
    print("  result:", "OK" if ok else "CHECK GAPS")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
