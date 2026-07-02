#!/usr/bin/env python3
"""EVAL-02 (Sprint 1): plot per-pair relative transform error vs ground truth.

Reads the CSV written by eval02_relative_error_node (columns: pair_id, stamp_i,
stamp_j, t_err_m, r_err_deg, converged, verdict, fitness, gt_dt_i_ms,
gt_dt_j_ms) and draws two scatter panels: translation error and rotation error
vs pair_id. Reliable pairs are drawn one colour; pairs that did NOT converge or
that NDT-11 flagged (verdict != Reliable) are drawn in a different colour/marker
so bad measurements stand out. Saves a PNG next to the CSV.

Usage: eval02_relative_error.py [eval02_pairs.csv]
       (default analysis/eval02_pairs.csv relative to CWD)
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")  # headless: always save a PNG, no display needed
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "analysis/eval02_pairs.csv"

    good = {"id": [], "t": [], "r": []}
    bad = {"id": [], "t": [], "r": []}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                reliable = row["converged"] == "1" and row["verdict"] == "Reliable"
                bucket = good if reliable else bad
                bucket["id"].append(int(row["pair_id"]))
                bucket["t"].append(float(row["t_err_m"]))
                bucket["r"].append(float(row["r_err_deg"]))
    except FileNotFoundError:
        print(f"eval02: CSV not found: {path}", file=sys.stderr)
        return 1

    total = len(good["id"]) + len(bad["id"])
    if total == 0:
        print(f"eval02: no rows in {path}", file=sys.stderr)
        return 1

    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, key, ylabel in ((ax_t, "t", "translation error [m]"),
                            (ax_r, "r", "rotation error [deg]")):
        ax.scatter(good["id"], good[key], s=18, c="tab:blue",
                   label=f"reliable ({len(good['id'])})")
        ax.scatter(bad["id"], bad[key], s=36, c="tab:red", marker="x",
                   label=f"not-converged / unreliable ({len(bad['id'])})")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    ax_r.set_xlabel("pair_id")
    ax_t.set_title("EVAL-02 per-pair relative transform error vs ground truth")
    fig.tight_layout()

    out = path.rsplit(".", 1)[0].replace("_pairs", "_relative_error") + ".png"
    if out == path:  # CSV had no expected suffix; fall back
        out = path.rsplit(".", 1)[0] + "_relative_error.png"
    fig.savefig(out, dpi=120)
    print(f"eval02: saved {out} ({total} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
