#!/usr/bin/env python3
"""EVAL-03 2D overlay: plot the source/target/aligned clouds top-down (x-y).

Reads the CSV written by eval03_overlay_node (columns: label,x,y,z) and draws a
matplotlib scatter: source RED, target GREEN, aligned BLUE. Good alignment ->
blue (aligned) covers green (target). Always saves a PNG next to the CSV; also
shows a window when a display is available.

Usage: eval03_plot.py [overlay.csv]
"""
import csv
import sys

import matplotlib.pyplot as plt

COLORS = {"source": "red", "target": "green", "aligned": "blue"}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "eval03_overlay.csv"
    pts = {label: ([], []) for label in COLORS}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"]
                if label in pts:
                    pts[label][0].append(float(row["x"]))
                    pts[label][1].append(float(row["y"]))
    except FileNotFoundError:
        print(f"eval03_plot: CSV not found: {path}", file=sys.stderr)
        return 1

    plt.figure(figsize=(8, 8))
    # Draw target first so source/aligned sit on top.
    for label in ("target", "source", "aligned"):
        xs, ys = pts[label]
        plt.scatter(xs, ys, s=3, c=COLORS[label], label=f"{label} ({len(xs)})", alpha=0.5)
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EVAL-03 NDT overlay (top-down)\nblue (aligned) should cover green (target)")
    plt.tight_layout()

    out = path.rsplit(".", 1)[0] + ".png"
    plt.savefig(out, dpi=120)
    print(f"eval03_plot: saved {out}")
    try:
        plt.show()
    except Exception:  # noqa: BLE001 - headless / no display is fine, PNG is saved
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
