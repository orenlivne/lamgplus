#!/usr/bin/env python3
"""Reproduce LAMG paper Fig 4.1 from linear_scaling_results.csv.

Three panels: setup, solve, total time PER EDGE vs number of edges m.
A horizontal (slope-0) trend in log-x means O(m). We also fit the log-log
slope of per-edge time vs m; slope ~ 0 confirms linear scaling.
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INP = sys.argv[1] if len(sys.argv) > 1 else "linear_scaling_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "doc/paper-latex/linear_scaling.pdf"

m, setup_pe, solve_pe, total_pe, cyc = [], [], [], [], []
with open(INP) as f:
    for row in csv.DictReader(f):
        if int(row["ok"]) != 1:
            continue
        m.append(float(row["m"]))
        setup_pe.append(float(row["setup_per_edge_s"]))
        solve_pe.append(float(row["solve_per_edge_s"]))
        total_pe.append(float(row["total_per_edge_s"]))
        cyc.append(float(row["cycles"]))
m = np.array(m); setup_pe = np.array(setup_pe)
solve_pe = np.array(solve_pe); total_pe = np.array(total_pe)

def loglog_slope(x, y):
    lx, ly = np.log(x), np.log(y)
    return np.polyfit(lx, ly, 1)[0]

print(f"instances (ok)          : {len(m)}")
print(f"m range                 : {int(m.min())} .. {int(m.max())}")
print(f"median setup  us/edge   : {np.median(setup_pe)*1e6:.3f}")
print(f"median solve  us/edge   : {np.median(solve_pe)*1e6:.3f}")
print(f"median total  us/edge   : {np.median(total_pe)*1e6:.3f}")
print(f"mean   cycles           : {np.mean(cyc):.2f}")
print("-- log-log slope of PER-EDGE time vs m (0 => O(m)) --")
print(f"  setup  : {loglog_slope(m, setup_pe):+.4f}")
print(f"  solve  : {loglog_slope(m, solve_pe):+.4f}")
print(f"  total  : {loglog_slope(m, total_pe):+.4f}")
# Equivalent total-runtime exponent beta = slope(time vs m): time = m * per_edge
print(f"  total-time exponent (slope of t_total vs m) : "
      f"{loglog_slope(m, total_pe*m):+.4f}  (1.0 => exactly O(m))")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
panels = [("setup", setup_pe, "(a) setup time per edge"),
          ("solve", solve_pe, "(b) solve time per edge"),
          ("total", total_pe, "(c) total time per edge")]
for a, (_, y, title) in zip(ax, panels):
    a.scatter(m, y * 1e6, s=8, c="#00435A", alpha=0.45, edgecolors="none")
    med = np.median(y) * 1e6
    a.axhline(med, color="#31CBC8", ls="--", lw=1.4,
              label=f"median {med:.2f} µs/edge")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("# edges  m")
    a.set_ylabel("time per edge  [µs]")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8, loc="upper right")
    a.grid(True, which="both", ls=":", alpha=0.3)
fig.suptitle(f"LAMG.jl linear scaling — {len(m)} real-world graphs "
             f"(m = {int(m.min()):,}..{int(m.max()):,})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, bbox_inches="tight")
print(f"\nwrote {OUT}")
