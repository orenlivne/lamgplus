#!/usr/bin/env python3
# Figure: accuracy of the degree-threshold rule "LAMG+ wins iff mean-degree d > t" as a function of
# the threshold t, over the m>1e6 competition graphs. The accuracy is maximized near t=29, justifying
# the simple a-priori predictor. Caption also reports the point-biserial correlation of log(d) with a
# LAMG+ win.  Usage: python3 scripts/plot_predictor.py [full_tol_sweep.csv] [out.pdf]
import csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def F(x):
    try: return float(x)
    except: return float("nan")

INP = sys.argv[1] if len(sys.argv) > 1 else "doc/paper_program/results/full_tol_sweep_new.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "doc/paper_program/predictor.pdf"
r = [x for x in csv.DictReader(open(INP)) if x["lp_ok"] == "1" and x["ac_ok"] == "1"]
deg = np.array([2 * F(x["m"]) / F(x["n"]) for x in r])
win = np.array([(F(x["lp_setup"]) + F(x["lp_s8"])) < (F(x["ac_setup"]) + F(x["ac_s8"])) for x in r])
thr = np.linspace(2, 120, 600)
acc = np.array([np.mean((deg > t) == win) for t in thr])
best = thr[int(np.argmax(acc))]; bacc = acc.max()
r_pb = np.corrcoef(np.log(deg), win.astype(float))[0, 1]

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(thr, 100 * acc, color="#d00000", lw=2.8)
ax.axvline(best, color="0.4", ls=":", lw=1.8)
ax.scatter([best], [100 * bacc], color="#d00000", s=60, zorder=5)
ax.annotate(f"  max {100*bacc:.0f}% at $\\bar d \\approx {best:.0f}$",
            (best, 100 * bacc), fontsize=14, va="center")
ax.set_xlabel(r"mean-degree threshold $\bar d$", fontsize=14)
ax.set_ylabel("classifier accuracy  [%]", fontsize=14)
ax.grid(True, ls=":", alpha=0.4); ax.set_ylim(45, 95)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print(f"best d={best:.1f}  acc={100*bacc:.1f}%  point-biserial r={r_pb:+.2f}")
