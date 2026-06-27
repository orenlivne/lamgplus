#!/usr/bin/env python3
"""Plot a LAMG+ hierarchical-coarsening figure on a social graph (analogue of coarsening_airfoil.png).
Reads <prefix>_{edges,labels,meta}.csv from fig_coarsening_social.jl; lays the graph out once
(force-directed) and shows fine + three coarsening levels, nodes colored by aggregate.
Usage: python3 fig_coarsening_social.py <prefix> <out.png>"""
import sys, csv
import numpy as np
import networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

pref, out = sys.argv[1], sys.argv[2]
edges = [(int(a), int(b)) for a, b in csv.reader(open(pref + "_edges.csv"))]
rows = list(csv.reader(open(pref + "_labels.csv")))[1:]
labels = np.array([[int(x) for x in r] for r in rows])           # n x 4 (L1..L4), 1-based
meta = dict(csv.reader(open(pref + "_meta.csv")))
n, m = int(meta["n"]), int(meta["edges"])
nc = [int(meta[f"ncoarse{k}"]) for k in (1, 2, 3, 4)]

G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
# Kamada–Kawai reveals modular structure better than spring on dense social graphs; seed spring with it.
pos0 = nx.kamada_kawai_layout(G)
pos = nx.spring_layout(G, pos=pos0, seed=7, iterations=60, k=2.2 / np.sqrt(n))
P = np.array([pos[i] for i in range(n)])
seg = np.array([[P[a], P[b]] for a, b in edges])

# panels: fine (level 0) + three coarsening levels
panels = [("Level 0 — fine graph", None, n),
          (f"Level 1 — {nc[0]} aggregates ({n/nc[0]:.1f}× coarser)", 0, nc[0]),
          (f"Level 2 — {nc[1]} aggregates ({n/nc[1]:.1f}× coarser)", 1, nc[1]),
          (f"Level 3 — {nc[2]} aggregates ({n/nc[2]:.1f}× coarser)", 2, nc[2])]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.7))
for ax, (title, col, count) in zip(axes, panels):
    ax.add_collection(LineCollection(seg, colors="0.6", linewidths=0.18, alpha=0.5, zorder=1))
    if col is None:
        c = "#3b6ea5"
    else:
        rng = np.random.default_rng(0)
        palette = rng.random((count, 3)) * 0.82 + 0.08          # distinct random color per aggregate
        c = palette[labels[:, col] - 1]
    ax.scatter(P[:, 0], P[:, 1], s=11, c=c, linewidths=0, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_visible(False)

glabel = sys.argv[3] if len(sys.argv) > 3 else "social network"
fig.suptitle(f"LAMG+ hierarchical coarsening of a {glabel}  —  "
             f"each color = one aggregate  ({n} nodes, {m} edges)", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
