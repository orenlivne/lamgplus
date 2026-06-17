#!/usr/bin/env python3
# Aggregate class_comparison.csv (long format: instance,class,solver,per_nnz_us,ok) into a
# GKS-style table: per (class, solver) the MEAN and WORST-CASE time-per-nonzero over the
# converged instances, plus a DNF count (instances the solver did not converge / errored).
# Emits a markdown table for inspection and a LaTeX tabular for the paper.
#   python3 scripts/class_comparison_table.py doc/paper_program/results/class_comparison.csv
import csv, math, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "doc/paper_program/results/class_comparison.csv"
rows = list(csv.DictReader(open(path)))
SOLVERS = ["LAMG+","approxChol","AC","BoomerAMG","pyAMG","CMG"]
# class display order
ORDER = ["FE/structural","mesh/grid","social/citation","web","road",
         "chimera","wtd-chimera","aniso-grid"]

# (class,solver) -> list of per_nnz for ok runs; and counts
okvals = defaultdict(list); total = defaultdict(int); dnf = defaultdict(int)
classes = []
for r in rows:
    c, s = r["class"], r["solver"]
    if c not in classes: classes.append(c)
    total[(c,s)] += 1
    if int(r["ok"]) == 1 and r["per_nnz_us"] not in ("","NaN","nan"):
        try: okvals[(c,s)].append(float(r["per_nnz_us"]))
        except: dnf[(c,s)] += 1
    else:
        dnf[(c,s)] += 1

classes = [c for c in ORDER if c in classes] + [c for c in classes if c not in ORDER]
def cell(c,s):
    v = okvals[(c,s)]
    if not v: return "DNF"
    mean = sum(v)/len(v); worst = max(v)
    tag = f" [{dnf[(c,s)]}✗]" if dnf[(c,s)] else ""
    return f"{mean:.2f}/{worst:.2f}{tag}"

print("# Per-class mean/worst setup+solve time per nonzero (µs/nnz), to rel-residual 1e-8.")
print("# cell = mean/worst ; [k✗] = k of the class's instances did not converge (DNF).\n")
hdr = "| class (n) | " + " | ".join(SOLVERS) + " |"
print(hdr); print("|" + "---|"*(len(SOLVERS)+1))
for c in classes:
    ninst = max(total[(c,s)] for s in SOLVERS)
    print(f"| {c} ({ninst}) | " + " | ".join(cell(c,s) for s in SOLVERS) + " |")

# LaTeX version for the paper.
print("\n% --- LaTeX ---")
print("\\begin{tabular}{l" + "r"*len(SOLVERS) + "}")
print("\\toprule")
print("class & " + " & ".join("\\texttt{"+s.replace('+','{+}')+"}" for s in SOLVERS) + "\\\\")
print("\\midrule")
for c in classes:
    print(c.replace('/','/') + " & " + " & ".join(cell(c,s).replace('✗','$\\times$') for s in SOLVERS) + "\\\\")
print("\\bottomrule")
print("\\end{tabular}")
