#!/usr/bin/env python3
# Aggregate full_tol_sweep.csv into head-to-head win-rates and geomean speedups
# for LAMG+ vs approxChol and vs AC, at tol 1e-8 and 1e-4.
import csv, math, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/full_tol_sweep.csv"
rows = list(csv.DictReader(open(path)))
def F(x):
    try: return float(x)
    except: return float('nan')

def gm(xs):
    xs = [x for x in xs if x > 0 and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

def analyze(rows, tol_suffix, ac_setup_k, ac_solve_k, ac_ok_k, label):
    # LAMG+ total = lp_setup + lp_s{tol}; opponent total = {oppo}_setup + {oppo}_s{tol}
    wins=0; tot=0; spd=[]
    for r in rows:
        if int(r['lp_ok'])!=1 or int(r[ac_ok_k])!=1: continue
        lp = F(r['lp_setup']) + F(r['lp_s'+tol_suffix])
        op = F(r[ac_setup_k]) + F(r[ac_solve_k])
        if math.isnan(lp) or math.isnan(op) or lp<=0 or op<=0: continue
        tot+=1
        if lp < op: wins+=1
        spd.append(op/lp)   # >1 => LAMG+ faster
    return wins, tot, gm(spd)

# keep real m>1e6 (already filtered in the run, but be safe)
rows = [r for r in rows if F(r['m'])>1_000_000]
print(f"graphs in file (m>1e6): {len(rows)}")
print(f"  lp_ok={sum(int(r['lp_ok']) for r in rows)}  ac_ok={sum(int(r['ac_ok']) for r in rows)}  ac2_ok={sum(int(r['ac2_ok']) for r in rows)}")

for tol,suf in (("1e-8","8"),("1e-4","4")):
    print(f"\n=== tol {tol} ===")
    for name,suk,sok,okk in (("approxChol","ac_setup","ac_s"+suf,"ac_ok"),
                             ("AC (lap2)","ac2_setup","ac2_s"+suf,"ac2_ok")):
        w,t,g = analyze(rows, suf, suk, sok, okk, name)
        print(f"  LAMG+ vs {name:12}: wins {w}/{t} = {100*w/t:.0f}%   geomean speedup {g:.2f}x")
