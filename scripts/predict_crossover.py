#!/usr/bin/env python3
# NOVELTY #1: an a-priori solver-selection criterion for graph Laplacians.
# Claim: the LAMG+ vs approxChol winner is PREDICTABLE from cheap graph structure
# (mean degree = 2m/n), computable before running either solver, because approxChol's
# sampled-Cholesky fill grows with elimination-clique size ~ degree, while caliber-1
# aggregation collapses density at O(1) cost. We find the mean-degree threshold that best
# separates the winners and report its accuracy, at both tolerances and per graph family.
#   python3 scripts/predict_crossover.py doc/paper_program/results/full_tol_sweep_new.csv
import csv, math, sys

path = sys.argv[1] if len(sys.argv) > 1 else "doc/paper_program/results/full_tol_sweep_new.csv"
rows = list(csv.DictReader(open(path)))
def F(x):
    try: return float(x)
    except: return float('nan')

def family(name):
    n = name.lower()
    if any(k in n for k in ("bmwcra","crankseg","troll","pwtk","bone","ldoor","hood","fault","af_shell","nasa","engine","gearbox","shipsec","thread","x104","s3dkt","ct20","pkustk","bcsstk","nd6k","nd12k","nd24k")): return "FE/structural"
    if any(k in n for k in ("web-","cnr","eu-2005","in-2004","uk-","webbase","wiki")): return "web"
    if any(k in n for k in ("soc","epinion","slashdot","email","amazon","com-","cit-","coauth","flickr","hollywood","dblp","citeseer","p2p","gowalla","brightkite")): return "social/citation"
    if any(k in n for k in ("road","osm")): return "road"
    if any(k in n for k in ("delaunay","grid","apache","thermal","ecology","parabolic","af23","g3_circuit","tmt","atmos","rgg","333sp","adaptive","channel","hugetric","hugebubble","venturi","wave","debr","net","cage","mesh")): return "mesh/grid"
    return "other"

# Build records: a-priori mean degree, and winners at each tol.
recs = []
for r in rows:
    n, m = F(r['n']), F(r['m'])
    if n <= 0 or m <= 0: continue
    if int(r['lp_ok']) != 1 or int(r['ac_ok']) != 1: continue
    deg = 2.0*m/n                                   # a-priori: mean degree
    lp8 = F(r['lp_setup']) + F(r['lp_s8']); ac8 = F(r['ac_setup']) + F(r['ac_s8'])
    lp4 = F(r['lp_setup']) + F(r['lp_s4']); ac4 = F(r['ac_setup']) + F(r['ac_s4'])
    if any(math.isnan(x) or x<=0 for x in (lp8,ac8,lp4,ac4)): continue
    recs.append(dict(name=r['instance'], deg=deg, fam=family(r['instance']),
                     win8=lp8 < ac8, win4=lp4 < ac4, r8=ac8/lp8, r4=ac4/lp4))

print(f"records: {len(recs)}")

def best_threshold(key):
    # predict LAMG+ wins iff deg > t*; scan candidate thresholds (the observed degrees).
    degs = sorted(set(round(x['deg'],3) for x in recs))
    best = (0.0, -1, None)   # (acc, t, ...)
    for t in degs:
        correct = sum(1 for x in recs if (x['deg'] > t) == x[key])
        acc = correct/len(recs)
        if acc > best[0]: best = (acc, t, None)
    return best[0], best[1]

for tol,key in (("1e-8","win8"),("1e-4","win4")):
    acc, t = best_threshold(key)
    above = [x for x in recs if x['deg'] > t]; below = [x for x in recs if x['deg'] <= t]
    win_above = sum(1 for x in above if x[key]); win_below = sum(1 for x in below if x[key])
    print(f"\n=== tol {tol}: rule 'LAMG+ wins iff mean-degree > {t:.1f}' ===")
    print(f"  accuracy {acc*100:.1f}%   ({len(above)} above thr: {win_above} LAMG+ wins; {len(below)} below: {win_below} LAMG+ wins)")

# Win-rate by family (shows the structural split directly).
print("\n=== winner by family (1e-8) ===")
fams = {}
for x in recs: fams.setdefault(x['fam'], []).append(x)
for f in sorted(fams):
    g = fams[f]; w = sum(1 for x in g if x['win8'])
    degs = sorted(x['deg'] for x in g)
    med = degs[len(degs)//2]
    print(f"  {f:16s} n={len(g):3d}  LAMG+ wins {w:3d}/{len(g):3d} ({100*w/len(g):3.0f}%)  median deg {med:6.1f}")

# Logistic-style separation strength: correlation of log-degree with the win indicator.
import statistics
ld = [math.log(x['deg']) for x in recs]; y = [1.0 if x['win8'] else 0.0 for x in recs]
if len(set(y)) > 1:
    mx,my = statistics.mean(ld), statistics.mean(y)
    cov = sum((a-mx)*(b-my) for a,b in zip(ld,y))/len(ld)
    r = cov/(statistics.pstdev(ld)*statistics.pstdev(y)+1e-30)
    print(f"\nlog(mean-degree) vs LAMG+-win point-biserial correlation: {r:+.2f}  (positive ⇒ higher degree favors LAMG+)")
