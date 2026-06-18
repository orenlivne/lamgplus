import csv, os, time
import numpy as np, scipy.io, scipy.sparse as sp
CSV="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/results/full_tol_sweep_new.csv"
DATA="/Users/oren/code/mg/maxflow/LAMG.jl/data"
rows=list(csv.DictReader(open(CSV)))
out=open("/tmp/graph_features.csv","w",newline=""); W=None; t0=time.time()
def dfeat(d):
    m=d.mean(); s=d.std()
    return dict(deg_mean=float(m), deg_median=float(np.median(d)), deg_max=float(d.max()),
        deg_p90=float(np.percentile(d,90)), deg_p99=float(np.percentile(d,99)),
        deg_std=float(s), deg_cv=float(s/m) if m>0 else 0.0,
        deg_skew=float(((d-m)**3).mean()/(s**3+1e-12)),
        hub_ratio=float(d.max()/m) if m>0 else 0.0,
        frac_leaf=float((d<=1).mean()), frac_hi=float((d>5*m).mean()))
for i,r in enumerate(rows):
    inst=r["instance"]; path=os.path.join(DATA,inst)
    try:
        A=sp.csr_matrix(scipy.io.mmread(path))
        deg=np.diff(A.indptr).astype(float) - (A.diagonal()!=0).astype(float)
        deg=np.maximum(deg,0.0)
        f=dfeat(deg); n=int(r["n"]); m=int(r["m"])
        f.update(dict(instance=inst,n=n,m=m,mean_deg_nm=2.0*m/n,
            log10n=float(np.log10(n)),log10m=float(np.log10(m)),log10deg=float(np.log10(2.0*m/n)),
            lp_ok=int(r["lp_ok"]),ac_ok=int(r["ac_ok"]),
            lp8=float(r["lp_setup"])+float(r["lp_s8"]),ac8=float(r["ac_setup"])+float(r["ac_s8"]),
            lp4=float(r["lp_setup"])+float(r["lp_s4"]),ac4=float(r["ac_setup"])+float(r["ac_s4"])))
        if W is None: W=csv.DictWriter(out,fieldnames=list(f.keys())); W.writeheader()
        W.writerow(f); out.flush()
        print(f"[{i+1}/{len(rows)}] {inst} n={n} {time.time()-t0:.0f}s",flush=True)
    except Exception as e:
        print(f"[{i+1}/{len(rows)}] FAIL {inst}: {e}",flush=True)
out.close(); print("DONE %.0fs"%(time.time()-t0),flush=True)
