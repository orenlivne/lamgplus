import csv, math, statistics as st, sys
rows=list(csv.DictReader(open(sys.argv[1])))
by={}
for r in rows: by.setdefault(r['instance'],{})[r['rhs']]=r
full=[k for k,v in by.items() if 'LX' in v and 'RAND' in v]
def gm(v): return math.exp(st.mean([math.log(x) for x in v])) if v else float('nan')
print("graphs:",len(full))
for rhs in ['LX','RAND']:
    lp=[float(by[k][rhs]['lp_setup_s'])+float(by[k][rhs]['lp_solve_s']) for k in full]
    ap=[float(by[k][rhs]['apx_setup_s'])+float(by[k][rhs]['apx_solve_s']) for k in full]
    ac=[float(by[k][rhs]['ac_setup_s'])+float(by[k][rhs]['ac_solve_s']) for k in full]
    cyc=[int(by[k][rhs]['lp_cycles']) for k in full]; its=[int(by[k][rhs]['apx_its']) for k in full]
    conv=sum(1 for k in full if float(by[k][rhs]['lp_relres'])<=1e-7)
    print(f"{rhs}: LAMG+ vs approxChol geomean speedup {gm([a/l for a,l in zip(ap,lp)]):.2f}x win {100*sum(l<a for a,l in zip(ap,lp))/len(full):.0f}% | vs AC {gm([a/l for a,l in zip(ac,lp)]):.2f}x win {100*sum(l<a for a,l in zip(ac,lp))/len(full):.0f}% | LAMG+ cycles median {st.median(cyc)} max {max(cyc)} converged {conv}/{len(full)} | apx its median {st.median(its)} max {max(its)}")
r=[int(by[k]['RAND']['lp_cycles'])/int(by[k]['LX']['lp_cycles']) for k in full]; a=[int(by[k]['RAND']['apx_its'])/int(by[k]['LX']['apx_its']) for k in full]
print(f"RAND/LX cycle ratio LAMG+ geomean {gm(r):.2f}, approxChol its ratio {gm(a):.2f}")
worst=sorted(full,key=lambda k:-int(by[k]['RAND']['lp_cycles']))[:8]
print("worst LAMG+ under RAND:",[(k.replace('.mtx',''),int(by[k]['LX']['lp_cycles']),int(by[k]['RAND']['lp_cycles']),by[k]['RAND']['lp_acf']) for k in worst])
