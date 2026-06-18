import csv, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rows=list(csv.DictReader(open('/tmp/graph_features.csv')))
feats=['log10n','log10m','mean_deg_nm','log10deg','deg_median','deg_max','deg_p90','deg_p99',
       'deg_std','deg_cv','deg_skew','hub_ratio','frac_leaf','frac_hi']
X=[];y8=[];y4=[];deg=[]
for r in rows:
    if int(r['lp_ok'])!=1 or int(r['ac_ok'])!=1: continue
    lp8,ac8,lp4,ac4=float(r['lp8']),float(r['ac8']),float(r['lp4']),float(r['ac4'])
    if min(lp8,ac8,lp4,ac4)<=0: continue
    X.append([float(r[f]) for f in feats]); y8.append(int(lp8<ac8)); y4.append(int(lp4<ac4)); deg.append(float(r['mean_deg_nm']))
X=np.array(X);y8=np.array(y8);y4=np.array(y4);deg=np.array(deg)
print(f"N={len(y8)}  LAMG+ win rate: 1e-8 {y8.mean():.3f}  1e-4 {y4.mean():.3f}")

def best_thr(d,y):
    b=(0,0)
    for t in np.unique(d):
        a=((d>t)==y).mean()
        if a>b[0]: b=(a,t)
    return b
a,t=best_thr(deg,y8); print(f"single-threshold rule (in-sample) 1e-8: {a*100:.1f}% at d>{t:.1f}")
a4,t4=best_thr(deg,y4); print(f"single-threshold rule (in-sample) 1e-4: {a4*100:.1f}% at d>{t4:.1f}")

def ev(name,clf,X,y):
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=0)
    sc=cross_val_score(clf,X,y,cv=cv,scoring='accuracy')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)
    clf.fit(Xtr,ytr); te=clf.score(Xte,yte)
    print(f"  {name:16s} CV {sc.mean()*100:.1f}±{sc.std()*100:.1f}%   80/20-test {te*100:.1f}%")

def mkhgb(): return HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,l2_regularization=1.0,random_state=0)
print("\n== tol 1e-8 ==")
ev("HGB tree (all)",mkhgb(),X,y8)
ev("RandomForest",RandomForestClassifier(n_estimators=400,max_depth=4,random_state=0),X,y8)
ev("LogReg(linear)",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),X,y8)
ev("HGB deg-only",HistGradientBoostingClassifier(max_depth=2,max_iter=200,random_state=0),deg.reshape(-1,1),y8)
print("\n== tol 1e-4 ==")
ev("HGB tree (all)",mkhgb(),X,y4)

h=mkhgb(); h.fit(X,y8)
imp=permutation_importance(h,X,y8,n_repeats=50,random_state=0)
o=np.argsort(imp.importances_mean)[::-1]
print("\npermutation importance (1e-8, top 8):")
for i in o[:8]: print(f"  {feats[i]:12s} {imp.importances_mean[i]:+.3f}")
