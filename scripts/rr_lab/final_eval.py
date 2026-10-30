import sys, pickle, numpy as np, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
SPd = Path(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "build":
    sys.argv = [sys.argv[0], sys.argv[1]]
    import build_feats as B, lab
    rows = lab.load() + pickle.load(open(SPd/"cache_vdb.pkl","rb")) + pickle.load(open(SPd/"cache_vdb_age.pkl","rb"))
    B.main(rows, SPd/"feats_all.pkl"); sys.exit(0)
from sklearn.ensemble import HistGradientBoostingClassifier
d = pickle.load(open(SPd/"feats_all.pkl","rb")); X = d["X"]; M = d["meta"]
wi = np.array([m[0] for m in M]); ds = np.array([m[1] for m in M]); rec = np.array([m[2] for m in M])
fc = np.array([m[4] for m in M]); y = np.array([m[5] for m in M]); gold = np.array([m[6] for m in M])
rule_rr = np.array([m[7] for m in M]); rule_conf = np.array([m[8] for m in M])
groups = sorted(set(ds))

def sess_err(acc, gl):
    if len(acc) < 3: return np.nan
    a = np.asarray(acc); q1,q3 = np.percentile(a,[25,75]); iqr=q3-q1
    keep = a[(a>=q1-1.5*iqr)&(a<=q3+1.5*iqr)]
    if len(keep)==0: keep=a
    return abs(np.median(keep)-np.median(gl))
def per_group(sel, thr):
    by = defaultdict(list)
    for w,(rr,c,g,dsn,rc) in sel.items(): by[(dsn,rc)].append((rr,c,g))
    out = defaultdict(list); sup=defaultdict(lambda:[0,0])
    for (dsn,rc),L in by.items():
        acc=[(rr,g) for rr,c,g in L if c>=thr]; sup[dsn][0]+=len(L)-len(acc); sup[dsn][1]+=len(L)
        out[dsn].append(sess_err([a for a,_ in acc],[g for _,g in acc]))
    return {k:(np.nanmean(v), sum(1 for e in v if e==e), len(v), 100*sup[k][0]/sup[k][1]) for k,v in out.items()}
def pick_best(prob):
    sel={}; order=np.lexsort((-prob, wi)); seen=set()
    for i in order:
        if wi[i] in seen: continue
        seen.add(wi[i]); sel[wi[i]]=(fc[i],prob[i],gold[i],ds[i],rec[i])
    return sel
base={}
for i in range(len(wi)): base[wi[i]]=(rule_rr[i], rule_conf[i]/100, gold[i], ds[i], rec[i])
B0 = per_group(base, 0.30)
allc=list(range(X.shape[1])); S1=[c for c in allc if c not in (0,2,3,4,5,33)]
print("n windows", len(set(wi)), " groups:", {g:int(len(set(wi[ds==g]))) for g in groups})
res = {}
for name, cols in (("S1", S1), ("ALL", allc)):
    prob = np.zeros(len(y))
    for held in groups:
        tr = ds != held; te = ds == held
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=3, min_samples_leaf=100,
                                           l2_regularization=2.0, random_state=0).fit(X[tr][:,cols], y[tr])
        prob[te] = m.predict_proba(X[te][:,cols])[:,1]
    res[name] = per_group(pick_best(prob), 0.5)
print(f"\n{'group':14s} {'SHIPPED':>9s} {'ML S1 (LOGO)':>14s} {'ML all (LOGO)':>14s}")
for g in groups:
    print(f"{g:14s} {B0[g][0]:9.2f} {res['S1'][g][0]:14.2f} {res['ALL'][g][0]:14.2f}    (n={B0[g][2]})")
print(f"{'MEAN':14s} {np.mean([B0[g][0] for g in groups]):9.2f} {np.mean([res['S1'][g][0] for g in groups]):14.2f} {np.mean([res['ALL'][g][0] for g in groups]):14.2f}")
