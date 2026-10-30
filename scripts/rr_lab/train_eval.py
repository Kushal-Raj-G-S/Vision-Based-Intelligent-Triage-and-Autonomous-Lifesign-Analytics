import sys, pickle, numpy as np, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
SPd = Path(sys.argv[1])
d = pickle.load(open(SPd/"feats.pkl","rb")); X = d["X"]; M = d["meta"]
wi = np.array([m[0] for m in M]); ds = np.array([m[1] for m in M]); rec = np.array([m[1]+":"+m[2] for m in M])
fc = np.array([m[4] for m in M]); y = np.array([m[5] for m in M]); gold = np.array([m[6] for m in M])
rule_rr = np.array([m[7] for m in M]); rule_conf = np.array([m[8] for m in M])

def sess_err(acc, gl):
    if len(acc) < 3: return np.nan
    a = np.asarray(acc); q1,q3 = np.percentile(a,[25,75]); iqr=q3-q1
    keep = a[(a>=q1-1.5*iqr)&(a<=q3+1.5*iqr)]
    if len(keep)==0: keep=a
    return abs(np.median(keep)-np.median(gl))

def evaluate(sel, label=None):
    """sel: dict window_index -> (rr, conf0to1, gold, ds, rec)"""
    by = defaultdict(list)
    for w,(rr,c,g,dsn,rc) in sel.items(): by[(dsn,rc)].append((rr,c,g))
    return by

def score(by, thr):
    res = defaultdict(list); sup = defaultdict(lambda:[0,0])
    for (dsn,rc), L in by.items():
        acc=[(rr,g) for rr,c,g in L if c>=thr]
        sup[dsn][0]+=len(L)-len(acc); sup[dsn][1]+=len(L)
        res[dsn].append(sess_err([a for a,_ in acc],[g for _,g in acc]))
    out={}
    for dsn,v in res.items():
        a=np.array([e for e in v if e==e]); out[dsn]=(a.mean(), len(a), len(v), 100*sup[dsn][0]/sup[dsn][1], 100*(a<=2).mean())
    return out

def pick_best(prob, mask=None):
    sel = {}
    idx = np.arange(len(prob)) if mask is None else np.where(mask)[0]
    order = idx[np.lexsort((-prob[idx], wi[idx]))]
    seen=set()
    for i in order:
        if wi[i] in seen: continue
        seen.add(wi[i]); sel[wi[i]]=(fc[i], prob[i], gold[i], ds[i], rec[i].split(":")[1])
    return sel

def show(name, by, thrs):
    for t in thrs:
        o = score(by, t)
        print(f"{name:30s} thr={t:4.2f} " + " | ".join(f"{k.upper()} {v[0]:5.2f} (n{v[1]}/{v[2]}, sup {v[3]:4.1f}%, w2 {v[4]:4.1f}%)" for k,v in sorted(o.items())))

# baseline: shipped rule
base = {}
for i in range(len(wi)):
    base[wi[i]] = (rule_rr[i], rule_conf[i]/100.0, gold[i], ds[i], rec[i].split(":")[1])
show("SHIPPED RULE", evaluate(base), [0.30])

def mk(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, l2_regularization=1.0, random_state=0)

# leave-one-dataset-out
prob = np.zeros(len(y))
for held in ("bidmc","capno","vdb"):
    tr = ds != held; te = ds == held
    m = mk().fit(X[tr], y[tr]); prob[te] = m.predict_proba(X[te])[:,1]
by = evaluate(pick_best(prob))
print("\n== LEAVE-ONE-DATASET-OUT (train on two populations, score the third) ==")
show("RE-RANKER (LODO)", by, [0.2,0.3,0.4,0.5,0.6])
# grouped CV
prob2 = np.zeros(len(y)); gkf = GroupKFold(n_splits=5)
for tr,te in gkf.split(X,y,groups=rec):
    m = mk().fit(X[tr], y[tr]); prob2[te]=m.predict_proba(X[te])[:,1]
print("\n== GROUPED 5-FOLD (by recording, pooled populations) ==")
show("RE-RANKER (5-fold)", evaluate(pick_best(prob2)), [0.3,0.4,0.5,0.6])
pickle.dump(dict(prob_lodo=prob, prob_cv=prob2), open(SPd/"probs.pkl","wb"))
