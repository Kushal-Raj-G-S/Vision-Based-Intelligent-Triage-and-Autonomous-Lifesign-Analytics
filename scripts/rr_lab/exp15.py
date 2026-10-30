import sys, pickle, numpy as np
from pathlib import Path
SPd = Path(sys.argv[1]); sys.argv=[sys.argv[0], sys.argv[1]]
import lab
import core.rppg as R
from exp5 import q
rows = lab.load(); vdb = pickle.load(open(SPd/"cache_vdb.pkl","rb"))
A = R.AdvancedRPPG

def make_rule(hi_min=0.50, prior=None, a=0.06, b=0.30, zone=12.0, ratio=0.30):
    def rule(w, ctx=None):
        if w["y"] is None: return 0.0, 0.0
        hr = w.get("beat_hz", 0.0)*60
        lo, hi = (0.10, 0.50) if hr <= 0 else (q(min(max(a*hr/60,0.10),0.25)), q(min(max(b*hr/60,hi_min),0.80)))
        c, k = [], []
        for ch in ("riav","riiv"):
            r = lab.psd(w["y"][ch], lo=lo, hi=hi)
            if r is None: continue
            f, p = r
            if prior is not None and hr > 0:
                cc, sg, fl = prior
                z = np.log(np.maximum(f*60, 1e-6)/(cc*hr))
                p = p*(fl + (1-fl)*np.exp(-0.5*(z/sg)**2))
            kk = int(np.argmax(p))
            if f[kk]*60 < zone:
                m = (f >= 1.6*f[kk]) & (f <= 2.8*f[kk])
                if m.any():
                    idx = np.where(m)[0]; k2 = int(idx[np.argmax(p[idx])])
                    if p[k2] >= ratio*p[kk]: kk = k2
            kk = A._promote_subharmonic(f, p, kk, hi); kk = A._demote_harmonic(f, p, kk, lo)
            c.append(float(f[kk]*60)); k.append(float(np.sum(p[max(0,kk-2):kk+3])/(p.sum()+1e-12)*100))
        return lab.fuse_prod(c, k, w.get("beat_hz",0.0))
    return rule

def both(label, rule):
    lab.evaluate(rows, rule, label=label+" [95]")
    lab.evaluate(vdb, rule, label=label+" [VDB]")

both("baseline (hi_min .50, no prior)", make_rule())
for hm in (0.45, 0.40, 0.35):
    both(f"hi_min={hm}", make_rule(hi_min=hm))
for pr in ((0.18,0.7,0.5), (0.18,0.55,0.4), (0.18,0.9,0.5), (0.20,0.6,0.5)):
    both(f"prior c,s,floor={pr}", make_rule(prior=pr))
