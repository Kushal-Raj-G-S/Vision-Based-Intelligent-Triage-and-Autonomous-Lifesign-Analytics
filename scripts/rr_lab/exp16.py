import sys, pickle, numpy as np
from pathlib import Path
SPd = Path(sys.argv[1]); sys.argv=[sys.argv[0], sys.argv[1]]
import lab
import core.rppg as R
from exp5 import q
rows = lab.load(); vdb = pickle.load(open(SPd/"cache_vdb.pkl","rb"))
A = R.AdvancedRPPG; TOL = 2.0

def pick(f, p, lo, hi, zone=12.0, ratio=0.30):
    kk = int(np.argmax(p))
    if f[kk]*60 < zone:
        m = (f >= 1.6*f[kk]) & (f <= 2.8*f[kk])
        if m.any():
            idx = np.where(m)[0]; k2 = int(idx[np.argmax(p[idx])])
            if p[k2] >= ratio*p[kk]: kk = k2
    kk = A._promote_subharmonic(f, p, kk, hi); kk = A._demote_harmonic(f, p, kk, lo)
    return float(f[kk]*60), float(np.sum(p[max(0,kk-2):kk+3])/(p.sum()+1e-12)*100)

def make_rule(mode, pair_cap=60.0, pair_slope=8.0, a=0.06, b=0.30):
    def rule(w, ctx=None):
        if w["y"] is None: return 0.0, 0.0
        hr = w.get("beat_hz",0.0)*60
        lo, hi = (0.10,0.50) if hr<=0 else (q(min(max(a*hr/60,0.10),0.25)), q(min(max(b*hr/60,0.50),0.80)))
        C = {}
        for ch in ("riav","riiv","rifv"):
            r = lab.psd(w["y"][ch], lo=lo, hi=hi)
            if r is not None: C[ch] = pick(*r, lo, hi)
        bh = w.get("beat_hz",0.0)
        if "riav" not in C or "riiv" not in C: return lab.fuse_prod([v[0] for k,v in C.items() if k!="rifv"], [v[1] for k,v in C.items() if k!="rifv"], bh)
        av, iv = C["riav"], C["riiv"]
        base = lab.fuse_prod([av[0], iv[0]], [av[1], iv[1]], bh)
        if mode == "base" or abs(av[0]-iv[0]) <= TOL or "rifv" not in C: return base
        fv = C["rifv"]
        # av and iv disagree: does RIFV side with one of them?
        gaps = sorted([(abs(fv[0]-av[0]), av[0]), (abs(fv[0]-iv[0]), iv[0])])
        g, other = gaps[0]
        if g <= TOL:
            rr = 0.5*(fv[0]+other)
            return lab.derate(rr, float(np.clip(pair_cap - pair_slope*g, 15, pair_cap)), bh)
        return base
    return rule

def both(label, rule):
    lab.evaluate(rows, rule, label=label+" [95]"); lab.evaluate(vdb, rule, label=label+" [VDB]")
both("base (shipped)", make_rule("base"))
both("RIFV tiebreak cap60", make_rule("vote", 60, 8))
both("RIFV tiebreak cap45", make_rule("vote", 45, 8))
both("RIFV tiebreak cap35", make_rule("vote", 35, 4))
