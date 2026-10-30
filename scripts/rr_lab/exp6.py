import sys, numpy as np
sys.argv=[sys.argv[0], sys.argv[1]]
import lab
import core.rppg as R
from exp5 import q
rows = lab.load()

def make_pick(zone, r, span=(1.6, 2.8), smooth=1):
    def pick(f, p):
        ps = np.convolve(p, np.ones(smooth)/smooth, mode="same") if smooth > 1 else p
        k = int(np.argmax(ps))
        fk = f[k]*60
        if fk < zone:
            m = (f >= span[0]*f[k]) & (f <= span[1]*f[k])
            if m.any():
                idx = np.where(m)[0]; k2 = idx[int(np.argmax(ps[idx]))]
                if ps[k2] >= r*ps[k]:
                    k = int(k2)
        k = R.AdvancedRPPG._promote_subharmonic(f, p, k)
        k = R.AdvancedRPPG._demote_harmonic(f, p, k)
        conc = float(np.sum(p[max(0,k-2):k+3])/(float(np.sum(p))+1e-12)*100)
        return float(f[k]*60), conc
    return pick

def rule_with(pick, band=("fixed",0.10,0.50), ab=None):
    def rule(w, ctx=None):
        if w["y"] is None: return 0.0, 0.0
        if ab is None: lo, hi = band[1], band[2]
        else:
            hr = w.get("beat_hz",0.0)*60
            if hr <= 0: lo, hi = 0.10, 0.50
            else: lo = q(min(max(ab[0]*hr/60,0.10),0.25)); hi = q(min(max(ab[1]*hr/60,0.50),0.80))
        R._RESP_LOW, R._RESP_HIGH = lo, hi
        c, k = [], []
        for ch in ("riav","riiv"):
            s = lab.psd(w["y"][ch], lo=lo, hi=hi)
            if s is None: continue
            rr, cc = pick(*s); c.append(rr); k.append(cc)
        return lab.fuse_prod(c, k, w.get("beat_hz",0.0))
    return rule

if __name__ == "__main__":
    for zone in (10.5, 12.0, 14.0):
        for r in (0.2, 0.35, 0.55):
            lab.evaluate(rows, rule_with(make_pick(zone, r)), label=f"fixedband zone={zone} r={r}")
    R._RESP_LOW, R._RESP_HIGH = 0.10, 0.50
