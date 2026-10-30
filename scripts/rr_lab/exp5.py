import sys, numpy as np
sys.argv=[sys.argv[0], sys.argv[1]]
import lab
import core.rppg as R
rows = lab.load()

def q(x): return round(x*100)/100

def hr_band_rule(a, b, lo_min=0.10, lo_max=0.25, hi_min=0.50, hi_max=0.80):
    def rule(w, ctx=None):
        if w["y"] is None: return 0.0, 0.0
        hr = w.get("beat_hz", 0.0)*60
        if hr <= 0: lo, hi = 0.10, 0.50
        else:
            lo = q(min(max(a*hr/60, lo_min), lo_max)); hi = q(min(max(b*hr/60, hi_min), hi_max))
        R._RESP_LOW, R._RESP_HIGH = lo, hi
        cands, concs = [], []
        for ch in ("riav","riiv"):
            r = lab.psd(w["y"][ch], lo=lo, hi=hi)
            if r is None: continue
            rr, c = lab.pick(*r); cands.append(rr); concs.append(c)
        return lab.fuse_prod(cands, concs, w.get("beat_hz",0.0))
    return rule

if __name__ == "__main__":
    for a in (0.07, 0.09, 0.11):
        for b in (0.30, 0.35, 0.45):
            lab.evaluate(rows, hr_band_rule(a,b), label=f"HRband a={a:.2f} b={b:.2f}")
    R._RESP_LOW, R._RESP_HIGH = 0.10, 0.50
