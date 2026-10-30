#!/usr/bin/env python3
"""
Fine-tune the simulation-pretrained spectro-temporal tracker on ALL 125 sessions (several seeds) and export plain numpy
weights (models/hr_tracker.npz) plus a small parity file the main venv uses to check the numpy forward pass.

    tools/gpu_lab/venv/Scripts/python.exe scripts/export_hr_tracker.py [--seeds 3] [--ft-epochs 8]

Needs models-in-dataset/hr_tracker_pretrained.pt (written by scripts/hr_deep4.py).  Same recipe as the out-of-fold run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hr_deep4 as H4                                           # noqa: E402
from hr_deep import CACHE, load_truth                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=8)
    ap.add_argument("--pre", default=str(ROOT / "dataset" / "models" / "hr_tracker_pretrained.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "hr_tracker.npz"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = argparse.Namespace(feat=str(CACHE / "feat"), roi=str(CACHE / "roi"))
    sessions = H4.load_real(ns, load_truth())
    print(len(sessions), "sessions on", dev, flush=True)
    net0 = H4.SpecTracker().to(dev)
    net0.load_state_dict(torch.load(a.pre, map_location=dev))
    net0.eval()
    pack, nets = {}, []
    for r in range(a.seeds):
        net = H4.finetune(net0, sessions, a.ft_epochs, dev, seed=500 + r)
        nets.append(net)
        for k, v in net.state_dict().items():
            pack[f"net{r}_{k.replace('.', '_')}"] = v.detach().cpu().numpy().astype(np.float32)
        print(f"seed {r + 1}/{a.seeds} fine-tuned", flush=True)
    pack["n_nets"] = np.int64(a.seeds)
    Path(a.out).parent.mkdir(exist_ok=True)
    np.savez_compressed(a.out, **pack)
    print("written", a.out, flush=True)
    # parity reference: real-session sequences in, torch logits out (net 0)
    s = sessions[::9][:12]
    xs = []
    for q in s:
        ks = np.arange(150, len(q["X"]), 400)[:2]
        xs.append(q["X"][H4.seq_index(ks, len(q["X"]))])
    x = np.concatenate(xs).astype(np.float32)
    with torch.no_grad():
        logits = [n(torch.from_numpy(x).to(dev)).cpu().numpy() for n in nets]
    np.savez_compressed(CACHE / "tracker_parity.npz", x=x, logits=np.stack(logits))
    print("parity reference", x.shape, flush=True)


if __name__ == "__main__":
    main()
