# -*- coding: utf-8 -*-
"""Route B allele-aware reader head training.
Architecture: peptide CNN -> pool; allele groove CNN -> pool; concat -> MLP -> logit.
Trains on MHCFlurry-curated (peptide, HLA-A/B/C allele, binder label) pairs.
Heartbeat (SIR-3a) + checkpoint/resume (SIR-4) built in.
Device: follows CUDA_VISIBLE_DEVICES set by gpuq scheduler; CPU fallback ok.
"""
import json
import os
import sys
import time
import threading

import numpy as np
import torch
import torch.nn as nn

DATA = r"C:\WorkBuddyBigFiles\routeb\routeb_dataset.npz"
CKPT_DIR = r"C:\WorkBuddyBigFiles\routeb\ckpt"
OUT_METRICS = r"C:\WorkBuddyBigFiles\routeb\train_metrics.json"

MAX_PEP, MAX_ALE = 15, 40
VOCAB = 21  # 0 pad + 20 AA
EMB = 64
BATCH = 4096
EPOCHS = 30
LR = 3e-4


# ---------- SIR-3a heartbeat ----------
def _start_scheduler_heartbeat():
    task_id = os.environ.get("SCHEDULER_TASK_ID")
    log_dir = os.environ.get("SCHEDULER_LOG_DIR")
    if not task_id or not log_dir:
        return
    hb = os.path.join(log_dir, f"{task_id}.heartbeat")

    def _pulse():
        while True:
            try:
                tmp = hb + ".tmp"
                with open(tmp, "w") as f:
                    f.write(str(time.time()))
                os.replace(tmp, hb)
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(target=_pulse, daemon=True).start()


_start_scheduler_heartbeat()

# ---------- model ----------
class CnnEncoder(nn.Module):
    def __init__(self, maxlen, emb=EMB, channels=128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.out = channels

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)          # B,E,L
        h = self.conv(h)
        mask = (x != 0).unsqueeze(1).float()
        h = (h * mask).sum(2) / mask.sum(2).clamp(min=1)
        return h                                  # B,C


class ReaderHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pep = CnnEncoder(MAX_PEP)
        self.ale = CnnEncoder(MAX_ALE)
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, p, a):
        return self.head(torch.cat([self.pep(p), self.ale(a)], 1)).squeeze(1)


# ---------- utils ----------
def auroc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # tie handling via average ranks
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def save_ckpt(path, model, opt, epoch, best):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best": best}, path)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    d = np.load(DATA, allow_pickle=True)
    Xp, Xa, y = d["X_pep"], d["X_ale"], d["y"].astype(np.float32)
    tr, va = d["train_ids"], d["val_ids"]
    print(f"train={len(tr)} val={len(va)}", flush=True)

    model = ReaderHead().to(dev)
    pos_weight = torch.tensor([(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)],
                              dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    start_epoch, best = 0, -1.0
    os.makedirs(CKPT_DIR, exist_ok=True)
    latest = os.path.join(CKPT_DIR, "latest.ckpt")
    bestp = os.path.join(CKPT_DIR, "best.ckpt")
    if os.path.exists(latest) and "--resume" in sys.argv:
        try:
            ck = torch.load(latest, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_epoch, best = ck["epoch"] + 1, ck["best"]
            print(f"resumed from epoch {start_epoch}, best={best:.4f}", flush=True)
        except Exception as e:
            print(f"resume failed: {e}", flush=True)

    Tp = torch.from_numpy(Xp[tr]).long(); Ta = torch.from_numpy(Xa[tr]).long()
    Ty = torch.from_numpy(y[tr])
    Vp = torch.from_numpy(Xp[va]).long().to(dev)
    Vay = torch.from_numpy(Xa[va]).long().to(dev)
    vy = y[va]

    n = len(tr)
    metrics = []
    for ep in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pb = Tp[idx].to(dev, non_blocking=True)
            ab = Ta[idx].to(dev, non_blocking=True)
            yb = Ty[idx].to(dev, non_blocking=True)
            opt.zero_grad()
            loss = lossf(model(pb, ab), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(Vp, Vay)).cpu().numpy()
        auc = auroc(vy, s)
        ap = float(np.mean([np.mean(s[vy == 1]) > np.mean(s[vy == 0])] if False else [0])) if False else None
        # average precision (step-wise)
        order = np.argsort(-s)
        yy = vy[order]
        tp = np.cumsum(yy == 1)
        prec = tp / (np.arange(len(yy)) + 1)
        ap = float(prec[yy == 1].mean()) if (yy == 1).any() else float("nan")
        best_flag = auc > best
        if best_flag:
            best = auc
            save_ckpt(bestp, model, opt, ep, best)
        save_ckpt(latest, model, opt, ep, best)
        metrics.append({"epoch": ep, "loss": tot / n, "val_auroc": auc, "val_auprc": ap, "best": best})
        with open(OUT_METRICS, "w") as f:
            json.dump(metrics, f, indent=1)
        print(f"[EP {ep+1}/{EPOCHS}] loss={tot/n:.4f} val_auroc={auc:.4f} val_auprc={ap:.4f} best={best:.4f}", flush=True)

    # ---- leave-allele-out generalization ----
    model.load_state_dict(torch.load(bestp, map_location=dev, weights_only=False)["model"])
    model.eval()
    loo = d["loo_ids"]
    per_allele = {}
    with torch.no_grad():
        for aid in np.unique(d["A_idx"][loo]):
            ids = loo[d["A_idx"][loo] == aid]
            lp = torch.from_numpy(Xp[ids]).long().to(dev)
            la = torch.from_numpy(Xa[ids]).long().to(dev)
            s = torch.sigmoid(model(lp, la)).cpu().numpy()
            name = str(d["alleles"][aid])
            per_allele[name] = {"n": int(len(ids)), "auroc": auroc(y[ids], s)}
    with open(OUT_METRICS) as f:
        m = json.load(f)
    m.append({"loo": per_allele})
    with open(OUT_METRICS, "w") as f:
        json.dump(m, f, indent=1)
    print("LOO per-allele AUROC:", json.dumps(per_allele, indent=1), flush=True)
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
