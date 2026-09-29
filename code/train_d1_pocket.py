# -*- coding: utf-8 -*-
"""D1 — train the anchor-resolved PocketReaderHead on real routeb data (CPU).

Demonstrates the field-biased head on the actual paired dataset:
  peptide sequence (anchor positions P2/C-term made explicit) x allele groove
  37-aa sequence (data-driven B/F pocket localization) -> binding label.

Run on CPU so it never competes with the GPU esm2_3b extraction.
Outputs: val pair AUROC + supertype-stratified AUROC.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, r"C:/Users/TS/WorkBuddy/2026-09-05-21-31-32")
from d1_pocket_reader import PocketReaderHead  # noqa: E402

DSET = r"C:/WorkBuddyBigFiles/routeb/routeb_dataset.npz"
GROOVE_CSV = r"C:/WorkBuddyBigFiles/mhcflurry_models/models.combined/allele_sequences.csv"
META = r"C:/WorkBuddyBigFiles/routeb/allele_metadata.csv"
AA = "ACDEFGHIKLMNPQRSTVWY"
N_TRAIN = 200_000        # CPU-friendly subsample of 370,600
EPOCHS = 4
BATCH = 512
LR = 3e-3
dev = "cpu"
torch.manual_seed(0)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    ds = np.load(DSET, allow_pickle=True)
    tr, va = ds["train_ids"], ds["val_ids"]
    Xp = ds["X_pep"]
    A_idx = ds["A_idx"]
    alleles = ds["alleles"].astype(str)
    y = ds["y"].astype(np.float32)

    # groove sequence lookup allele -> 37aa
    grv = pd.read_csv(GROOVE_CSV, header=None, names=["allele", "seq"]).set_index("allele")
    def groove(a):
        return grv.loc[a, "seq"] if a in grv.index else "Y" * 37

    def build(ids, n_sub=None):
        if n_sub and len(ids) > n_sub:
            rng = np.random.default_rng(0)
            ids = rng.choice(ids, n_sub, replace=False)
        pep = ["".join(AA[i - 1] for i in row if i > 0) for row in Xp[ids]]
        grv_seq = [groove(alleles[A_idx[i]]) for i in ids]
        yl = y[ids]
        return pep, grv_seq, yl

    print("building train subset...", flush=True)
    t_pep, t_grv, t_y = build(tr, N_TRAIN)
    print(f"train: {len(t_pep)} pairs (pos rate {t_y.mean():.3f})", flush=True)
    v_pep, v_grv, v_y = build(va)
    print(f"val: {len(v_pep)} pairs", flush=True)

    model = PocketReaderHead(d_model=64, n_heads=2, hid=256).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(t_y == 0).sum() / t_y.sum()]))

    n = len(t_pep)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pb = [t_pep[k] for k in idx]
            gb = [t_grv[k] for k in idx]
            yb = torch.from_numpy(t_y[idx]).to(dev)
            logit, g, _ = model(pb, gb)
            loss = lossf(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        # val
        model.eval()
        vout = []
        with torch.no_grad():
            for i in range(0, len(v_pep), BATCH):
                pb = v_pep[i:i + BATCH]; gb = v_grv[i:i + BATCH]
                logit, _, _ = model(pb, gb)
                vout.append(torch.sigmoid(logit).cpu().numpy())
        vs = np.concatenate(vout)
        print(f"[EP {ep+1}/{EPOCHS}] loss={tot/n:.4f}  val pair AUROC={auc(v_y, vs):.4f}", flush=True)

    # supertype-stratified on val
    meta = pd.read_csv(META).set_index("allele")
    va_alleles = [alleles[A_idx[i]] for i in va]
    sup = [meta.loc[a, "supertype"] if a in meta.index else "unknown" for a in va_alleles]
    print("\n=== D1 val pair AUROC by supertype ===")
    for s in sorted(set(sup)):
        m = np.array([x == s for x in sup])
        if m.sum() >= 20:
            print(f"  {s:16s} n={int(m.sum()):6d}  auroc={auc(v_y[m], vs[m]):.4f}")
    print("\nD1 POCKET READER TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
