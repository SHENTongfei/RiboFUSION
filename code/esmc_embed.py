# -*- coding: utf-8 -*-
"""Embed all unique cohort peptides (9-mers) with ESM-C-600M (local, GPU-first).

Source of sequences: internal + external npz (same as mhcflurry_baseline).
Output: seq -> mean-pooled last-layer embedding (1152-d) npz.
Uses the do-sci-research P4 official entry (PGPR-PHYTOSHIELD env).
"""
import os
import numpy as np
import torch
from esm import pretrained

OUT = "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32/esmc_pep_emb.npz"
COHORTS = [
    "C:/Users/TS/WorkBuddy/sorf-mhc-present/data/processed/dataset.npz",
    "C:/Users/TS/WorkBuddy/sorf-mhc-present/data/external/dataset_external.npz",
    "C:/Users/TS/WorkBuddy/sorf-mhc-present/data/external/chong2020/dataset_external_chong.npz",
]

pep_set = set()
for p in COHORTS:
    d = np.load(p, allow_pickle=True)
    pep_set.update(d["pep_seq"].astype(str).tolist())
peps = sorted(pep_set)
print("unique peptides:", len(peps), flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
model = pretrained.ESMC_600M_202412(device=DEVICE).eval()

# smoke test
tok = model._tokenize(["AAAAAAAAA"]).to(DEVICE)
with torch.inference_mode():
    o = model(sequence_tokens=tok)
print("smoke emb dim:", o.embeddings.shape, flush=True)
assert o.embeddings.dim() == 3

embs = np.zeros((len(peps), o.embeddings.shape[-1]), dtype=np.float32)
B = 256
with torch.inference_mode():
    for i in range(0, len(peps), B):
        batch = peps[i:i + B]
        tokens = model._tokenize(batch).to(DEVICE)
        out = model(sequence_tokens=tokens)
        # mean-pool over real residues (exclude BOS/EOS): use attention-free mean of 1:-1 per seq (all same length 9 here, so uniform)
        e = out.embeddings[:, 1:-1, :].mean(dim=1).float().cpu().numpy()
        embs[i:i + B] = e
        if (i // B) % 20 == 0:
            print(f"{i + len(batch)}/{len(peps)}", flush=True)

np.savez_compressed(OUT, seqs=np.array(peps), emb=embs)
print("saved", OUT, embs.shape, flush=True)
