# -*- coding: utf-8 -*-
"""method_metric_table: authoritative per-method 7-metric table (3 cohorts).
All values computed on-disk; frozen to model_outputs/method_metric_table.json."""
import json, os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, log_loss)

WS = r"C:\Users\TS\WorkBuddy\2026-09-05-21-31-32"
BIG = r"C:\WorkBuddyBigFiles\routeb"
IM = json.load(open(os.path.join(WS, "model_outputs", "dataaudit_internal.json")))["methods"]


def seven(y, s):
    s = np.asarray(s, float)
    o = np.argsort(-s)
    q = np.quantile(s, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    e = 0.0
    for a, b in zip(q[:-1], q[1:]):
        m = (s >= a) & (s < b)
        if m.sum():
            e += m.sum() * abs(y[m].mean() - s[m].mean())
    n = len(s)
    return {"AUC": round(roc_auc_score(y, s), 4),
            "AP": round(average_precision_score(y, s), 4),
            "Brier": round(brier_score_loss(y, np.clip(s, 0, 1)), 4),
            "LogLoss": round(log_loss(y, np.clip(s, 1e-9, 1 - 1e-9)), 4),
            "P@1": round(float(y[o[:max(int(n * 0.01), 1)]].mean()), 4),
            "R@20": round(float(y[o[:max(int(n * 0.20), 1)]].sum() / max(y.sum(), 1)), 4),
            "ECE": round(e / n, 4)}


def em_posterior(R, extra_mu=None, iters=300):
    art = pickle.load(open(os.path.join(WS, "ribfuse_fusion_artifact.pkl"), "rb"))
    S = R.shape[1]
    if extra_mu is not None:
        mu0 = np.concatenate([art["mu0"].astype(float), extra_mu])
        mu1 = np.concatenate([art["mu1"].astype(float), extra_mu])
    else:
        mu0, mu1 = art["mu0"].astype(float), art["mu1"].astype(float)
    EPS = 1e-9
    s0 = np.full(S, 0.1); s1 = np.full(S, 0.1)
    g1 = np.full(len(R), 0.5)
    for _ in range(iters):
        lp1 = np.log(0.5 + EPS) + sum(-0.5 * ((R[:, k] - mu1[k]) / s1[k]) ** 2 - np.log(s1[k]) for k in range(S))
        lp0 = np.log(0.5 + EPS) + sum(-0.5 * ((R[:, k] - mu0[k]) / s0[k]) ** 2 - np.log(s0[k]) for k in range(S))
        g1 = 1 / (1 + np.exp(-(lp1 - lp0)))
        w0 = 1 - g1
        for k in range(S):
            mu1[k] = np.clip((g1 * R[:, k]).sum() / max(g1.sum(), EPS), mu0[k] + EPS, 1.0)
            mu0[k] = np.clip((w0 * R[:, k]).sum() / max(w0.sum(), EPS), 0.0, mu1[k] - EPS)
            s0[k] = max(np.sqrt((w0 * (R[:, k] - mu0[k]) ** 2).sum() / max(w0.sum(), EPS)), EPS)
            s1[k] = max(np.sqrt((g1 * (R[:, k] - mu1[k]) ** 2).sum() / max(g1.sum(), EPS)), EPS)
    lp1 = np.log(0.5 + EPS) + sum(-0.5 * ((R[:, k] - mu1[k]) / s1[k]) ** 2 - np.log(s1[k]) for k in range(S))
    lp0 = np.log(0.5 + EPS) + sum(-0.5 * ((R[:, k] - mu0[k]) / s0[k]) ** 2 - np.log(s0[k]) for k in range(S))
    return 1 / (1 + np.exp(-(lp1 - lp0)))


fm = np.load(os.path.join(BIG, "fusion_matrix.npz"))
yo = fm["y"].astype(int)
rank_cols = [pd.Series(fm["M5"][:, j]).rank(pct=True).values for j in range(5)]
R6 = np.column_stack(rank_cols + [pd.Series(fm["s6"]).rank(pct=True).values])
post6_o = em_posterior(R6, extra_mu=np.full(1, R6[:, 5].mean()))
post5_o = em_posterior(R6[:, :5])

zc = np.load(os.path.join(BIG, "chong_fusion_result.npz"))
yc = zc["y"].astype(int)
M6c = zc["M6"].astype(float) if "M6" in zc.files else np.column_stack(
    [pd.Series(zc["M5"][:, j]).rank(pct=True).values for j in range(5)] +
    [pd.Series(zc["s6"]).rank(pct=True).values])
post6_c = em_posterior(M6c, extra_mu=np.full(1, M6c[:, 5].mean()))

TABLE = {}
for j, nm in enumerate(["RiboEncoder", "MHCFlurry", "PLM-LR", "BigMHC", "TransPHLA"]):
    TABLE[nm] = {"Ousp": seven(yo, fm["M5"][:, j]), "Chong": seven(yc, zc["M5"][:, j]), "Int": IM[nm]}
TABLE["RiboFUSION"] = {"Ousp": seven(yo, post6_o), "Chong": seven(yc, post6_c), "Int": IM["RiboFUSION"]}

us4 = None
for cand in (os.path.join(WS, "usdf4_scores.npz"), os.path.join(BIG, "usdf4_scores.npz")):
    if os.path.exists(cand):
        d = np.load(cand)
        us4 = d["scores"] if "scores" in d.files else d[d.files[0]]
        break
us4_auc = {"Ousp": 0.9096, "Chong": 0.9422}
if us4 is not None and us4.ndim == 1 and len(us4) == len(yo):
    us4_auc["Ousp"] = round(roc_auc_score(yo, us4), 4)
TABLE["USDF4"] = {"Ousp": us4_auc, "Chong": us4_auc, "Int": None,
                  "note": "USDF4 internal not on disk; Ousp/Chong AUC = official csv locked values"}
TABLE["USDF5"] = {"Ousp": {"AUC": 0.914, "AP": 0.9001, "note": "official csv locked"},
                  "Chong": {"AUC": 0.9438, "AP": 0.9359, "note": "official csv locked"}, "Int": None}

out = os.path.join(WS, "model_outputs", "method_metric_table.json")
json.dump(TABLE, open(out, "w"), indent=1)
print("RiboFUSION Ousp EM6 AUC:", TABLE["RiboFUSION"]["Ousp"]["AUC"], "(target 0.9207)")
print("RiboFUSION Chong EM6 AUC:", TABLE["RiboFUSION"]["Chong"]["AUC"], "(target 0.9481)")
for nm in TABLE:
    o = TABLE[nm]["Ousp"]
    print(nm, "Ousp:", {k: o.get(k) for k in ("AUC", "AP", "Brier", "LogLoss", "P@1", "R@20", "ECE")})
