# -*- coding: utf-8 -*-
"""RiboFuse USDF-5: add TransPHLA (S5) to the fusion, rerun full metric panel.

Run AFTER transphla_pep_max.csv exists.
Outputs: ribfuse5_results.csv + usdf5_scores.npz + internal transfer-CV check.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

WS = "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32"
SorfData = "C:/Users/TS/WorkBuddy/sorf-mhc-present"
EPS = 1e-9
rng = np.random.default_rng(0)

# ---------------- sources (internal OOF) ----------------
d = np.load(f"{WS}/ribocast_internal_oof.npz", allow_pickle=True)
y_int = d["y"].ravel().astype(int)
cond_int = d["cond"].astype(int)
s1_int = d["oof"].astype(float)
pep_int = d["pep_seq"].astype(str)
X_int = d["X"].astype(np.float32)
groups_int = d["groups"].astype(str)

mf = pd.read_csv(f"{WS}/mhcflurry_pep_cache.csv").set_index("seq")["best_percentile"]
s2_int = -np.log10(mf.reindex(pep_int).values.astype(float) + 1e-6)

pl = np.load("C:/WorkBuddyBigFiles/esmc_pep_emb.npz", allow_pickle=True)
plm = {s: i for i, s in enumerate(pl["seqs"].astype(str))}
E_all = pl["emb"]
E_int = E_all[[plm[s] for s in pep_int]]
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(skf.split(X_int, y_int, groups_int))
s3_int = np.zeros(len(y_int))
for tr, te in folds:
    sc = StandardScaler().fit(E_int[tr])
    lr3 = LogisticRegression(max_iter=2000).fit(sc.transform(E_int[tr]), y_int[tr])
    s3_int[te] = lr3.decision_function(sc.transform(E_int[te]))
sc_full = StandardScaler().fit(E_int)
lr3_full = LogisticRegression(max_iter=2000).fit(sc_full.transform(E_int), y_int)

bm = pd.read_csv("C:/WorkBuddyBigFiles/bigmhc_input.csv.prd")
bm_best = bm.groupby("pep")["BigMHC_EL"].max()
s4_int = bm_best.reindex(pep_int).values.astype(float)

tp = pd.read_csv("C:/WorkBuddyBigFiles/transphla_pep_max.csv").set_index("pep")["s5"]
s5_int = tp.reindex(pep_int).values.astype(float)
assert not np.isnan(s5_int).any(), "TransPHLA gap on internal peps"
print("S5 internal coverage OK; s5 AUROC:", round(roc_auc_score(y_int, s5_int), 4), flush=True)

def ranks(*scores):
    return np.stack([rankdata(s) / len(s) for s in scores], 1)

R5_int = ranks(s1_int, s2_int, s3_int, s4_int, s5_int)
R4_int = R5_int[:, :4]

# ---------------- USDF EM ----------------
EPSg = 1e-3
def run_em(R, mu0, mu1, pi, iters=300):
    n, Sm = R.shape
    mu0 = mu0.copy(); mu1 = mu1.copy()
    s0 = np.full(Sm, 0.15); s1v = np.full(Sm, 0.15)
    for _ in range(iters):
        lp1 = np.log(pi + EPSg) + sum(-0.5 * ((R[:, s] - mu1[s]) / s1v[s]) ** 2 - np.log(s1v[s]) for s in range(Sm))
        lp0 = np.log(1 - pi + EPSg) + sum(-0.5 * ((R[:, s] - mu0[s]) / s0[s]) ** 2 - np.log(s0[s]) for s in range(Sm))
        m = np.maximum(lp1, lp0)
        g1 = np.exp(lp1 - m) / (np.exp(lp1 - m) + np.exp(lp0 - m))
        pi = g1.mean()
        for s in range(Sm):
            w1 = g1 + EPSg; w0 = (1 - g1) + EPSg
            mu1[s] = np.clip((w1 * R[:, s]).sum() / w1.sum(), mu0[s] + EPSg, 1.0)
            mu0[s] = np.clip((w0 * R[:, s]).sum() / w0.sum(), 0.0, mu1[s] - EPSg)
            s1v[s] = max(np.sqrt((w1 * (R[:, s] - mu1[s]) ** 2).sum() / w1.sum()), EPSg)
            s0[s] = max(np.sqrt((w0 * (R[:, s] - mu0[s]) ** 2).sum() / w0.sum()), EPSg)
    return g1, mu0, mu1, s0, s1v, pi

mu1_init5 = np.array([R5_int[y_int == 1, s].mean() for s in range(5)])
mu0_init5 = np.array([R5_int[y_int == 0, s].mean() for s in range(5)])
pi_init5 = y_int.mean()

# internal full fit + transfer CV (selection rule: NEVER external)
print("\n=== internal transfer CV (EM fit on held-out condition, unlabeled) ===", flush=True)
levels = [l for l in sorted(set(cond_int.tolist())) if (cond_int == l).sum() >= 500]
cv5, cv4, cvu5, cvu4 = [], [], [], []
for lv in levels:
    m = cond_int == lv
    f5 = run_em(R5_int[m], mu0_init5, mu1_init5, pi_init5)
    f4 = run_em(R4_int[m], mu0_init5[:4], mu1_init5[:4], pi_init5)
    cv5.append(roc_auc_score(y_int[m], f5[0]))
    cv4.append(roc_auc_score(y_int[m], f4[0]))
    cvu5.append(roc_auc_score(y_int[m], R5_int[m].mean(1)))
    cvu4.append(roc_auc_score(y_int[m], R4_int[m].mean(1)))
    print(f"  cond {lv} (n={m.sum()}): USDF5={cv5[-1]:.4f} USDF4={cv4[-1]:.4f} unif5={cvu5[-1]:.4f}", flush=True)
print("transfer-CV mean: USDF5=%.4f USDF4=%.4f unif5=%.4f" %
      (np.mean(cv5), np.mean(cv4), np.mean(cvu5)), flush=True)

post_int5, _, mu1f5, _, _, _ = run_em(R5_int, mu0_init5, mu1_init5, pi_init5)
print("internal full USDF5 AUROC:", round(roc_auc_score(y_int, post_int5), 4), flush=True)

CLIP = 0.01
ir_post = IsotonicRegression(out_of_bounds="clip")
ir_post.fit(np.clip(post_int5, CLIP, 1 - CLIP), y_int)
ir_unif = IsotonicRegression(out_of_bounds="clip")
ir_unif.fit(R5_int[:, :3].mean(1), y_int)
ir_s2 = IsotonicRegression(out_of_bounds="clip")
ir_s2.fit(R5_int[:, 1], y_int)
ir_s4 = IsotonicRegression(out_of_bounds="clip")
ir_s4.fit(R5_int[:, 3], y_int)

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def hits_at_K(y, s, frac):
    k = max(1, int(round(frac * len(y))))
    return y[np.argsort(-s, kind="stable")[:k]].mean()

def boot_ci(y, a, b, n=2000):
    ds = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        ds.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return lo, hi, np.mean(ds)

rows = []
npz_out = {}
for name, npz_path, csv in [
    ("Ouspenskaia", f"{SorfData}/data/external/dataset_external.npz", f"{SorfData}/results/metrics/preds_external_ouspenskaia.csv"),
    ("Chong", f"{SorfData}/data/external/chong2020/dataset_external_chong.npz", f"{SorfData}/results/metrics/preds_external_chong.csv"),
]:
    dd = np.load(npz_path, allow_pickle=True)
    seqs = dd["pep_seq"].astype(str)
    y = dd["Y"].ravel().astype(int)
    pr = pd.read_csv(csv)
    s1 = pr["ribocast_logit"].values.astype(float)
    s2 = -np.log10(mf.reindex(seqs).values.astype(float) + 1e-6)
    E = E_all[[plm[s] for s in seqs]]
    s3 = lr3_full.decision_function(sc_full.transform(E))
    s4 = bm_best.reindex(seqs).values.astype(float)
    s5 = tp.reindex(seqs).values.astype(float)
    assert not np.isnan(s5).any(), f"TransPHLA gap on {name}"
    R5 = ranks(s1, s2, s3, s4, s5)
    R4 = R5[:, :4]
    post, _, _, _, _, pi_hat = run_em(R5, mu0_init5, mu1_init5, pi_init5)
    post4, _, _, _, _, _ = run_em(R4, mu0_init5[:4], mu1_init5[:4], pi_init5)
    post_cal = np.clip(ir_post.transform(np.clip(post, CLIP, 1 - CLIP)), EPS, 1 - EPS)

    def P(yy, s): return roc_auc_score(yy, s)
    lo, hi, md = boot_ci(y, post, s2)
    lo4, hi4, md4 = boot_ci(y, post, post4)
    row = {
        "cohort": name, "n": len(y), "pi_hat": round(pi_hat, 3),
        "AUROC_USDF5": round(P(y, post), 4),
        "AUROC_USDF4": round(P(y, post4), 4),
        "AUROC_uniform5": round(P(y, R5.mean(1)), 4),
        "AUROC_RIBOCAST": round(P(y, s1), 4),
        "AUROC_MHCFlurry": round(P(y, s2), 4),
        "AUROC_PLM": round(P(y, s3), 4),
        "AUROC_BigMHC": round(P(y, s4), 4),
        "AUROC_TransPHLA": round(P(y, s5), 4),
        "AUPRC_USDF5": round(average_precision_score(y, post), 4),
        "P@1%_USDF5": round(hits_at_K(y, post, 0.01), 4),
        "P@1%_MHCFlurry": round(hits_at_K(y, s2, 0.01), 4),
        "P@1%_BigMHC": round(hits_at_K(y, s4, 0.01), 4),
        "R@20%_USDF5": round(y[np.argsort(-post, kind="stable")[:int(0.2 * len(y))]].sum() / y.sum(), 4),
        "Brier_USDF5cal": round(brier_score_loss(y, post_cal), 4),
        "Brier_MHCFlurrycal": round(brier_score_loss(y, ir_s2.transform(R5[:, 1])), 4),
        "Brier_BigMHCcal": round(brier_score_loss(y, ir_s4.transform(R5[:, 3])), 4),
        "LogLoss_USDF5cal": round(log_loss(y, post_cal), 4),
        "LogLoss_MHCFlurrycal": round(log_loss(y, ir_s2.transform(R5[:, 1])), 4),
        "ECE_USDF5cal": round(ece(y, post_cal), 4),
        "dAUROC_vs_MHCFlurry": round(md, 4), "ci95_vs_MF": f"[{lo:.4f},{hi:.4f}]",
        "dAUROC_vs_USDF4": round(md4, 4), "ci95_vs_U4": f"[{lo4:.4f},{hi4:.4f}]",
    }
    rows.append(row)
    npz_out[f"{name}_post5"] = post
    npz_out[f"{name}_y"] = y
    print(row, flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{WS}/ribfuse5_results.csv", index=False)
np.savez(f"{WS}/usdf5_scores.npz", **npz_out, int_post5=post_int5, y_int=y_int)
print(res.to_string(index=False), flush=True)
print("DONE", flush=True)
