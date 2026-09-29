# -*- coding: utf-8 -*-
"""USDF: Unsupervised Source-reliability Fusion (RiboFuse operating point).

Principle: learned weights trained on internal data overtrust the in-domain
strongest source and fail under distribution shift. Instead, estimate source
reliability ON the deployment cohort from UNLABELED score distributions only
(transductive, zero label access):

  Latent z in {0,1} (presentation), z ~ Bern(pi).
  Each source s observes rank_s | z=k ~ N(mu_{s,k}, sigma_{s,k}).
  Joint EM on the target cohort's rank matrix alone (Dawid-Skene 1979 for
  continuous observations). Initialization from internal class-conditional
  rank moments (positive component = higher mean, order enforced).

Two operating variants (selected by INTERNAL condition-transfer CV only):
  USDF-post : fused score = EM posterior P(z=1 | x)
  USDF-rank : rank fusion with weights w_s prop. to per-source d' separation

No external labels touch fitting, weighting, or selection.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WS = "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32"
EPS = 1e-3

def run_em(R, mu0, mu1, pi, iters=300):
    """Joint EM. R: (n,3) ranks. mu0/mu1: (3,) init component means. Returns
    posterior p(z=1|x), per-source (mu0,mu1,sigma0,sigma1)."""
    n, S = R.shape
    mu0 = mu0.copy(); mu1 = mu1.copy()
    s0 = np.full(S, 0.15); s1 = np.full(S, 0.15)
    for _ in range(iters):
        # E-step (log-space for stability)
        logp1 = np.log(pi + EPS) + sum(
            -0.5 * ((R[:, s] - mu1[s]) / s1[s]) ** 2 - np.log(s1[s]) for s in range(S))
        logp0 = np.log(1 - pi + EPS) + sum(
            -0.5 * ((R[:, s] - mu0[s]) / s0[s]) ** 2 - np.log(s0[s]) for s in range(S))
        m = np.maximum(logp1, logp0)
        r1 = np.exp(logp1 - m); r0 = np.exp(logp0 - m)
        g1 = r1 / (r1 + r0)                       # (n,)
        # M-step
        pi = g1.mean()
        for s in range(S):
            w1 = g1 + EPS; w0 = (1 - g1) + EPS
            mu1[s] = np.clip((w1 * R[:, s]).sum() / w1.sum(), mu0[s] + EPS, 1.0)
            mu0[s] = np.clip((w0 * R[:, s]).sum() / w0.sum(), 0.0, mu1[s] - EPS)
            s1[s] = max(np.sqrt((w1 * (R[:, s] - mu1[s]) ** 2).sum() / w1.sum()), EPS)
            s0[s] = max(np.sqrt((w0 * (R[:, s] - mu0[s]) ** 2).sum() / w0.sum()), EPS)
    return g1, mu0, mu1, s0, s1, pi

def usdf_fit(R, mu0_init, mu1_init, pi_init):
    g1, mu0, mu1, s0, s1, pi = run_em(R, mu0_init, mu1_init, pi_init)
    dprime = (mu1 - mu0) / np.sqrt(s1 ** 2 + s0 ** 2)
    w = dprime / dprime.sum()
    return {"post": g1, "w": w, "dprime": dprime,
            "mu0": mu0, "mu1": mu1, "pi": pi}

# ---------------- load caches ----------------
d = np.load(f"{WS}/ribocast_internal_oof.npz", allow_pickle=True)
y_int = d["y"].ravel().astype(int)
cond_int = d["cond"].astype(int)

R_int = np.load(f"{WS}/_R_int.npy")
R_ousp = np.load(f"{WS}/_ext_Ousp_R.npy"); y_ousp = np.load(f"{WS}/_ext_Ousp_y.npy")
R_chong = np.load(f"{WS}/_ext_Chong_R.npy"); y_chong = np.load(f"{WS}/_ext_Chong_y.npy")

# internal init from class-conditional rank moments (labels used ONLY here,
# on internal OOF -- never on any external cohort)
mu1_init = np.array([R_int[y_int == 1, s].mean() for s in range(3)])
mu0_init = np.array([R_int[y_int == 0, s].mean() for s in range(3)])
pi_init = y_int.mean()
print("init mu1:", mu1_init.round(3), "mu0:", mu0_init.round(3), "pi:", round(pi_init, 3))

print("\n=== internal OOF (labels for eval only) ===")
f_int = usdf_fit(R_int, mu0_init, mu1_init, pi_init)
print("USDF-post AUROC:", round(roc_auc_score(y_int, f_int["post"]), 4))
print("USDF-rank AUROC:", round(roc_auc_score(y_int, (f_int["w"] * R_int).sum(1)), 4),
      " w:", f_int["w"].round(3), " d':", f_int["dprime"].round(3))
print("uniform-rank   :", round(roc_auc_score(y_int, R_int.mean(1)), 4))

print("\n=== internal condition-transfer CV (EM fit on held-out condition, unlabeled) ===")
levels = [l for l in sorted(set(cond_int.tolist())) if (cond_int == l).sum() >= 500]
res_cv = {"post": [], "rank": [], "unif": []}
for lv in levels:
    m = cond_int == lv
    f = usdf_fit(R_int[m], mu0_init, mu1_init, pi_init)
    a_post = roc_auc_score(y_int[m], f["post"])
    a_rank = roc_auc_score(y_int[m], (f["w"] * R_int[m]).sum(1))
    a_unif = roc_auc_score(y_int[m], R_int[m].mean(1))
    res_cv["post"].append(a_post); res_cv["rank"].append(a_rank); res_cv["unif"].append(a_unif)
    print(f"  cond {lv} (n={m.sum()}): post={a_post:.4f} rank={a_rank:.4f} unif={a_unif:.4f}  w={f['w'].round(2)}")
print("transfer-CV mean: post=%.4f rank=%.4f unif=%.4f" %
      (np.mean(res_cv["post"]), np.mean(res_cv["rank"]), np.mean(res_cv["unif"])))

print("\n=== external (zero label access; EM fit on cohort scores only) ===")
for name, R, y in [("Ouspenskaia", R_ousp, y_ousp), ("Chong", R_chong, y_chong)]:
    f = usdf_fit(R, mu0_init, mu1_init, pi_init)
    print(f"{name}: pi_hat={f['pi']:.3f}  d'={f['dprime'].round(3)}  w={f['w'].round(3)}")
    print("  USDF-post:", round(roc_auc_score(y, f["post"]), 4))
    print("  USDF-rank:", round(roc_auc_score(y, (f["w"] * R).sum(1)), 4))
    print("  uniform  :", round(roc_auc_score(y, R.mean(1)), 4))

# save per-sample fused scores for the chosen downstream reporting
np.savez(f"{WS}/usdf_scores.npz",
         int_post=f_int["post"], int_rank=(f_int["w"] * R_int).sum(1), y_int=y_int)
print("\nDONE")
