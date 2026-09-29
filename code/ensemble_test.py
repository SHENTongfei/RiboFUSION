# -*- coding: utf-8 -*-
"""Chong MHCflurry scoring + ensemble test (RIBOCAST logit + MHCflurry best percentile)."""
import os
os.environ["MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR"] = (
    "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32/mhcflurry_models/models.combined"
)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from mhcflurry import Class1AffinityPredictor

MET = "C:/Users/TS/WorkBuddy/sorf-mhc-present/results/metrics"
CACHE = "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32/mhcflurry_pep_cache.csv"
CHONG = "C:/Users/TS/WorkBuddy/sorf-mhc-present/data/external/chong2020/dataset_external_chong.npz"
OUT = "C:/Users/TS/WorkBuddy/2026-09-05-21-31-32"

ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:06", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*23:01", "HLA-A*24:02", "HLA-A*26:01", "HLA-A*30:01", "HLA-A*31:01",
    "HLA-A*32:01", "HLA-A*33:01",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*15:01", "HLA-B*18:01", "HLA-B*27:05",
    "HLA-B*35:01", "HLA-B*38:01", "HLA-B*39:01", "HLA-B*40:01", "HLA-B*44:02",
    "HLA-B*44:03", "HLA-B*51:01", "HLA-B*53:01", "HLA-B*57:01", "HLA-B*58:01",
    "HLA-C*01:02", "HLA-C*03:04", "HLA-C*04:01", "HLA-C*05:01", "HLA-C*06:02",
    "HLA-C*07:01", "HLA-C*07:02", "HLA-C*08:02", "HLA-C*12:03", "HLA-C*15:02",
]

# ---- 1. score Chong peptides with MHCflurry (cache-aware) ----
d = np.load(CHONG, allow_pickle=True)
seqs = d["pep_seq"].astype(str)
y_chong = d["Y"].ravel().astype(int)
cache = pd.read_csv(CACHE)
done = set(cache["seq"])
todo = sorted(set(seqs) - done)
print("chong peptides:", len(set(seqs)), "| new to score:", len(todo))

if todo:
    predictor = Class1AffinityPredictor.load(os.environ["MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR"])
    frames = []
    for i in range(0, len(todo), 2000):
        batch = todo[i:i + 2000]
        fs = []
        for allele in ALLELES:
            fs.append(predictor.predict_to_dataframe(peptides=batch, alleles=[allele] * len(batch)))
        df = pd.concat(fs, ignore_index=True).rename(columns={
            "peptide": "seq", "prediction": "affinity", "prediction_percentile": "percentile"})
        g = df.groupby("seq").agg(best_percentile=("percentile", "min"),
                                  min_affinity=("affinity", "min"))
        g["best_allele"] = df.loc[df.groupby("seq")["percentile"].idxmin(), "allele"].values
        frames.append(g.reset_index())
        print(f"  scored {min(i+2000, len(todo))}/{len(todo)}", flush=True)
    cache = pd.concat([cache] + frames, ignore_index=True).drop_duplicates("seq", keep="last")
    cache.to_csv(CACHE, index=False)

sm = cache.set_index("seq")[["best_percentile", "min_affinity"]]
chong_bp = sm.loc[seqs, "best_percentile"].values.astype(float)

# ---- 2. assemble per-cohort table: y, ribocast logit, mhcflurry percentile ----
def load_ext(name, npz_path):
    dd = np.load(npz_path, allow_pickle=True)
    preds = pd.read_csv(f"{MET}/preds_external_{name}.csv")
    assert len(preds) == len(dd["Y"]), (name, len(preds), len(dd["Y"]))
    preds["seq"] = dd["pep_seq"].astype(str)
    return preds

ousp = load_ext("ouspenskaia", "C:/Users/TS/WorkBuddy/sorf-mhc-present/data/external/dataset_external.npz")
chong = pd.DataFrame({"row": np.arange(len(seqs)), "y": y_chong,
                      "ribocast_logit": pd.read_csv(f"{MET}/preds_external_chong.csv")["ribocast_logit"],
                      "seq": seqs})

for df_, bp in [(ousp, sm.loc[ousp["seq"], "best_percentile"].values.astype(float)),
                (chong, chong_bp)]:
    df_["mhcflurry_pct"] = bp

results = []
from scipy.stats import wilcoxon
for name, df_ in [("Ouspenskaia", ousp), ("Chong", chong)]:
    y = df_["y"].values
    rl = df_["ribocast_logit"].values
    mf = -df_["mhcflurry_pct"].values  # higher = better
    a_r = roc_auc_score(y, rl)
    a_m = roc_auc_score(y, mf)
    # simple score-level ensemble (rank-average)
    from scipy.stats import rankdata
    ens = (rankdata(rl) + rankdata(mf)) / (2 * len(y))
    a_e = roc_auc_score(y, ens)
    results.append({"cohort": name, "n": len(y),
                    "auroc_ribocast": round(a_r, 4),
                    "auroc_mhcflurry_best": round(a_m, 4),
                    "auroc_rank_ensemble": round(a_e, 4)})
    # significance of ribocast vs mhcflurry (DeLong-style bootstrap)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], rl) - roc_auc_score(y[idx], mf))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    results[-1]["delta_95CI"] = f"[{lo:+.3f},{hi:+.3f}]"
    print(results[-1], flush=True)

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/external_baseline_comparison.csv", index=False)
print(res.to_string(index=False))
print("DONE")
