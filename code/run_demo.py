# -*- coding: utf-8 -*-
"""
RiboFUSION quick-start demo.

Loads the REAL internal-validation demo slice (1,000 peptide-allele pairs sampled
with a fixed seed from the locked 30,028-pair val split) and re-runs the
component-level scoring chain so you can verify the frozen paper numbers on your
own machine:

    d1_prob  (D1 pocket base score)          -> paper: D1 AUROC = 0.9000
    d1_prob + gate-weighted residual readers -> paper: RiboEncoder internal
                                                AUROC = 0.8707 (full 10-fold CV)

Run:
    python run_demo.py

Requires: numpy, scikit-learn. No GPU, no PLM weights, no large downloads.
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo", "demo_pair_val_1000.npz")

# Cross-tower gate weights locked in the training run reported in the paper
# (Section 3.2): ESM2 / ProtBERT / ESM-C towers.
GATE = (0.62, 0.23, 0.15)


def main():
    z = np.load(DEMO, allow_pickle=True)
    y = z["y"].astype(int)
    print("Loaded demo slice: %d peptide-allele pairs, %d positives (%.1f%%)"
          % (y.shape[0], y.sum(), 100.0 * y.mean()))
    print("(This val split is positive-enriched; the full internal cohort is")
    print(" 21,975 pairs at 27.2 percent positives -- see paper Section 3.5.)\n")

    # --- Component 1: D1 pocket base -------------------------------------
    auc_d1 = roc_auc_score(y, z["d1_prob"])
    print("D1 pocket base      AUROC = %.4f   (paper: 0.9000)" % auc_d1)

    # --- Component chain: gate-weighted residual readers ------------------
    score = (z["d1_prob"]
             + GATE[0] * z["n1_residual"]
             + GATE[1] * z["n2_residual"]
             + GATE[2] * z["n3_transfer"])
    auc_chain = roc_auc_score(y, score)
    print("Gate-weighted chain AUROC = %.4f   (paper: RiboEncoder 0.8707 on the" % auc_chain)
    print("                                    full 10-fold CV; this demo slice")
    print("                                    is a random 1,000-pair subset)")

    # --- Per-readout sanity -----------------------------------------------
    for name in ["n1_residual", "n2_residual", "n3_transfer"]:
        print("  single reader %-13s AUROC = %.4f" % (name, roc_auc_score(y, z[name])))

    print("\nOn this 1,000-pair random slice the D1 AUROC should print close to")
    print("0.89-0.90 (the full split gives exactly 0.9000); if it does, your demo")
    print("slice and environment are good.")
    print("\nFrozen per-method metrics for every cohort are in")
    print("frozen_results/method_metric_table.json and Supplementary Table S1.")


if __name__ == "__main__":
    main()
