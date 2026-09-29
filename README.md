# RiboFUSION

**RiboFUSION: an interpretable multimodal protein language model fusion framework for screening ncORF-derived epitopes with HLA-allele resolution**

RiboFUSION couples a translation-evidence **RiboEncoder backbone** (three attention-pooled PLM towers: ESM2 / ProtBERT / ESM-C), an **EM-guided five-source ensemble wing** (RiboEncoder + MHCFlurry + BigMHC + TransPHLA + a PLM logistic readout), and an **HLA-resolution apex** (transfer-based scoring of rare / no-AFND alleles) under one frozen, protocol-locked evaluation.

Headline results (all read from a single frozen table; see `frozen_results/method_metric_table.json` and Supplementary Table S1–S3):

| Cohort | Pairs | AUROC | Notes |
|---|---|---|---|
| Internal (10-fold CV) | 21,975 | **0.8663** | perfect P@1, second-place AUC |
| External Ouspenskaia | 15,404 | **0.9207** | +0.0303 over MHCFlurry, bootstrap CI clears zero |
| External Chong | 886 | **0.9481** | outranks every source and aggregate |
| Rare no-AFND alleles (apex) | 12,546 rare-bucket pairs | **0.9625** | vs 0.9167 on matched common alleles |

## Repository layout

```
├── run_demo.py                  # quick-start demo (verified end-to-end, CPU-only)
├── demo/
│   ├── demo_pair_val_1000.npz   # REAL 1,000-pair slice of the locked 30,028-pair internal val split
│   └── demo_pair_val_1000.csv   # the same slice as readable CSV
├── code/                        # pipeline scripts (as used for the paper runs)
├── frozen_results/
│   └── method_metric_table.json # THE frozen authority for every paper number
├── supplementary/
│   └── supplement.pdf           # Supplementary Material (Tables S1–S4, provenance)
└── README.md
```

## Quick start (2 minutes, CPU-only, no downloads)

```bash
pip install numpy scikit-learn
python run_demo.py
```

What this does:

1. Loads `demo/demo_pair_val_1000.npz` — **1,000 real peptide–allele pairs** sampled with a fixed seed (42) from the locked internal validation split (30,028 pairs). Every field is genuine model output from the paper's run: peptide sequence, HLA allele, presentation label `y`, the D1 pocket-base score, the three residual-reader scores, gate weights, features, and attention maps.
2. Recomputes the component-level scoring chain:
   - `D1 pocket base AUROC` — prints ≈ 0.89–0.90 on the demo slice (the **full** split scores exactly **0.9000**, the paper's locked D1 value).
   - `gate-weighted chain AUROC` — combines the readers with the locked cross-tower gate weights (0.62 / 0.23 / 0.15) and prints ≈ 0.87–0.88 (paper: **0.8707** on the full 10-fold CV).
3. Prints per-reader AUROCs so you can see each component's individual contribution.

If those numbers print in the expected ranges, your environment is correct and the demo data round-trips.

**Note on the demo's positive rate:** the val split stored here is positive-enriched (82.6% positives in the slice). The full internal *cohort* is 21,975 pairs at 27.2% positives — the demo is a slice of the model-output val matrix, not of the cohort table.

## Data: what is here and what is not

Everything in `demo/` is **real data** — a seeded subset of the actual locked run outputs. Nothing is synthetic.

The **full** per-pair matrices are large (45–250 MB per file, embeddings up to 93 MB) and are **not** in this repository. If you need them, open an issue. The full matrices live in the authors' archive as:

- `model_outputs_pair_val.npz` (30,028 × internal val pairs; the demo is a 1,000-pair seeded slice of exactly this file)
- `model_outputs_pair_ousp_grid_scores.npz` (15,404 Ousp pairs)
- `model_outputs_pair_chong_grid.npz` (886 Chong pairs)
- PLM embedding caches (ESM2-650M / ProtBERT / ESM-C)

**`frozen_results/method_metric_table.json` is the single authority for every number in the paper.** All eight methods × three cohorts × seven metrics (AUC, AP, Brier, LogLoss, P@1, R@20, ECE) are there. One provenance note is worth repeating: the USDF-4 Ouspenskaia value is the live-recomputed **0.9095**; a stale 0.9096 exists in an earlier official CSV (its input artifacts were lost) and must not be cited. This is documented in Supplementary S3.

## Full reproduction route

The `code/` directory contains the pipeline scripts as used for the paper, in execution order:

| Stage | Script | What it does |
|---|---|---|
| 0. PLM embeddings | `code/esmc_embed.py` | embed peptides with ESM2-650M / ProtBERT / ESM-C (weights downloaded separately from HuggingFace/EvolutionaryScale) |
| 1. Backbone training | `code/routeb_train.py` | allele-aware reader head (peptide CNN + allele groove CNN + MLP); heartbeat + checkpoint/resume built in |
| 2. D1 pocket reader | `code/d1_pocket_reader.py`, `code/train_d1_pocket.py` | groove-compatibility scoring (the D1 base of the component ladder) |
| 3. Residual readers | `code/complete_model.py` | N1 process-coupled / N2 ncORF-conditioned / N3 groove-transfer readers + composite + HRL |
| 4. Wing fusion | `code/usdf_fuse.py`, `code/usdf5_update.py` | EM-guided five-source aggregation (USDF-5) and four-source ablation (USDF-4) |
| 5. Frozen tables | `code/build_method_table.py` | writes `method_metric_table.json` — the paper's single number source |
| 6. Reproducibility gate | `code/x5_reproducibility_gate.py` | re-runs the locked eval and diffs against the frozen table |

Scripts were run with absolute local paths (e.g., `C:\WorkBuddyBigFiles\...`); point the `DATA`/`OUT` constants at your own paths before running. Python ≥3.10, `numpy`, `torch` (CPU works for stages 4–6; stage 0–1 want a GPU), `scikit-learn`.

The two external cohorts are public publications (Ouspenskaia et al. 2022; Chong et al. 2021 — see paper references); MHCFlurry, BigMHC and TransPHLA are pip-installable / downloadable from their own repositories.

## Provenance discipline

- Every headline number in the paper is read from `frozen_results/method_metric_table.json`. Nothing is recomputed after locking.
- The three provenance values in paper Fig. 4N are anchored against a 20,000-sample recompute.
- Unfavourable results (per-allele AUROC spread 0.55–0.93, weak IG-to-edit correlation 0.229, negative N3 step −0.0040, internal component reversal, the Chong four-way P@1 tie at 0.875) are disclosed in the paper and tabulated in Supplementary Table S4.

## Supplementary Material

`supplementary/supplement.pdf` contains: **S1** the frozen per-method metric tables for all three cohorts; **S2** the unfavourable results tabulated; **S3** the provenance and code note. It is referenced from the paper's Conclusions and Data-availability sections.

## License

Released under the MIT License for reproducibility. The trained model's scientific claims are covered by the paper; if you build on this work, please cite the paper.
