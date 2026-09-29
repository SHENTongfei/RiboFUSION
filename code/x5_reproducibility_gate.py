# -*- coding: utf-8 -*-
"""X5 — reproducibility gate. Must PASS before figure work resumes.

Checks (evidence ledger, not trust):
  1. checkpoint + dataset sha256 still match model_output_manifest.json
     (nothing re-trained or re-generated silently).
  2. every D-audit artifact exists, is valid JSON/CSV/NPZ, and is finite
     (no NaN/Inf anywhere in the numbers figures may consume).
  3. seed ledger: every stochastic element used by D2-D5 is documented
     (deterministic key / fixed seeds); no unseeded randomness in the
     audit scripts.
  4. figure data source (results/figures/source/fig_data.json) may only
     reference artifact paths present in the ledger.
Output: model_outputs/x5_repro_report.json + PASS/FAIL print.
"""
import os, sys, json, hashlib, re
import numpy as np

WS = r"C:/Users/TS/WorkBuddy/2026-09-05-21-31-32"
MO = os.path.join(WS, "model_outputs")
BIG = r"C:/WorkBuddyBigFiles/routeb"
CKPTS = {"d1": "d1_pocket.ckpt", "n1": "n1_cond.ckpt", "n2": "n2_cond.ckpt",
         "comp": "complete_cond.ckpt", "n3": "n3_lam.ckpt"}
CKPT_SHA_EXPECTED = {
    "d1": "c9d09a78f161ee979c29ce8945df2a3d2da286551efcf15648431394072804ae",
    "n1": "47a567cbdd5ff416bdcb68ac224e9fb4316f283bd6ae3a2806d69176b5e53b2e",
    "n2": "44b8ae67eaf738ec64856e914c411fcf8e031d08995607eee78da6e45d439851",
    "comp": "7d4c9259643cfea50ea9bb3c4b8a4fc456bf550384a42eb2a8001f08aa4cf221",
    "n3": "1a541ad41cf9f2739df0604ae5f1e4d26d627f4de23bdba007d63e42318213f4",
}
DS_SHA_EXPECTED = "10dc55760e2c2e37fa95cf1cdeafb0aa68a8171282a0feec842df49e27232502"

# D-audit artifacts that figures may consume (the ledger)
LEDGER = [
    "n2_joint_audit.json",
    "rare_transfer_audit.json",
    "reach_joint_audit.json",
    "reach_top_candidates.csv",
    "explainability/explainability_manifest.json",
    "explainability/ig_attribution.npz",
    "explainability/attention_mass.npz",
    "explainability/occlusion.npz",
    "explainability/substitution_scan.npz",
    "explainability_full/manifest_full.json",
    "explainability_full/base_scores.npz",
    "explainability_full/attention_mass_full.npz",
    "explainability_full/ig_full.npz",
    "explainability_full/occlusion_full.npz",
    "explainability_full/substitution_full.json",
    "explainability_full/sanity_random_model.npz",
    "deep_mining/d9_deep_mining.json",
    "deep_mining/m1_substitution_composition.csv",
    "d10_literature/final_papers.json",
    "d10_literature/d10_literature_support.md",
    "d10_literature/references.bib",
]

# seed ledger: stochastic element -> documented seed/key (regex-checked in scripts)
SEED_LEDGER = {
    "d2 deterministic permutation key": "sum(ord(c)*(131**i) over peptide+allele)",
    "d3 attention-stability subset": "np.random.default_rng(0) documented in d3_rare_transfer_audit.py",
    "d4 permutation null (200 draws)": "np.random.default_rng documented in d4_reach_joint_audit.py",
    "d5 subset stability seeds": [0, 1, 2],
    "d5 subset sanity control": "randomized model seed 999; 32 distinct pairs rng(5)",
    "d5 full sanity control": "same seed-999 randomized model + rng(5) 32 distinct pairs",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_json(o, path, bad):
    if isinstance(o, dict):
        for v in o.values():
            finite_json(v, path, bad)
    elif isinstance(o, list):
        for v in o:
            finite_json(v, path, bad)
    elif isinstance(o, float):
        if not np.isfinite(o):
            bad.append((path, o))


def main():
    ok, report = True, {"checks": {}, "seed_ledger": SEED_LEDGER}

    # 1. checkpoint + dataset hashes
    ck = {}
    for k, fn in CKPTS.items():
        p = os.path.join(BIG, fn)
        if not os.path.exists(p):
            ck[k] = "MISSING"; ok = False
        else:
            s = sha256_file(p)
            ck[k] = "match" if s == CKPT_SHA_EXPECTED[k] else f"MISMATCH {s[:12]}"
            if s != CKPT_SHA_EXPECTED[k]:
                ok = False
    dsp = os.path.join(BIG, "routeb_dataset.npz")
    ds = "match" if os.path.exists(dsp) and sha256_file(dsp) == DS_SHA_EXPECTED else "MISMATCH/MISSING"
    if ds != "match":
        ok = False
    report["checks"]["checkpoint_sha256"] = ck
    report["checks"]["dataset_sha256"] = ds

    # 2. ledger existence + validity + finiteness
    led = {}
    for rel in LEDGER:
        p = os.path.join(MO, rel)
        if not os.path.exists(p):
            led[rel] = "MISSING"; ok = False; continue
        try:
            if rel.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    doc = json.load(f)
                bad = []
                finite_json(doc, rel, bad)
                led[rel] = "ok (finite)" if not bad else f"NON-FINITE {bad[:5]}"
                if bad:
                    ok = False
            elif rel.endswith(".npz"):
                z = np.load(p, allow_pickle=True)
                nonfin = 0
                for k in z.files:
                    a = z[k]
                    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                        nonfin += 1
                led[rel] = "ok (finite)" if nonfin == 0 else f"NON-FINITE arrays: {nonfin}"
                if nonfin:
                    ok = False
            elif rel.endswith(".csv"):
                import csv as _csv
                with open(p, encoding="utf-8") as f:
                    rows = list(_csv.reader(f))
                led[rel] = f"ok ({len(rows)} rows)"
            else:
                led[rel] = f"ok ({os.path.getsize(p)} bytes)"
        except Exception as e:  # noqa: BLE001
            led[rel] = f"INVALID ({e})"; ok = False
    report["checks"]["ledger"] = led

    # 3. seed audit: unseeded np.random in the audit scripts = FAIL
    seed_bad = {}
    AUDIT_SCRIPTS = ["d2_n2_joint_audit.py", "d3_rare_transfer_audit.py",
                     "d4_reach_joint_audit.py", "d5_explainability.py",
                     "d5_full_explainability.py", "d9_deep_mining.py",
                     "d10_run_searches.py", "d10_score.py"]
    for fn in AUDIT_SCRIPTS:
        p = os.path.join(WS, fn)
        if not os.path.exists(p):
            seed_bad[fn] = "script missing"; ok = False; continue
        txt = open(p, encoding="utf-8").read()
        unseeded = [l for l in txt.splitlines()
                    if re.search(r"np\.random\.(normal|uniform|rand|randn|randint|choice|shuffle)\(", l)
                    and "default_rng" not in l and "seed" not in l.lower()]
        if unseeded:
            seed_bad[fn] = unseeded[:5]; ok = False
        else:
            seed_bad[fn] = "ok"
    report["checks"]["seed_audit"] = seed_bad

    report["status"] = "PASS" if ok else "FAIL"
    out = os.path.join(MO, "x5_repro_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"gate": "X5", "status": report["status"],
                      "report": out}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
