# -*- coding: utf-8 -*-
"""One-time Condition-B unblinding and paired Condition-A comparison."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from turkish_ai_detection.data.study import resolve_frozen_record_path, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import _pauc  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402

CONFIG = ROOT / "configs" / "artifact_sensitivity.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/artifact_sensitivity/run_fit.py"
STATS = Path(__file__)
FREEZE = ROOT / "provenance" / "experiments" / "artifact_sensitivity" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "artifact_sensitivity"
RESULTS = OUT / "final_statistics"

PRIMARY_OUT = ROOT / "results" / "primary"
PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = PRIMARY_OUT / "final_statistics" / "final_statistics_manifest.json"

SEEDS = [42,314,2026]
OUTERS = [0,1,2,3,4]
OBJECTIVES = ["ce","supcon"]
METRICS = ["roc_auc","average_precision","standardized_pauc_003"]


def triplet(frame):
    y = frame.label.astype(str).eq("AI").astype(int).to_numpy()
    s = frame.raw_decision_score.to_numpy(float)
    return {
        "roc_auc": float(roc_auc_score(y,s)),
        "average_precision": float(average_precision_score(y,s)),
        "standardized_pauc_003": float(_pauc(y,s)),
    }


def verify_freeze(cfg):
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid Condition-B freeze")
    policy = validate_current_migrated_reproduction(ROOT, "artifact_sensitivity", RUNNER)
    return freeze, policy


def load_b(freeze, policy):
    frames = {}
    for obj in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in OUTERS:
                base = OUT / obj / f"seed_{seed}"
                pred = base / f"outer_{outer}_predictions.csv"
                side = base / f"outer_{outer}_run.json"
                if not pred.is_file() or not side.is_file():
                    raise RuntimeError(f"missing Condition-B fit: {obj}/{seed}/{outer}")
                run = json.loads(side.read_text(encoding="utf-8"))
                if run.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete Condition-B fit: {obj}/{seed}/{outer}")
                if run["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"Condition-B prediction hash mismatch: {obj}/{seed}/{outer}")
                expected_config, expected_runner = expected_run_hashes(run, freeze, policy, config_key="config", runner_key="runner")
                if run["runner_sha256"] != expected_runner or run["config_sha256"] != expected_config:
                    raise RuntimeError(f"Condition-B runner/config mismatch: {obj}/{seed}/{outer}")
                if run["preexecution_freeze_sha256"] != sha256_file(FREEZE):
                    raise RuntimeError(f"Condition-B freeze mismatch: {obj}/{seed}/{outer}")
                f = pd.read_csv(pred)
                if len(f) != 600 or f.text_id.nunique() != 600:
                    raise RuntimeError(f"Condition-B fold shape mismatch: {obj}/{seed}/{outer}")
                parts.append(f)
            pooled = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            if len(pooled) != 3000 or pooled.text_id.nunique() != 3000 or pooled.pair_id.nunique() != 1500:
                raise RuntimeError(f"Condition-B pooled OOF mismatch: {obj}/{seed}")
            frames[(obj,seed)] = pooled
    return frames


def load_a(cfg):
    ref = cfg["primary_condition_a_reference"]
    checks = [
        (PRIMARY_RUNNER, ref["primary_runner_sha256"]),
        (PRIMARY_CONFIG, ref["primary_config_sha256"]),
        (PRIMARY_FREEZE, ref["primary_preexecution_freeze_sha256"]),
        (PRIMARY_MANIFEST, ref["primary_final_statistics_manifest_sha256"]),
    ]
    if not all(path.is_file() and sha256_file(path) == expected for path, expected in checks):
        raise RuntimeError("Condition-A primary lineage mismatch")
    frames = {}
    for obj in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in OUTERS:
                base = PRIMARY_OUT / obj / f"seed_{seed}"
                pred = base / f"outer_{outer}_predictions.csv"
                side = base / f"outer_{outer}_run.json"
                if not pred.is_file() or not side.is_file():
                    raise RuntimeError(f"missing Condition-A reference: {obj}/{seed}/{outer}")
                run = json.loads(side.read_text(encoding="utf-8"))
                if run["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"Condition-A prediction hash mismatch: {obj}/{seed}/{outer}")
                parts.append(pd.read_csv(pred))
            pooled = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            if len(pooled) != 3000 or pooled.text_id.nunique() != 3000:
                raise RuntimeError(f"Condition-A pooled OOF mismatch: {obj}/{seed}")
            frames[(obj,seed)] = pooled
    return frames


def check_alignment(a,b):
    keys = ["text_id","pair_id","label","domain","generator"]
    ref = b[("ce",SEEDS[0])][keys].reset_index(drop=True)
    for name, frames in (("A",a),("B",b)):
        for obj in OBJECTIVES:
            for seed in SEEDS:
                if not frames[(obj,seed)][keys].reset_index(drop=True).equals(ref):
                    raise RuntimeError(f"paired alignment failed: condition={name}, objective={obj}, seed={seed}")


def observed_tables(a,b):
    per_seed = []
    for seed in SEEDS:
        row_base = {"seed": seed}
        vals = {}
        for cond_name, frames in (("A",a),("B",b)):
            for obj in OBJECTIVES:
                vals[(cond_name,obj)] = triplet(frames[(obj,seed)])
                for metric in METRICS:
                    per_seed.append({
                        "seed": seed,
                        "condition": cond_name,
                        "objective": obj,
                        "metric": metric,
                        "value": vals[(cond_name,obj)][metric],
                    })
    long = pd.DataFrame(per_seed)

    method_summary = []
    for (condition, objective, metric), g in long.groupby(["condition","objective","metric"], sort=True):
        v = g.value.to_numpy(float)
        method_summary.append({
            "condition": condition,
            "objective": objective,
            "metric": metric,
            "mean_across_3_seeds": float(v.mean()),
            "sd_across_3_seeds": float(v.std(ddof=1)),
            "min_seed_value": float(v.min()),
            "max_seed_value": float(v.max()),
        })

    delta_rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in METRICS:
            def get(c,o):
                return float(long[(long.seed==seed)&(long.condition==c)&(long.objective==o)&(long.metric==metric)].value.iloc[0])
            a_ce,a_sc,b_ce,b_sc = get("A","ce"),get("A","supcon"),get("B","ce"),get("B","supcon")
            row[f"condition_a_supcon_minus_ce_{metric}"] = a_sc-a_ce
            row[f"condition_b_supcon_minus_ce_{metric}"] = b_sc-b_ce
            row[f"delta_change_b_minus_a_{metric}"] = (b_sc-b_ce)-(a_sc-a_ce)
            row[f"ce_b_minus_a_{metric}"] = b_ce-a_ce
            row[f"supcon_b_minus_a_{metric}"] = b_sc-a_sc
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    paired_summary = []
    for metric in METRICS:
        for quantity, col in [
            ("condition_a_supcon_minus_ce", f"condition_a_supcon_minus_ce_{metric}"),
            ("condition_b_supcon_minus_ce", f"condition_b_supcon_minus_ce_{metric}"),
            ("delta_change_b_minus_a", f"delta_change_b_minus_a_{metric}"),
            ("ce_b_minus_a", f"ce_b_minus_a_{metric}"),
            ("supcon_b_minus_a", f"supcon_b_minus_a_{metric}"),
        ]:
            v = deltas[col].to_numpy(float)
            paired_summary.append({
                "quantity": quantity,
                "metric": metric,
                "mean_across_3_seeds": float(v.mean()),
                "sd_across_3_seeds": float(v.std(ddof=1)),
            })
    return long, pd.DataFrame(method_summary), deltas, pd.DataFrame(paired_summary)


def metric_from(y,s):
    return {
        "roc_auc": float(roc_auc_score(y,s)),
        "average_precision": float(average_precision_score(y,s)),
        "standardized_pauc_003": float(_pauc(y,s)),
    }


def bootstrap(cfg,a,b):
    B = int(cfg["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    ref = b[("ce",SEEDS[0])]
    pair_ids = np.asarray(sorted(ref.pair_id.unique()))
    idx_map = {p: np.flatnonzero(ref.pair_id.to_numpy()==p) for p in pair_ids}
    if len(pair_ids) != 1500 or any(len(v)!=2 for v in idx_map.values()):
        raise RuntimeError("invalid 1,500-pair bootstrap structure")
    y_full = ref.label.astype(str).eq("AI").astype(int).to_numpy()
    scores = {}
    for c,frames in (("A",a),("B",b)):
        for obj in OBJECTIVES:
            for seed in SEEDS:
                scores[(c,obj,seed)] = frames[(obj,seed)].raw_decision_score.to_numpy(float)

    rows = []
    for bi in range(1,B+1):
        sampled = rng.choice(pair_ids,size=len(pair_ids),replace=True)
        idx = np.concatenate([idx_map[p] for p in sampled])
        y = y_full[idx]
        q = {m: {name: [] for name in [
            "condition_a_supcon_minus_ce",
            "condition_b_supcon_minus_ce",
            "delta_change_b_minus_a",
            "ce_b_minus_a",
            "supcon_b_minus_a",
        ]} for m in METRICS}
        for seed in SEEDS:
            A_CE = metric_from(y,scores[("A","ce",seed)][idx])
            A_SC = metric_from(y,scores[("A","supcon",seed)][idx])
            B_CE = metric_from(y,scores[("B","ce",seed)][idx])
            B_SC = metric_from(y,scores[("B","supcon",seed)][idx])
            for m in METRICS:
                da = A_SC[m]-A_CE[m]
                db = B_SC[m]-B_CE[m]
                q[m]["condition_a_supcon_minus_ce"].append(da)
                q[m]["condition_b_supcon_minus_ce"].append(db)
                q[m]["delta_change_b_minus_a"].append(db-da)
                q[m]["ce_b_minus_a"].append(B_CE[m]-A_CE[m])
                q[m]["supcon_b_minus_a"].append(B_SC[m]-A_SC[m])
        row = {"bootstrap_id": bi}
        for m in METRICS:
            for name, vals in q[m].items():
                row[f"{name}_{m}"] = float(np.mean(vals))
        rows.append(row)
    draws = pd.DataFrame(rows)

    summary = []
    for m in METRICS:
        for name in [
            "condition_a_supcon_minus_ce",
            "condition_b_supcon_minus_ce",
            "delta_change_b_minus_a",
            "ce_b_minus_a",
            "supcon_b_minus_a",
        ]:
            vals = draws[f"{name}_{m}"].to_numpy(float)
            summary.append({
                "quantity": name,
                "metric": m,
                "bootstrap_unit": "pair_id",
                "replicates": B,
                "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
                "bootstrap_mean": float(vals.mean()),
                "ci95_lower": float(np.quantile(vals,0.025)),
                "ci95_upper": float(np.quantile(vals,0.975)),
                "inference_scope": cfg["statistics"]["inference_scope"],
            })
    return draws, pd.DataFrame(summary)


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze, policy = verify_freeze(cfg)
    b = load_b(freeze, policy)
    a = load_a(cfg)
    check_alignment(a,b)
    long, method_summary, deltas, paired_summary = observed_tables(a,b)
    draws, boot_summary = bootstrap(cfg,a,b)

    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_seed_condition_metrics.csv": long,
        "method_condition_summary.csv": method_summary,
        "condition_seed_deltas.csv": deltas,
        "paired_delta_summary.csv": paired_summary,
        "paired_pair_bootstrap_draws.csv": draws,
        "paired_pair_bootstrap_summary.csv": boot_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS/name,index=False)

    manifest = {
        "schema_version": "combined-v2-condition-b-final-statistics-v1",
        "status": "COMPLETE",
        "unblinded": True,
        "fits_verified": 30,
        "seeds": SEEDS,
        "outer_folds": OUTERS,
        "statistics_config": cfg["statistics"],
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(RUNNER),
        "statistics_sha256": sha256_file(STATS),
        "freeze_sha256": sha256_file(FREEZE),
        "primary_condition_a_reference_manifest_sha256": sha256_file(PRIMARY_MANIFEST),
        "outputs": {name: sha256_file(RESULTS/name) for name in outputs},
    }
    write_json(RESULTS/"final_statistics_manifest.json", manifest)
    print(json.dumps({
        "status":"COMPLETE",
        "fits_verified":30,
        "results_directory":str(RESULTS),
        "results_are_now_unblinded":True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
