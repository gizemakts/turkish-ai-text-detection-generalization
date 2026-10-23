"""Manual-only seed-grid extension delegating all training math to the frozen terminal runner."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
CFG=ROOT/"results/robustness_analyses/external/EXTERNAL_5SEED_EXTENSION_CONFIG.json"
EXPECTED_BASE="0a1314ee51fe79f820404ec7cd4e845709411a43ff0388376170f602f300d76c"
EXPECTED_RUNNER="96f2af846f15cb3bb66bea7e8addba3f49da0c1b3457f96b8b1eb1d46f30e558"
EXPECTED_FOLDS="16ffae53a5d9027a95e12f3975fd062f17038ef5bf8970a74dd3ef7339d9a100"
EXPECTED_DATA="ea5797ac19348957472626d1a23b740de531cd19577f613e1a73894e34d66b44"
EXPECTED_MODEL="dbmdz/bert-base-turkish-cased"
EXPECTED_REVISION="b6e1de16c983e0f2c70664591ea3f22810072608"
ADDED=(137,3407)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
    c=json.loads(CFG.read_text(encoding="utf-8")); assert c["status"]=="FROZEN_NOT_EXECUTED" and tuple(c["added_seeds"])==ADDED
    assert sha(ROOT/"configs/external_cross_corpus.json")==EXPECTED_BASE
    assert sha(ROOT/"src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py")==EXPECTED_RUNNER
    assert sha(ROOT/"data/study/nested_folds.csv")==EXPECTED_FOLDS
    for obj in ("ce","supcon"):
      for s in ADDED:
       for f in range(5):
        p=ROOT/f"results/primary/{obj}/seed_{s}/outer_{f}_run.json"; assert p.is_file(), (obj,s,f)
        r=json.loads(p.read_text(encoding="utf-8")); assert r["objective"]==obj and r["seed"]==s and r["outer_fold"]==f
        assert r["dataset_sha256"]==EXPECTED_DATA and r["fold_sha256"]==EXPECTED_FOLDS
        assert r["model"]==EXPECTED_MODEL and r["revision"]==EXPECTED_REVISION
    return {"status":"PASS_STATIC_EXTENSION","config_sha256":sha(CFG),"base_sha256":EXPECTED_BASE,"runner_sha256":EXPECTED_RUNNER,"fold_sha256":EXPECTED_FOLDS,"added_seeds":ADDED,"target_overlap_with_existing_three_seed_grid":False}
def train(obj,seed,fold):
    if seed not in ADDED or obj not in ("ce","supcon") or fold not in range(5): raise ValueError("outside frozen extension")
    from turkish_ai_detection.experiments.external_cross_corpus import rerun_fit as r
    r.COMMON_SEEDS=(42,137,314,2026,3407)
    record=r.run_one(obj,seed,fold,False)
    rp=ROOT/f"artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun/{obj}_seed_{seed}_outer_{fold}/run.json"
    doc=json.loads(rp.read_text(encoding="utf-8")); doc["five_seed_extension"]={"config_path":CFG.relative_to(ROOT).as_posix(),"config_sha256":sha(CFG),"added_seed":seed,"base_recipe_unchanged":True,"wrapper_sha256":sha(Path(__file__)),"recorded_utc":datetime.now(timezone.utc).isoformat()}; rp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"status":"COMPLETE","run_record":rp.relative_to(ROOT).as_posix()}
def infer(ds,method,seed,fold):
    if seed not in ADDED or method not in ("CE_ONLY_MATCHED","SUPCON") or fold not in range(5): raise ValueError("outside frozen extension")
    from turkish_ai_detection.experiments.external_cross_corpus import inference as inf
    inf.SEEDS=(42,137,314,2026,3407)
    return inf.run_one(ROOT/"configs/external_cross_corpus.json",ds,method,seed,fold)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--validate-only",action="store_true"); p.add_argument("--train",action="store_true"); p.add_argument("--infer",action="store_true"); p.add_argument("--objective",choices=["ce","supcon"]); p.add_argument("--method",choices=["CE_ONLY_MATCHED","SUPCON"]); p.add_argument("--seed",type=int); p.add_argument("--outer-fold",type=int); p.add_argument("--dataset-id",choices=["renklier2026","tum2024_tr","bostancioglu2026_tr"]); a=p.parse_args(); validate()
    if a.validate_only: print(json.dumps(validate(),indent=2)); return
    if a.train: print(json.dumps(train(a.objective,a.seed,a.outer_fold),indent=2)); return
    if a.infer: print(json.dumps(infer(a.dataset_id,a.method,a.seed,a.outer_fold),indent=2)); return
    p.error("choose --validate-only, --train, or --infer")
if __name__=="__main__": main()
