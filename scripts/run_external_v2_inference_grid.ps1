[CmdletBinding()]
param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-gpu\Scripts\python.exe"
$Config = Join-Path $Root "configs\external_cross_corpus.json"
$Freeze = Join-Path $Root "provenance\experiments\external_cross_corpus_v2\PREEXECUTION_FREEZE.json"
$PredecessorFreeze = Join-Path $Root "provenance\experiments\external_cross_corpus_v2\PREEXECUTION_FREEZE_PREDECESSOR_20261010_LIT_RUNTIME.json"
$InferenceModule = "turkish_ai_detection.experiments.external_cross_corpus.inference"
$FreezeModule = "turkish_ai_detection.experiments.external_cross_corpus.freeze_v2"
$InferencePath = Join-Path $Root "src\turkish_ai_detection\experiments\external_cross_corpus\inference.py"
$PresealRoot = Join-Path $Root "artifacts\predictions\external_cross_corpus\preseal"
$ExpectedOutputColumns = @(
    "external_id",
    "text_sha256",
    "method",
    "seed",
    "outer_fold",
    "raw_score_or_logit"
)
$ExpectedInputColumns = @("external_id", "text", "text_sha256")
$ExpectedMethods = @(
    "CE_ONLY_MATCHED",
    "SUPCON",
    "LIT2026_REIMPL_V1_1"
)
$ExpectedTotal = 135
$ExpectedPredecessorFreezeSha256 = "333e533b1a96454992bee9b5ac3fc892a0e9be781678eae07d032bc93e87a5eb"
$ExpectedPredecessorInternalIdentity = "44eb08420d0ec0bbcf845a166e17929cb9219d96e65380797c9009dccf225034"
$PredecessorInferenceSha256 = $null

function Fail([string]$Message) {
    throw $Message
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-RelativePath([string]$Path) {
    $resolvedRoot = (Resolve-Path $Root).Path
    $resolvedPath = (Resolve-Path $Path).Path
    return $resolvedPath.Substring($resolvedRoot.Length).TrimStart("\").Replace("\", "/")
}

function Assert-Equal([object]$Actual, [object]$Expected, [string]$Message) {
    if ($Actual -ne $Expected) {
        Fail "$Message. Expected '$Expected', got '$Actual'."
    }
}

function Assert-ArrayEqual([object[]]$Actual, [object[]]$Expected, [string]$Message) {
    $actualText = ($Actual -join "|")
    $expectedText = ($Expected -join "|")
    Assert-Equal $actualText $expectedText $Message
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "Missing JSON: $Path"
    }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Get-CellPaths([string]$DatasetId, [string]$Method, [int]$Seed, [int]$OuterFold) {
    $cellRoot = Join-Path (Join-Path (Join-Path $PresealRoot $DatasetId) $Method) "seed_$Seed"
    return @{
        Csv = Join-Path $cellRoot "outer_$OuterFold.csv"
        Sidecar = Join-Path $cellRoot "outer_$OuterFold.json"
    }
}

function Get-DatasetBinding([object]$Config, [string]$DatasetId) {
    foreach ($binding in @($Config.external_data_bindings.primary)) {
        if ($binding.dataset_id -eq $DatasetId) {
            return $binding
        }
    }
    Fail "Dataset is not present in the frozen primary binding set: $DatasetId"
}

function Verify-PredecessorFreeze() {
    if (-not (Test-Path -LiteralPath $PredecessorFreeze -PathType Leaf)) {
        Fail "Missing authorized predecessor freeze archive: $PredecessorFreeze"
    }
    Assert-Equal (Get-Sha256 $PredecessorFreeze) $ExpectedPredecessorFreezeSha256 "Predecessor freeze file hash mismatch"
    $predecessorDocument = Read-Json $PredecessorFreeze
    Assert-Equal $predecessorDocument.freeze_sha256 $ExpectedPredecessorInternalIdentity "Predecessor internal identity mismatch"
    $code = @"
import hashlib
import json
from pathlib import Path
from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import (
    canonical_json,
    scientific_config_identity,
    sha256_bytes,
)
root = Path.cwd()
freeze_path = root / "provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE_PREDECESSOR_20261010_LIT_RUNTIME.json"
config_path = root / "configs/external_cross_corpus.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
config = json.loads(config_path.read_text(encoding="utf-8"))
recorded = freeze.pop("freeze_sha256", None)
if recorded != sha256_bytes(canonical_json(freeze)):
    raise RuntimeError("predecessor freeze self-hash mismatch")
if freeze["contract"]["path"] != "configs/external_cross_corpus.json":
    raise RuntimeError("predecessor contract path mismatch")
if freeze["contract"]["scientific_identity_sha256"] != scientific_config_identity(config):
    raise RuntimeError("predecessor scientific config identity mismatch")
inference_rel = "src/turkish_ai_detection/experiments/external_cross_corpus/inference.py"
inference_sha = None
for record in freeze["bindings"]:
    path = record["path"]
    expected = record["sha256"]
    if path == inference_rel:
        inference_sha = expected
        continue
    actual_path = root / path
    digest = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"predecessor binding mismatch: {path}")
if not isinstance(inference_sha, str) or len(inference_sha) != 64:
    raise RuntimeError("predecessor inference binding missing")
print(inference_sha)
"@
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($code)
    )
    $bootstrap = "import base64;exec(base64.b64decode(chr(39)+'$encoded'+chr(39)))"
    $output = (& $Python -u -c $bootstrap | Out-String).Trim()
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Fail "Predecessor freeze source-binding verification failed with exit code $exitCode."
    }
    $script:PredecessorInferenceSha256 = $output.Split([Environment]::NewLine)[-1].Trim()
    if ($script:PredecessorInferenceSha256.Length -ne 64) {
        Fail "Predecessor inference binding digest was not returned."
    }
}

function Assert-ExistingCell([string]$DatasetId, [string]$Method, [int]$Seed, [int]$OuterFold, [object]$Binding, [object]$FreezeRecord) {
    $paths = Get-CellPaths $DatasetId $Method $Seed $OuterFold
    $csvExists = Test-Path -LiteralPath $paths.Csv -PathType Leaf
    $sidecarExists = Test-Path -LiteralPath $paths.Sidecar -PathType Leaf
    if (-not $csvExists -and -not $sidecarExists) {
        return $false
    }
    if (-not $csvExists -or -not $sidecarExists) {
        Fail "Incomplete existing prediction cell; refusing to delete or overwrite: $DatasetId/$Method/seed_$Seed/outer_$OuterFold"
    }

    $sidecar = Read-Json $paths.Sidecar
    Assert-Equal $sidecar.schema_version "external-cross-corpus-preseal-inference-v1" "Invalid sidecar schema"
    Assert-Equal $sidecar.status "COMPLETE_LABEL_BLIND_PRESEAL_INFERENCE" "Existing sidecar is not complete"
    Assert-Equal $sidecar.dataset_id $DatasetId "Existing sidecar dataset mismatch"
    Assert-Equal $sidecar.method $Method "Existing sidecar method mismatch"
    Assert-Equal ([int]$sidecar.seed) $Seed "Existing sidecar seed mismatch"
    Assert-Equal ([int]$sidecar.outer_fold) $OuterFold "Existing sidecar fold mismatch"
    Assert-Equal ([int]$sidecar.rows) ([int]$Binding.rows) "Existing prediction row-count mismatch"
    Assert-ArrayEqual @($sidecar.input_projection_exact) $ExpectedInputColumns "Existing input projection mismatch"
    Assert-ArrayEqual @($sidecar.output_columns_exact) $ExpectedOutputColumns "Existing output-column contract mismatch"

    Assert-Equal ([bool]$sidecar.external_labels_read) $false "Existing sidecar reports external label access"
    Assert-Equal ([bool]$sidecar.external_subgroups_read) $false "Existing sidecar reports subgroup access"
    Assert-Equal ([bool]$sidecar.performance_metrics_computed) $false "Existing sidecar reports metric computation"
    Assert-Equal ([bool]$sidecar.external_adaptation) $false "Existing sidecar reports external adaptation"
    Assert-Equal ([bool]$sidecar.calibration_applied) $false "Existing sidecar reports calibration"

    Assert-Equal $sidecar.external_source.path $Binding.path "Existing external source path mismatch"
    Assert-Equal $sidecar.external_source.sha256 $Binding.sha256 "Existing external source identity mismatch"
    Assert-Equal $sidecar.preexecution_freeze.path $FreezeRecord.path "Existing freeze path mismatch"
    $recordedFreezeSha = [string]$sidecar.preexecution_freeze.sha256
    if ($recordedFreezeSha -eq $FreezeRecord.sha256) {
        Assert-Equal $sidecar.inference_implementation.path (Get-RelativePath $InferencePath) "Successor implementation path mismatch"
        Assert-Equal $sidecar.inference_implementation.sha256 (Get-Sha256 $InferencePath) "Successor implementation identity mismatch"
    } elseif ($recordedFreezeSha -eq $ExpectedPredecessorFreezeSha256) {
        Assert-Equal $sidecar.inference_implementation.path (Get-RelativePath $InferencePath) "Predecessor implementation path mismatch"
        Assert-Equal $sidecar.inference_implementation.sha256 $script:PredecessorInferenceSha256 "Predecessor implementation identity mismatch"
    } else {
        Fail "Unknown existing prediction freeze identity: $recordedFreezeSha"
    }
    Assert-Equal $sidecar.prediction.path (Get-RelativePath $paths.Csv) "Existing prediction path mismatch"
    Assert-Equal $sidecar.prediction.sha256 (Get-Sha256 $paths.Csv) "Existing prediction hash mismatch"

    $header = (Get-Content -LiteralPath $paths.Csv -TotalCount 1 -Encoding UTF8)
    Assert-Equal $header ($ExpectedOutputColumns -join ",") "Existing prediction header mismatch"
    $rows = @(Import-Csv -LiteralPath $paths.Csv)
    Assert-Equal $rows.Count ([int]$Binding.rows) "Existing prediction CSV row-count mismatch"
    foreach ($row in $rows) {
        if ($row.PSObject.Properties.Name -contains "label" -or $row.PSObject.Properties.Name -contains "binary_label") {
            Fail "Existing prediction contains a label column: $($paths.Csv)"
        }
        Assert-Equal $row.method $Method "Existing prediction method mismatch"
        Assert-Equal ([int]$row.seed) $Seed "Existing prediction seed mismatch"
        Assert-Equal ([int]$row.outer_fold) $OuterFold "Existing prediction fold mismatch"
    }
    return $true
}

function Invoke-Preflight([object]$Config) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        Fail "Python environment is missing: $Python"
    }
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        Fail "Active configuration is missing: $Config"
    }
    if (-not (Test-Path -LiteralPath $Freeze -PathType Leaf)) {
        Fail "Production freeze is missing: $Freeze"
    }

    Write-Host "Verifying authorized predecessor freeze cohort..."
    Verify-PredecessorFreeze

    Write-Host "Verifying production freeze..."
    & $Python -u -m $FreezeModule --contract $Config --freeze $Freeze --verify
    $freezeExitCode = $LASTEXITCODE
    if ($freezeExitCode -ne 0) {
        Fail "Production freeze verification failed with exit code $freezeExitCode."
    }

    Write-Host "Verifying label-blind inference contract..."
    & $Python -u -m $InferenceModule --config $Config --validate-only
    $validationExitCode = $LASTEXITCODE
    if ($validationExitCode -ne 0) {
        Fail "Inference validation failed with exit code $validationExitCode."
    }

    Write-Host "Verifying 30 CE/SupCon checkpoint payloads..."
    $checkpointCode = @"
from turkish_ai_detection.experiments.external_cross_corpus.inference import (
    METHODS, OUTER_FOLDS, SEEDS, verify_ce_supcon_payload,
)
for method in METHODS[:2]:
    for seed in SEEDS:
        for outer_fold in OUTER_FOLDS:
            verify_ce_supcon_payload(method, seed, outer_fold)
"@
    & $Python -u -c $checkpointCode
    $checkpointExitCode = $LASTEXITCODE
    if ($checkpointExitCode -ne 0) {
        Fail "CE/SupCon checkpoint binding verification failed with exit code $checkpointExitCode."
    }
    Write-Host "PASS_CE_SUPCON_CHECKPOINT_BINDINGS=30"

    Write-Host "Verifying 15 LIT2026 checkpoint identities..."
    & $Python -u -m turkish_ai_detection.experiments.external_cross_corpus.verify_lit2026_checkpoints
    $litExitCode = $LASTEXITCODE
    if ($litExitCode -ne 0) {
        Fail "LIT2026 checkpoint identity verification failed with exit code $litExitCode."
    }
}

if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    Fail "Active configuration is missing: $Config"
}

$env:PYTHONPATH = Join-Path $Root "src"
$configObject = Read-Json $Config
Assert-Equal $configObject.status "FROZEN_READY_TO_RUN" "Active configuration is not frozen"
Assert-Equal ([bool]$configObject.execution_permitted) $true "External execution is not permitted by the active configuration"
Assert-Equal ([bool]$configObject.evaluation.prediction_first_label_later) $true "Prediction-first contract is disabled"
Assert-Equal ([bool]$configObject.evaluation.external_adaptation) $false "External adaptation is enabled"
Assert-ArrayEqual @($configObject.evaluation.prediction_input_columns_exact) $ExpectedInputColumns "Frozen input projection mismatch"

$datasetBindings = @($configObject.external_data_bindings.primary)
$datasetIds = @($datasetBindings | ForEach-Object { [string]$_.dataset_id })
$methods = @($configObject.model_grid.methods.PSObject.Properties.Name)
$seeds = @($configObject.model_grid.seeds | ForEach-Object { [int]$_ })
$outerFolds = @($configObject.model_grid.outer_folds | ForEach-Object { [int]$_ })

Assert-ArrayEqual $datasetIds @("renklier2026", "tum2024_tr", "bostancioglu2026_tr") "Frozen dataset grid mismatch"
Assert-ArrayEqual $methods $ExpectedMethods "Frozen method grid mismatch"
Assert-ArrayEqual $seeds @(42, 314, 2026) "Frozen seed grid mismatch"
Assert-ArrayEqual $outerFolds @(0, 1, 2, 3, 4) "Frozen outer-fold grid mismatch"

$gridCount = $datasetIds.Count * $methods.Count * $seeds.Count * $outerFolds.Count
Assert-Equal $gridCount $ExpectedTotal "Unexpected inference grid size"

Invoke-Preflight $Config
$freezeRecord = @{
    path = Get-RelativePath $Freeze
    sha256 = Get-Sha256 $Freeze
}

$completed = 0
$index = 0
foreach ($datasetId in $datasetIds) {
    $binding = Get-DatasetBinding $configObject $datasetId
    foreach ($method in $methods) {
        foreach ($seed in $seeds) {
            foreach ($outerFold in $outerFolds) {
                $index++
                $paths = Get-CellPaths $datasetId $method $seed $outerFold
                $existing = Assert-ExistingCell $datasetId $method $seed $outerFold $binding $freezeRecord
                if ($existing) {
                    $completed++
                    Write-Host "[$index/$ExpectedTotal] PASS $datasetId $method seed=$seed outer-fold=$outerFold (existing verified)"
                    Write-Host "Completed: $completed/$ExpectedTotal; Remaining: $($ExpectedTotal - $completed)"
                    continue
                }

                Write-Host "[$index/$ExpectedTotal] START dataset=$datasetId method=$method seed=$seed outer-fold=$outerFold completed=$completed remaining=$($ExpectedTotal - $completed)"
                if (-not $Execute) {
                    Write-Host "[$index/$ExpectedTotal] DRY-RUN dataset=$datasetId method=$method seed=$seed outer-fold=$outerFold"
                    continue
                }

                & $Python -u -m $InferenceModule `
                    --config $Config `
                    --dataset-id $datasetId `
                    --method $method `
                    --seed $seed `
                    --outer-fold $outerFold
                $cellExitCode = $LASTEXITCODE
                if ($cellExitCode -ne 0) {
                    Fail "Inference failed for $datasetId/$method/seed_$seed/outer_$outerFold with exit code $cellExitCode."
                }

                $verified = Assert-ExistingCell $datasetId $method $seed $outerFold $binding $freezeRecord
                if (-not $verified) {
                    Fail "Inference returned success but did not produce a complete verified cell: $datasetId/$method/seed_$seed/outer_$outerFold"
                }
                $completed++
                Write-Host "[$index/$ExpectedTotal] PASS dataset=$datasetId method=$method seed=$seed outer-fold=$outerFold"
                Write-Host "Completed: $completed/$ExpectedTotal; Remaining: $($ExpectedTotal - $completed)"
            }
        }
    }
}

if ($Execute) {
    Assert-Equal $completed $ExpectedTotal "Execution finished without all prediction cells verified"
    Write-Host "STAGE 4 INFERENCE COMPLETE: 135/135 PREDICTION CELLS VERIFIED"
} else {
    Write-Host "DRY RUN COMPLETE: $completed/$ExpectedTotal existing prediction cells verified; no inference was executed."
}
