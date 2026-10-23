[CmdletBinding()]
param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv-gpu\Scripts\python.exe"
$Runner = Join-Path $Root "src\turkish_ai_detection\experiments\external_cross_corpus\rerun_fit.py"
$Contract = Join-Path $Root "configs\external_cross_corpus.json"
$Freeze = Join-Path $Root "provenance\experiments\external_cross_corpus_v2\PREEXECUTION_FREEZE.json"
$OutputRoot = Join-Path $Root "artifacts\checkpoints\external_cross_corpus\protocol_faithful_rerun"
$ExpectedFreezeSha256 = "b7d5ede34f271ada69524b6315a9d4ef93609d7359666d607202912a683a1aec"
$Seeds = @(42, 314, 2026)
$OuterFolds = @(0, 1, 2, 3, 4)

function Fail([string]$Message) {
    throw $Message
}

function Get-CellPath([string]$Objective, [int]$Seed, [int]$OuterFold) {
    return Join-Path $OutputRoot ("{0}_seed_{1}_outer_{2}" -f $Objective, $Seed, $OuterFold)
}

function Read-Json([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Assert-File([string]$Path, [bool]$NonEmpty = $false) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "Missing required file: $Path"
    }
    if ($NonEmpty -and ((Get-Item -LiteralPath $Path).Length -le 0)) {
        Fail "Required file is empty: $Path"
    }
}

function Assert-StringEqual([object]$Actual, [string]$Expected, [string]$Name) {
    if ([string]$Actual -ne $Expected) {
        Fail "$Name mismatch: expected '$Expected', observed '$Actual'"
    }
}

function Assert-IntEqual([object]$Actual, [int]$Expected, [string]$Name) {
    if ([int]$Actual -ne $Expected) {
        Fail "$Name mismatch: expected '$Expected', observed '$Actual'"
    }
}

function Test-VerifiedCell([string]$Objective, [int]$Seed, [int]$OuterFold) {
    $Cell = Get-CellPath $Objective $Seed $OuterFold
    Assert-File (Join-Path $Cell "checkpoint\model.safetensors") $true
    Assert-File (Join-Path $Cell "checkpoint\config.json")
    Assert-File (Join-Path $Cell "run.json")
    Assert-File (Join-Path $Cell "internal_predictions.csv") $true

    $Run = Read-Json (Join-Path $Cell "run.json")
    Assert-StringEqual $Run.status "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN" "status"
    Assert-StringEqual $Run.objective $Objective "objective"
    Assert-IntEqual $Run.seed $Seed "seed"
    Assert-IntEqual $Run.outer_fold $OuterFold "outer_fold"

    if ($null -eq $Run.checkpoint_roundtrip) {
        Fail "Missing checkpoint_roundtrip in $Cell\run.json"
    }
    Assert-StringEqual $Run.checkpoint_roundtrip.status "PASS_CHECKPOINT_ROUNDTRIP" "checkpoint_roundtrip.status"

    if ($null -eq $Run.payload) {
        Fail "Missing payload in $Cell\run.json"
    }
    $Model = Join-Path $Root ([string]$Run.payload.model)
    $Config = Join-Path $Root ([string]$Run.payload.config)
    Assert-File $Model $true
    Assert-File $Config

    $ObservedModelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Model).Hash.ToLowerInvariant()
    $ObservedConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant()
    Assert-StringEqual $ObservedModelHash ([string]$Run.payload.model_sha256).ToLowerInvariant() "payload.model_sha256"
    Assert-StringEqual $ObservedConfigHash ([string]$Run.payload.config_sha256).ToLowerInvariant() "payload.config_sha256"
    return $true
}

Assert-File $Python
Assert-File $Runner
Assert-File $Contract
Assert-File $Freeze

$ConfigObject = Read-Json $Contract
Assert-StringEqual $ConfigObject.status "FROZEN_READY_TO_RUN" "contract.status"
if ($ConfigObject.execution_permitted -ne $true) {
    Fail "contract.execution_permitted must be true"
}

$FreezeObject = Read-Json $Freeze
Assert-StringEqual ([string]$FreezeObject.freeze_sha256).ToLowerInvariant() $ExpectedFreezeSha256 "freeze_sha256"

$env:PYTHONPATH = Join-Path $Root "src"
& $Python -m turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 `
    --contract $Contract --freeze $Freeze --verify
if ($LASTEXITCODE -ne 0) {
    Fail "Production freeze verification failed with exit code $LASTEXITCODE"
}

$Cells = @()
foreach ($Objective in @("ce", "supcon")) {
    foreach ($Seed in $Seeds) {
        foreach ($OuterFold in $OuterFolds) {
            $Cells += [pscustomobject]@{
                Objective = $Objective
                Seed = $Seed
                OuterFold = $OuterFold
                Path = Get-CellPath $Objective $Seed $OuterFold
            }
        }
    }
}

$Completed = 0
$ToRun = @()
for ($CellIndex = 0; $CellIndex -lt $Cells.Count; $CellIndex++) {
    $Cell = $Cells[$CellIndex]
    $Cell | Add-Member -NotePropertyName Index -NotePropertyValue ($CellIndex + 1) -Force
    if (Test-Path -LiteralPath $Cell.Path) {
        Write-Host ("[{0}/30] VALIDATING EXISTING {1} seed={2} outer={3}" -f $Cell.Index, $Cell.Objective, $Cell.Seed, $Cell.OuterFold)
        try {
            Test-VerifiedCell $Cell.Objective $Cell.Seed $Cell.OuterFold | Out-Null
        } catch {
            Write-Host ("[{0}/30] FAIL existing checkpoint {1}; completed={2}; remaining={3}" -f `
                $Cell.Index, $Cell.Path, $Completed, ($Cells.Count - $Completed))
            throw
        }
        $Cell | Add-Member -NotePropertyName State -NotePropertyValue "SKIP_VERIFIED" -Force
        $Completed++
    } else {
        $Cell | Add-Member -NotePropertyName State -NotePropertyValue "RUN" -Force
        $ToRun += $Cell
    }
}

Write-Host ""
Write-Host ("Production freeze: PASS (freeze_sha256=$ExpectedFreezeSha256)")
Write-Host ("Grid: {0} cells; verified existing: {1}; remaining: {2}" -f $Cells.Count, $Completed, $ToRun.Count)
$Cells | Format-Table Objective, Seed, OuterFold, State -AutoSize | Out-Host

if (-not $Execute) {
    Write-Host "DRY RUN: no training launched. Re-run with -Execute to start sequential GPU training."
    exit 0
}

foreach ($Cell in $ToRun) {
    if (Test-Path -LiteralPath $Cell.Path) {
        Fail "Refusing to overwrite a cell created after planning: $($Cell.Path)"
    }

    $Started = Get-Date
    $Name = "{0}_seed_{1}_outer_{2}" -f $Cell.Objective, $Cell.Seed, $Cell.OuterFold
    Write-Host ("[{0}/30] START {1}; objective={2}; seed={3}; outer_fold={4}; completed={5}; remaining={6}" -f `
        $Cell.Index, $Name, $Cell.Objective, $Cell.Seed, $Cell.OuterFold, $Completed, ($Cells.Count - $Completed))
    & $Python $Runner --objective $Cell.Objective --seed $Cell.Seed --outer-fold $Cell.OuterFold
    $ExitCode = $LASTEXITCODE
    $Duration = (Get-Date) - $Started

    if ($ExitCode -ne 0) {
        Write-Host ("[{0}/30] FAIL {1}; exit={2}; duration={3}; completed={4}; remaining={5}" -f `
            $Cell.Index, $Name, $ExitCode, $Duration, $Completed, ($Cells.Count - $Completed))
        Fail "Training stopped after failure in $Name"
    }
    Test-VerifiedCell $Cell.Objective $Cell.Seed $Cell.OuterFold | Out-Null
    $Completed++
    Write-Host ("[{0}/30] PASS {1}; duration={2}; Completed: {3}/30; Remaining: {4}/30" -f `
        $Cell.Index, $Name, $Duration, $Completed, ($Cells.Count - $Completed))
}

Write-Host ""
Write-Host "STAGE 3 COMPLETE: 30/30 CHECKPOINTS VERIFIED"
$Cells | ForEach-Object {
    Test-VerifiedCell $_.Objective $_.Seed $_.OuterFold | Out-Null
}
