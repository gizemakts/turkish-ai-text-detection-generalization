param(
    [switch]$Execute,
    [switch]$IncludeM4
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = (Join-Path $ProjectRoot 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'

if (-not $Execute) {
    Write-Host 'DRY RUN: nothing was executed.'
    Write-Host 'Author-disjoint analysis: author-cluster uncertainty (10 authors).'
    Write-Host 'Run: powershell -ExecutionPolicy Bypass -File .\run_author_disjoint_newspaper.ps1 -Execute'
    Write-Host 'Optional M4 diagnostic: add -IncludeM4'
    exit 0
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing project Python: $Python" }
Set-Location -LiteralPath $ProjectRoot

Write-Host '[0/7] CUDA availability'
& $Python -c "import torch,sys; ok=torch.cuda.is_available(); print({'cuda_available':ok,'gpu':torch.cuda.get_device_name(0) if ok else None}); sys.exit(0 if ok else 2)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA availability failed' }

Write-Host '[1/7] Protocol and input checks'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.validate
if ($LASTEXITCODE -ne 0) { throw 'Design validation failed' }
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.summarize_author_cluster_statistics
if ($LASTEXITCODE -ne 0) { throw 'Statistical configuration validation failed' }

Write-Host '[2/7] CE fits (15)'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.run_matrix --execute --objective ce
if ($LASTEXITCODE -ne 0) { throw 'CE matrix failed' }

Write-Host '[3/7] SupCon fits (15)'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.run_matrix --execute --objective supcon
if ($LASTEXITCODE -ne 0) { throw 'SupCon matrix failed' }

Write-Host '[4/7] Prediction completeness/integrity validation'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.validate --predictions
if ($LASTEXITCODE -ne 0) { throw 'Prediction validation failed' }

Write-Host '[5/7] Author-cluster statistics and pair-level sensitivity'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.summarize_author_cluster_statistics --execute
if ($LASTEXITCODE -ne 0) { throw 'Author-cluster statistics failed' }

Write-Host '[6/7] Final manifest'
& $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.validate --finalize
if ($LASTEXITCODE -ne 0) { throw 'Final manifest failed' }

if ($IncludeM4) {
    Write-Host '[7/7 optional] M4 diagnostic predictions only; no final AUC summary'
    & $Python -m turkish_ai_detection.experiments.author_disjoint_newspaper.run_m4_diagnostic --execute
    if ($LASTEXITCODE -ne 0) { throw 'M4 diagnostic failed' }
}

Write-Host 'COMPLETE'
