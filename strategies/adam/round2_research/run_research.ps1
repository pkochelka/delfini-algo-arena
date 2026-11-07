# Run from the repository root. Fail immediately if a research stage fails.
$ErrorActionPreference = 'Stop'
$researchPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
foreach ($researchStage in @('extract','models','risk_structure','baselines','interpret','flow_models','diagnostics','report')) {
    & $researchPython -X utf8 -B "strategies\adam\round2_research\$researchStage.py"
    if ($LASTEXITCODE -ne 0) { throw "Research stage failed: $researchStage" }
}
