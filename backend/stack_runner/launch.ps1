# Starts the four shards of one stacking step as hidden background processes (two per question set).
# Each shard resumes from its own output file, so running this again continues where it stopped.
#   powershell -File launch.ps1 -Step 1 -RunDir C:\...\financial_agentic_rag_runs\stack_step1 -Python C:\...\python.exe
# Each shard writes <RunDir>\<set>_<i>.json and .log, and <RunDir>\<set>_<i>.done when it exits.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python
)
$backend = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
New-Item -ItemType Directory -Force $RunDir | Out-Null
foreach ($set in @("original", "outsample")) {
    foreach ($i in @(1, 2)) {
        $tag = "${set}_${i}"
        $cmd = "Set-Location '$backend'; & '$Python' stack_runner/run_step.py --step $Step --set $set --shard $i/2 --out '$RunDir\$tag.json' *> '$RunDir\$tag.log'; ('done ' + `$LASTEXITCODE) | Out-File '$RunDir\$tag.done'"
        Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $cmd)
    }
}
Write-Output "started 4 shards for step $Step in $RunDir"
