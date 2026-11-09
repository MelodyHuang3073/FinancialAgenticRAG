# Starts the four shards of ONE ablation row as hidden background processes (two per question set).
#   powershell -File launch.ps1 -Row 5 -RunDir C:\...\financial_agentic_rag_runs\ablation_row5 -Python C:\...\python.exe
# The row is passed here and nowhere else: this script sets ABLATE_ROW for the child processes and writes
# <RunDir>\config.json, so a run cannot be started with a row typed by hand into a shell.
# Each shard resumes from its own output file, so running this again continues where it stopped.
# Shard i of a set writes <RunDir>\<set>_<i>.json, .log and, when it exits, .done.
param(
    [Parameter(Mandatory = $true)][ValidateSet("1", "2a", "2b", "3", "4", "5")][string]$Row,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python,
    [int]$Limit = 0          # smoke test: only the first N questions of each shard's list
)
$backend = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
New-Item -ItemType Directory -Force $RunDir | Out-Null
$commit = (git -C $backend rev-parse HEAD).Trim()
$dirty = (git -C $backend status --porcelain -- app tests/run_single_file_subset.py)
if ($dirty) { Write-Error "app/ or the runner has uncommitted changes; commit first (code is frozen during a run)"; exit 1 }
@{ ablate_row = $Row; code_commit = $commit; started = (Get-Date).ToString("s"); limit = $Limit } |
    ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $RunDir "config.json")
foreach ($set in @("train", "out")) {
    foreach ($i in @(0, 1)) {
        $tag = "${set}_${i}"
        $lim = if ($Limit -gt 0) { "--limit $Limit" } else { "" }
        $cmd = "Set-Location '$backend'; `$env:ABLATE_ROW='$Row'; `$env:PYTHONIOENCODING='utf-8'; Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue; & '$Python' tests/run_single_file_subset.py --set $set --candidates experiment_results/all_candidates_$set.json --shard $i/2 $lim --out '$RunDir\$tag.json' 2> '$RunDir\$tag.log' | Out-File -Encoding utf8 '$RunDir\$tag.out'; ('done ' + `$LASTEXITCODE) | Out-File '$RunDir\$tag.done'"
        Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $cmd)
    }
}
Write-Output "started 4 shards for ablation row $Row in $RunDir (code $commit)"
