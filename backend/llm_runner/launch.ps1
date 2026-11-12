# Starts the four shards of ONE model-swap run as hidden background processes (two per question set).
#   powershell -File launch.ps1 -Model gpt-5-nano -RunDir C:\...\financial_agentic_rag_runs\llm_gpt-5-nano -Python C:\...\python.exe
# The model is passed here and nowhere else: this script sets OPENAI_MODEL for the child processes and writes
# <RunDir>\config.json, so a run cannot be started with a model typed by hand into a shell. The full system
# is unchanged (no ablation switch is set); only the model of every LLM call (answer and decomposer) differs.
# Running this again WITHOUT -Resume starts every shard from nothing and OVERWRITES its output file (the runner
# does not resume by itself). To re-run only the failed questions: run drop_failed.py --apply first (it removes the
# failed records from the shard files), then run this again with -Resume: the records still in the shard files are
# kept and only the questions without a record run. -Resume keeps the original start time in config.json and
# adds the time of each resume to its "resumed" list.
# Shard i of a set writes <RunDir>\<set>_<i>.json, .log and, when it exits, .done.
param(
    [Parameter(Mandatory = $true)][ValidateSet("gpt-5-nano", "gpt-4o-mini")][string]$Model,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python,
    [int]$Limit = 0,         # smoke test: only the first N questions of each shard's list
    [switch]$Resume
)
$backend = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
New-Item -ItemType Directory -Force $RunDir | Out-Null
$commit = (git -C $backend rev-parse HEAD).Trim()
$dirty = (git -C $backend status --porcelain -- app tests/run_single_file_subset.py)
if ($dirty) { Write-Error "app/ or the runner has uncommitted changes; commit first (code is frozen during a run)"; exit 1 }
$cfgPath = Join-Path $RunDir "config.json"
if ($Resume) {
    if (-not (Test-Path $cfgPath)) { Write-Error "-Resume needs an earlier run in $RunDir"; exit 1 }
    $old = Get-Content $cfgPath -Raw | ConvertFrom-Json
    if ($old.model -ne $Model) { Write-Error "the earlier run in $RunDir used model $($old.model), not $Model"; exit 1 }
    $resumed = @($old.resumed) + @((Get-Date).ToString("s")) | Where-Object { $_ }
    @{ model = $Model; code_commit = $old.code_commit; started = $old.started; limit = $old.limit; resumed = @($resumed); resume_code_commit = $commit } |
        ConvertTo-Json | Out-File -Encoding utf8 $cfgPath
} else {
    @{ model = $Model; code_commit = $commit; started = (Get-Date).ToString("s"); limit = $Limit } |
        ConvertTo-Json | Out-File -Encoding utf8 $cfgPath
}
$resumeArg = if ($Resume) { "--resume" } else { "" }
foreach ($set in @("train", "out")) {
    foreach ($i in @(0, 1)) {
        $tag = "${set}_${i}"
        $lim = if ($Limit -gt 0) { "--limit $Limit" } else { "" }
        $cmd = "Set-Location '$backend'; `$env:OPENAI_MODEL='$Model'; `$env:PYTHONIOENCODING='utf-8'; Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue; Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue; Remove-Item Env:ABLATE_ROW -ErrorAction SilentlyContinue; Remove-Item Env:ABLATE_POT -ErrorAction SilentlyContinue; & '$Python' tests/run_single_file_subset.py --set $set --candidates experiment_results/all_candidates_$set.json --shard $i/2 $lim $resumeArg --out '$RunDir\$tag.json' 2> '$RunDir\$tag.log' | Out-File -Encoding utf8 '$RunDir\$tag.out'; ('done ' + `$LASTEXITCODE) | Out-File '$RunDir\$tag.done'"
        Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $cmd)
    }
}
Write-Output "started 4 shards for $Model in $RunDir (code $commit)$(if ($Resume) { ', resume' })"
