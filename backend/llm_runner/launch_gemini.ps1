# Starts ONE Gemini model-swap run as a detached driver: first the 150 training questions (2 shards), then the 150
# independent-test questions (2 shards). Each phase's start and end time goes to <RunDir>\phases.json, so usage and
# time can be recorded per question set (the usage ledger has no question id; the time window is the key).
#   powershell -File launch_gemini.ps1 -Model gemini-3.1-flash-lite -RunDir C:\...\financial_agentic_rag_runs\llm_gemini-3.1-flash-lite -Python C:\...\python.exe
# The model reaches the system only through the environment of the child processes: GOOGLE_GENAI_MODEL is set to
# -Model, and run_gemini.py blanks OPENAI_API_KEY inside the process (PowerShell cannot: assigning an empty string
# there deletes the variable, backend\.env is read again and the run silently uses OpenAI). The Google key comes from
# backend\.env (GOOGLE_API_KEY). The full system is unchanged; no ablation switch is set.
# Free-tier quota is about 500 calls per model per day. When it runs out the runner records fallback-shaped answers;
# run drop_failed.py --apply and start this script again with -Resume the next day: records already in the shard
# files are kept and only the questions without a record run. Without -Resume every shard starts from nothing and
# OVERWRITES its output file.
# Shard i of a set writes <RunDir>\<set>_<i>.json, .log, .out and, when it exits, .done.
param(
    [Parameter(Mandatory = $true)][ValidateSet("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")][string]$Model,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python,
    [int]$Limit = 0,         # smoke test: only the first N questions of each shard's list
    [switch]$Resume,
    [switch]$DryRun          # only write driver.ps1 (to check it), start nothing
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
$lim = if ($Limit -gt 0) { "--limit $Limit" } else { "" }
$driver = @"
`$ErrorActionPreference = 'Continue'
Set-Location '$backend'
`$env:GOOGLE_GENAI_MODEL = '$Model'
`$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:ABLATE_ROW -ErrorAction SilentlyContinue
Remove-Item Env:ABLATE_POT -ErrorAction SilentlyContinue
`$phasesPath = '$RunDir\phases.json'
`$phases = @{}
if (Test-Path `$phasesPath) { `$o = Get-Content `$phasesPath -Raw | ConvertFrom-Json; foreach (`$p in `$o.PSObject.Properties) { `$phases[`$p.Name] = `$p.Value } }
foreach (`$set in @('train', 'out')) {
    `$start = (Get-Date).ToString('s')
    `$procs = @()
    foreach (`$i in @(0, 1)) {
        `$tag = "`${set}_`${i}"
        Remove-Item "$RunDir\`$tag.done" -ErrorAction SilentlyContinue
        `$args2 = @('llm_runner/run_gemini.py', '--set', `$set, '--candidates', "experiment_results/all_candidates_`$set.json", '--shard', "`$i/2", '--out', "$RunDir\`$tag.json")
        if ('$Limit' -ne '0') { `$args2 += @('--limit', '$Limit') }
        if ('$resumeArg' -ne '') { `$args2 += '--resume' }
        `$procs += Start-Process -FilePath '$Python' -ArgumentList `$args2 -NoNewWindow -PassThru -RedirectStandardOutput "$RunDir\`$tag.out" -RedirectStandardError "$RunDir\`$tag.log"
    }
    `$procs | Wait-Process
    `$k = 0
    foreach (`$i in @(0, 1)) { ('done ' + `$procs[`$k].ExitCode) | Out-File "$RunDir\`${set}_`${i}.done"; `$k++ }
    `$phases[`$set] = @{ start = `$start; end = (Get-Date).ToString('s') }
    `$phases | ConvertTo-Json | Out-File -Encoding utf8 `$phasesPath
}
"@
$drv = Join-Path $RunDir "driver.ps1"
$driver | Out-File -Encoding utf8 $drv
if ($DryRun) { Write-Output "dry run: wrote $drv"; exit 0 }
Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $drv)
Write-Output "started the driver for $Model in $RunDir (code $commit)$(if ($Resume) { ', resume' })"
