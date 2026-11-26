# Runs run_questions.py inside the container. Only the folders listed here are visible to it (not the project root,
# so backend/.env is not in the container); the API key is passed as an environment variable and never printed.
#   .\run_docker.ps1 -Source <ReAcTable> -Data <inputs> -Out <output folder> -RunArgs "--limit 3"
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$RunArgs = "",
    [string]$Name = "reactable-run",
    [switch]$Detach        # run in the background (survives closing the editor); progress: docker logs <Name>
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Split-Path -Parent (Split-Path -Parent $here)
# one ledger folder per container: every container's process id is 1, and the ledger is one file per process id
$ledger = Join-Path $Out "ledger"
New-Item -ItemType Directory -Force $Out, $ledger | Out-Null
$line = Get-Content (Join-Path $backend ".env") | Where-Object { $_ -match '^\s*OPENAI_API_KEY\s*=' } | Select-Object -First 1
if (-not $line) { throw "OPENAI_API_KEY not found in backend/.env" }
$env:OPENAI_API_KEY = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
$mounts = @(
    "-v", "${here}:/adapter/baselines/reactable:ro",
    "-v", "$(Join-Path $backend 'app\agent\usage_tracker.py'):/adapter/app/agent/usage_tracker.py:ro",
    "-v", "${ledger}:/adapter/.llm_usage",
    "-v", "${Source}:/src:ro",
    "-v", "${Data}:/data:ro",
    "-v", "${Out}:/out"
)
$extra = @()
if ($RunArgs) { $extra = $RunArgs -split '\s+' }
try {
    $mode = @("--rm")
    if ($Detach) { $mode = @("-d") }
    docker run @mode --name $Name --memory 3g -e OPENAI_API_KEY -e REACTABLE_SRC=/src -e PYTHONPATH=/adapter -w /out @mounts reactable-baseline:1 `
        python -m baselines.reactable.run_questions --data-dir /data --out /out/results.json @extra
} finally {
    Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
}
