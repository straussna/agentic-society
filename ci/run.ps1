#Requires -Version 7.4
# Runs this repository's CI suites with the shared runner, passing every argument through:
# ci/run.ps1 -Tier quick|full | -Suite a,b [-Clean] [-Plain] [-NoNotify] [-LogFile path].
# The runner is the repository's own .github/actions/ci-gate/run.ps1 when it has one, else the one
# in claude-config (~/.claude).
$ErrorActionPreference = 'Stop'
$runner = @(
    (Join-Path $PSScriptRoot '../.github/actions/ci-gate/run.ps1')
    (Join-Path $HOME '.claude/.github/actions/ci-gate/run.ps1')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $runner) {
    [Console]::Error.WriteLine("usage: no shared CI runner in this repository or ~/.claude; clone claude-config to ~/.claude")
    exit 2
}
try { & $runner -RepoRoot (Split-Path -Parent $PSScriptRoot) @args }
catch [Management.Automation.ParameterBindingException] { [Console]::Error.WriteLine("usage: $($_.Exception.Message)"); exit 2 }
catch { [Console]::Error.WriteLine("failed: $($_.Exception.Message)"); exit 1 }
exit $LASTEXITCODE
