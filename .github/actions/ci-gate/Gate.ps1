#Requires -Version 7.4
<#
.SYNOPSIS
Picks the suites in ci/suites.yml that a CI run of a tier still has to run.

.DESCRIPTION
Run by the ci-gate action from the repository root. Reads the manifest, keys each suite by its
inputs, lists the passed keys recorded in the repository's Actions cache and writes four
step outputs: `suites` (comma-separated names, empty when nothing is left to run), `record`
(the cache key that records them all as passed), `keys` (every suite's key) and `runner` (the
path of the shared CI runner, run.ps1 beside this script).

.PARAMETER Tier
quick or full.
.PARAMETER Force
true picks suites whose key already has a passed run.

.NOTES
Exit codes:
  0  outputs written
  1  failed: manifest, git or record key error
  2  usage error
A cache listing that fails picks every suite of the tier and logs a warning.
#>
param(
    [string] $Tier,
    [string] $Force = 'false',
    [Parameter(ValueFromRemainingArguments)] [string[]] $Rest
)
$ErrorActionPreference = 'Stop'

function Write-Line([string] $Event, [string] $Message, [string] $Glyph = '·') {
    [Console]::Error.WriteLine(('{0} {1} {2,-11} {3}' -f (Get-Date -Format 'HH:mm:ss'), $Glyph, $Event, $Message))
}
function Exit-Usage([string] $Message) { [Console]::Error.WriteLine("usage: $Message"); exit 2 }

if ($Rest) { Exit-Usage "unknown argument $($Rest[0])" }
if ($Tier -notin 'quick', 'full') { Exit-Usage "-Tier must be quick or full, not '$Tier'" }
if ($Force -notin 'true', 'false') { Exit-Usage "-Force must be true or false, not '$Force'" }

Import-Module (Join-Path $PSScriptRoot 'CiGate.psm1') -Force
$clock = [Diagnostics.Stopwatch]::StartNew()
$code, $reason = 0, ''
Write-Line 'start' "gate tier=$Tier force=$Force"
try {
    $manifest = Read-CiManifest 'ci/suites.yml'
    $keys = Get-SuiteKey $manifest @(Get-TrackedFile '.')

    $cacheKeys = @()
    if ($Force -eq 'false') {
        $cacheKeys = @(gh cache list -R $env:GITHUB_REPOSITORY --key ci-done- --limit 1000 --json key -q '.[].key' 2>&1)
        if ($LASTEXITCODE) {
            Write-Line 'warning' "cache listing failed, picking every suite of the tier: $($cacheKeys -join ' ')" '!'
            $cacheKeys = @()
        }
    }
    $done = Get-DoneKey $cacheKeys
    $picked = @(Select-Suite $manifest $keys $done $Tier -Force:($Force -eq 'true'))
    $record = Format-Record $keys $picked

    foreach ($suite in $manifest.suites) {
        $state = if ($picked -contains $suite.name) { 'run' }
        elseif ($suite.local) { 'local only' }
        elseif ($Tier -eq 'quick' -and $suite.tier -ne 'quick') { 'full tier' }
        else { 'done at this key' }
        Write-Line 'suite' "$($suite.name) $($keys[$suite.name]): $state"
    }
    $outputs = @("suites=$($picked -join ',')", "record=$record", "keys=$(($keys.GetEnumerator() | ForEach-Object { "$($_.Key).$($_.Value)" }) -join ',')",
        "runner=$(Join-Path $PSScriptRoot 'run.ps1')")
    if ($env:GITHUB_OUTPUT) { $outputs | Add-Content -LiteralPath $env:GITHUB_OUTPUT }
    else { $outputs | Write-Output }
    if ($env:GITHUB_STEP_SUMMARY) {
        $line = if ($picked) { "CI $Tier tier runs: $($picked -join ', ')" } else { "CI $Tier tier: every suite is done at its current inputs" }
        Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Value $line
    }
    $reason = "$($picked.Count) of $($manifest.suites.Count) suites to run"
}
catch {
    $code, $reason = 1, $_.Exception.Message
}
$elapsed = '{0}s' -f [int][Math]::Floor($clock.Elapsed.TotalSeconds)
if ($code) { Write-Line 'failed' "gate: $reason; exit=$code; elapsed=$elapsed" 'x' }
else { Write-Line 'done' "gate: $reason; exit=0; elapsed=$elapsed" }
exit $code
