# Suite selection for the shared CI workflow.
#
# A suite's key is a hash of its definition, the blobs of the tracked files its inputs match, and
# the blobs of every tracked file no suite claims and no ignore glob matches. A suite whose key
# already has a passed run is not picked again. Passed keys are recorded
# as actions/cache entries named ci-done-<name>.<key>_<name>.<key>...

Set-StrictMode -Version Latest

$SuiteNamePattern = '^[a-z0-9][a-z0-9-]{0,23}$'
$RecordPrefix = 'ci-done-'
$RecordLimit = 512

# A glob anchored at the repository root as a regular expression: `*` and `?` stay inside one
# path segment, `**/` matches zero or more folders and a trailing `**` everything below.
function ConvertTo-GlobRegex([string] $Glob) {
    $out = [Text.StringBuilder]::new('^')
    for ($i = 0; $i -lt $Glob.Length; $i++) {
        $c = $Glob[$i]
        if ($c -eq '*' -and $i + 1 -lt $Glob.Length -and $Glob[$i + 1] -eq '*') {
            if ($i + 2 -lt $Glob.Length -and $Glob[$i + 2] -eq '/') { [void] $out.Append('(?:.*/)?'); $i += 2 }
            else { [void] $out.Append('.*'); $i++ }
        }
        elseif ($c -eq '*') { [void] $out.Append('[^/]*') }
        elseif ($c -eq '?') { [void] $out.Append('[^/]') }
        else { [void] $out.Append([regex]::Escape([string] $c)) }
    }
    [void] $out.Append('$')
    $out.ToString()
}

# ci/suites.yml as a hashtable with `ignore` (globs) and `suites` (each with name, tier, inputs,
# run and local), checked against the manifest rules. Throws on the first broken rule.
function Read-CiManifest([string] $Path) {
    if (-not (Get-Command yq -ErrorAction SilentlyContinue)) { throw 'yq not found on PATH; CI needs mikefarah yq 4' }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "no manifest at $Path" }
    $json = yq -o=json '.' $Path 2>&1
    if ($LASTEXITCODE) { throw "yq could not parse ${Path}: $($json -join ' ')" }
    $raw = ($json -join "`n") | ConvertFrom-Json -AsHashtable
    if ($raw -isnot [hashtable] -or -not $raw.ContainsKey('suites') -or -not $raw.suites) { throw "$Path has no suites" }

    $ignore = @(if ($raw.ContainsKey('ignore') -and $raw.ignore) { $raw.ignore | ForEach-Object { [string] $_ } })
    $names = [Collections.Generic.HashSet[string]]::new()
    $suites = foreach ($entry in $raw.suites) {
        if ($entry -isnot [hashtable]) { throw "$Path has a suite that is not a mapping" }
        $name = [string] $entry['name']
        if ($name -notmatch $SuiteNamePattern) { throw "suite name '$name' must match $SuiteNamePattern" }
        if (-not $names.Add($name)) { throw "suite '$name' is listed twice" }
        $tier = [string] $entry['tier']
        if ($tier -notin 'quick', 'full') { throw "suite '$name' has tier '$tier'; use quick or full" }
        $inputs = @($entry['inputs'] | Where-Object { $_ } | ForEach-Object { [string] $_ })
        if (-not $inputs) { throw "suite '$name' has no inputs" }
        $run = [string] $entry['run']
        if (-not $run.Trim()) { throw "suite '$name' has no run command" }
        [ordered] @{ name = $name; tier = $tier; inputs = $inputs; run = $run; local = [bool] $entry['local'] }
    }
    @{ ignore = $ignore; suites = @($suites) }
}

# Every tracked file under $Root as @{ Path; Blob }, in git's order.
function Get-TrackedFile([string] $Root) {
    $raw = git -C $Root ls-files -s -z
    if ($LASTEXITCODE) { throw "git ls-files failed in $Root" }
    foreach ($record in (($raw -join "`n") -split "`0")) {
        if (-not $record) { continue }
        $meta, $path = $record -split "`t", 2
        @{ Path = $path; Blob = ($meta -split ' ')[1] }
    }
}

function Get-Sha256Hex([string] $Text) {
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($Text))).ToLowerInvariant()
}

# Suite name -> 12-hex key. ci/suites.yml itself is outside every key: each suite's own
# definition is part of its key instead.
function Get-SuiteKey([hashtable] $Manifest, [object[]] $Files, [string] $ManifestPath = 'ci/suites.yml') {
    $matchers = [ordered] @{}
    foreach ($suite in $Manifest.suites) {
        $matchers[$suite.name] = @($suite.inputs | ForEach-Object { [regex]::new((ConvertTo-GlobRegex $_)) })
    }
    $ignore = @($Manifest.ignore | ForEach-Object { [regex]::new((ConvertTo-GlobRegex $_)) })

    $claimed = @{}
    foreach ($name in $matchers.Keys) { $claimed[$name] = [Collections.Generic.List[string]]::new() }
    $global = [Collections.Generic.List[string]]::new()
    foreach ($file in $Files) {
        $line = "$($file.Blob) $($file.Path)"
        $owned = $false
        foreach ($name in $matchers.Keys) {
            foreach ($rx in $matchers[$name]) {
                if ($rx.IsMatch($file.Path)) { $claimed[$name].Add($line); $owned = $true; break }
            }
        }
        if ($owned -or $file.Path -eq $ManifestPath) { continue }
        $ignored = $false
        foreach ($rx in $ignore) { if ($rx.IsMatch($file.Path)) { $ignored = $true; break } }
        if (-not $ignored) { $global.Add($line) }
    }

    $globalText = ($global | Sort-Object -CaseSensitive) -join "`n"
    $keys = [ordered] @{}
    foreach ($suite in $Manifest.suites) {
        $definition = $suite | ConvertTo-Json -Compress
        $own = ($claimed[$suite.name] | Sort-Object -CaseSensitive) -join "`n"
        $keys[$suite.name] = (Get-Sha256Hex "suite`n$definition`nfiles`n$own`nglobal`n$globalText").Substring(0, 12)
    }
    $keys
}

# The passed `name.key` pairs in a list of cache keys.
function Get-DoneKey([string[]] $CacheKeys) {
    $done = [Collections.Generic.HashSet[string]]::new()
    foreach ($cacheKey in $CacheKeys) {
        if (-not $cacheKey.StartsWith($RecordPrefix)) { continue }
        foreach ($pair in $cacheKey.Substring($RecordPrefix.Length) -split '_') { if ($pair) { [void] $done.Add($pair) } }
    }
    , $done
}

# The suite names, in manifest order, a CI run of $Tier picks: quick picks quick suites, full picks
# every suite; local-only suites never; a suite whose key is done only when $Force.
function Select-Suite([hashtable] $Manifest, [Collections.Specialized.OrderedDictionary] $Keys,
    [Collections.Generic.HashSet[string]] $Done, [string] $Tier, [switch] $Force) {
    foreach ($suite in $Manifest.suites) {
        if ($suite.local) { continue }
        if ($Tier -eq 'quick' -and $suite.tier -ne 'quick') { continue }
        if (-not $Force -and $Done.Contains("$($suite.name).$($Keys[$suite.name])")) { continue }
        $suite.name
    }
}

# The cache key that records $Names as passed at their keys.
function Format-Record([Collections.Specialized.OrderedDictionary] $Keys, [string[]] $Names) {
    if (-not $Names) { return '' }
    $record = $RecordPrefix + (($Names | ForEach-Object { "$_.$($Keys[$_])" }) -join '_')
    if ($record.Length -gt $RecordLimit) { throw "record key is $($record.Length) characters, over $RecordLimit; shorten suite names" }
    $record
}

Export-ModuleMember -Function ConvertTo-GlobRegex, Read-CiManifest, Get-TrackedFile, Get-SuiteKey, Get-DoneKey,
    Select-Suite, Format-Record
