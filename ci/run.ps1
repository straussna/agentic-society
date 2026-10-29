#Requires -Version 7.4
<#
.SYNOPSIS
Runs this repository's CI suites from ci/suites.yml.

.DESCRIPTION
Runs every suite of a tier, or the suites -Suite names, one after another from the repository
root. Each suite's run command runs in a child pwsh whose output goes to its own file; a failed
suite's output is shown on stderr (all of it under CI, the last 40 lines otherwise). stdout holds
one result line per suite. CI runs the suites the gate picks with -Suite; a local -Tier run
includes the local-only suites.

The live block shows a moon for the time of day, the elapsed time (green, yellow or red against
the usual run), the suites done with a bar, the time left and a sparkline, and the running suite
in its own colour with a scanner.

.PARAMETER Tier
quick runs the quick suites; full runs every suite. Default quick.
.PARAMETER Suite
Comma-separated suite names to run instead of a tier.
.PARAMETER Clean
Runs in a throwaway worktree of HEAD, so only committed files count.
.PARAMETER Plain
No flair: no emoji, animation, card or notification.
.PARAMETER NoNotify
No desktop notification.
.PARAMETER LogFile
Log path. Default <DATA_ROOT>/<repo>/ci/logs/run/<date>-<handle>.log.

.NOTES
Exit codes:
  0    every suite passed
  1    failed: the suites could not run (manifest, worktree)
  2    usage error
  3    partial: a suite failed, or the selection holds no suite
  130  interrupted
Suite output goes to <log folder>/<date>-<handle>/<suite>.log, run history to
<log folder>/history.jsonl. DATA_ROOT defaults to C:\data on Windows and ~/data elsewhere.
#>
[CmdletBinding()]
param(
    [string] $Tier,
    [string] $Suite,
    [switch] $Clean,
    [switch] $Plain,
    [switch] $NoNotify,
    [string] $LogFile,
    [Parameter(ValueFromRemainingArguments)] [string[]] $Rest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Command = 'run'
$Adjectives = ('amber brave calm clever cosmic crisp daring deft eager fabled fierce gentle glad golden grand hardy ' +
    'humble jolly keen lively lucky merry mighty misty nimble noble plucky proud quick quiet rapid rustic sage shiny ' +
    'silent sleek snowy spry steady stout sunny swift tidy vivid warm wild wise witty zesty') -split ' '
$Animals = ('badger beaver bison bobcat crane dingo eagle egret ferret finch fox gecko heron ibex jackal koala lemur ' +
    'lynx magpie marmot marten mink moose newt ocelot okapi osprey otter owl panda puffin quail raven robin salmon ' +
    'seal shrew sloth stoat swallow tapir tern toad trout vole walrus wombat wren yak') -split ' '
$Remarks = @('still chewing', 'big one, this', 'not stuck, just thorough', "hasn't crashed, promise",
    'patience is a virtue', 'nearly… probably')
$Moons = @('🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘')
$Spark = '▁▂▃▄▅▆▇█'
$SparkAscii = ' .:-=+*#'
$Braille = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
$Eighths = ' ▏▎▍▌▋▊▉█'
$Stage256 = @(75, 141, 213, 111, 177, 117)
$Stage16 = @(34, 35, 94, 95)
$Rainbow256 = @(210, 179, 143, 78, 38, 75, 147, 176)
$Rainbow16 = @(91, 93, 92, 96, 95)
$Moods = @{ done = '✨'; partial = '🧩'; interrupted = '✋'; failed = '💥' }
$EventColours = @{ progress = @(36); done = @(32); partial = @(33); interrupted = @(33); failed = @(1, 31)
    milestone = @(35); start = @(1) }
$LevelColours = @{ debug = @(2); warning = @(33); error = @(1, 31) }
$SnakeBits = @(@(0x01, 0x08), @(0x02, 0x10), @(0x04, 0x20), @(0x40, 0x80))
$EventWidth = 11
$IdleAfter = 15
$NotifyAfter = 60
$CardAfter = 5
$OutputTail = 40
$AnsiPattern = '\e\[[0-9;]*m'
$ToastScript = @'
$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
$null = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime]
$xml = [Windows.Data.Xml.Dom.XmlDocument]::new()
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/></binding></visual></toast>')
$text = $xml.GetElementsByTagName('text')
[void] $text.Item(0).AppendChild($xml.CreateTextNode($env:CLI_NOTIFY_TITLE))
[void] $text.Item(1).AppendChild($xml.CreateTextNode($env:CLI_NOTIFY_BODY))
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show(
    [Windows.UI.Notifications.ToastNotification]::new($xml))
'@
$SuiteScript = @'
$ProgressPreference = 'SilentlyContinue'
$global:LASTEXITCODE = 0
try {
    & ([scriptblock]::Create($env:CI_SUITE_RUN)) *> $env:CI_SUITE_LOG
    $ok = $?
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $env:CI_SUITE_LOG
    exit 1
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if (-not $ok) { exit 1 }
exit 0
'@

# ---- pure helpers -----------------------------------------------------------------------------

function Format-Duration([double] $Seconds) {
    $s = [long][Math]::Floor($Seconds)
    if ($s -ge 3600) { return '{0}h{1:00}m' -f [Math]::Floor($s / 3600), [Math]::Floor(($s % 3600) / 60) }
    if ($s -ge 60) { return '{0}m{1:00}s' -f [Math]::Floor($s / 60), ($s % 60) }
    "${s}s"
}

function Get-Moon([datetime] $Now) { $Moons[[int][Math]::Floor(($Now.Hour * 60 + $Now.Minute + 90) / 180) % 8] }

function Get-Sparkline([int[]] $Deltas, [string] $Cells = $Spark) {
    $top = ($Deltas | Measure-Object -Maximum).Maximum
    if (-not $top) { $top = 1 }
    foreach ($d in $Deltas) {
        if ($d -eq 0) { [string] $Cells[0] }
        else { [string] $Cells[[Math]::Max(1, [int][Math]::Round($d / $top * ($Cells.Length - 1)))] }
    }
}

function Get-Scanner([int] $Frame, [int] $Width = 10) {
    $span = 2 * ($Width - 1)
    $head = $Frame % $span
    $step = 1
    if ($head -ge $Width) { $head = $span - $head; $step = -1 }
    $cells = [char[]] ('░' * $Width)
    $glyphs = '█▓▒'
    for ($offset = 0; $offset -lt 3; $offset++) {
        $at = $head - $offset * $step
        if ($at -ge 0 -and $at -lt $Width) { $cells[$at] = $glyphs[$offset] }
    }
    -join $cells
}

function Get-AsciiScanner([int] $Frame, [int] $Width = 10) {
    $span = 2 * ($Width - 3)
    $head = $Frame % $span
    if ($head -ge $Width - 2) { $head = $span - $head }
    (' ' * $head) + '<=>' + (' ' * ($Width - 3 - $head))
}

# The snake spinner: cells are encoded as (x + 8) * 8 + y, oldest first, head last.
function Get-SnakeEntry { , [int[]] @(for ($x = -6; $x -le 0; $x++) { ($x + 8) * 8 }) }

function Get-SnakeMove([int[]] $Body) {
    $head = $Body[-1]
    $hx = [Math]::Floor($head / 8) - 8
    $hy = $head % 8
    $blocked = $Body[1..($Body.Count - 1)]
    $out = foreach ($d in @(@(1, 0), @(-1, 0), @(0, 1), @(0, -1))) {
        $x = $hx + $d[0]; $y = $hy + $d[1]
        $cell = ($x + 8) * 8 + $y
        if ($x -ge 0 -and $x -lt 4 -and $y -ge 0 -and $y -lt 4 -and $blocked -notcontains $cell) { $cell }
    }
    , [int[]] @($out)
}

function Step-Snake([int[]] $Body, [Random] $Rng) {
    $free = Get-SnakeMove $Body
    if (-not $free.Count) { return $null }
    $rest = [int[]] $Body[1..($Body.Count - 1)]
    $rooms = @(foreach ($cell in $free) { (Get-SnakeMove ([int[]] ($rest + $cell))).Count })
    $cornered = -not ($rooms | Where-Object { $_ })
    $heading = $Body[-1] - $Body[-2]
    $weights = @(for ($i = 0; $i -lt $free.Count; $i++) {
            $straight = if ($free[$i] - $Body[-1] -eq $heading) { 3.0 } else { 1.0 }
            $straight * $(if ($cornered) { 1 } else { $rooms[$i] })
        })
    $pick = $Rng.NextDouble() * ($weights | Measure-Object -Sum).Sum
    $chosen = $free[-1]
    for ($i = 0; $i -lt $free.Count; $i++) {
        $pick -= $weights[$i]
        if ($pick -lt 0) { $chosen = $free[$i]; break }
    }
    , [int[]] ($rest + $chosen)
}

function Get-SnakeGlyph([int[]] $Body) {
    $cells = @(0, 0)
    foreach ($cell in $Body) {
        $x = [Math]::Floor($cell / 8) - 8; $y = $cell % 8
        if ($x -ge 0 -and $x -lt 4 -and $y -ge 0 -and $y -lt 4) {
            $cells[[Math]::Floor($x / 2)] = $cells[[Math]::Floor($x / 2)] -bor $SnakeBits[$y][$x % 2]
        }
    }
    [string] [char] (0x2800 + $cells[0]) + [char] (0x2800 + $cells[1])
}

# The next spinner frame as @(glyph, crashed); after a crash, six red frames shown and blank.
function Get-SnakeFrame([hashtable] $Snake) {
    if ($Snake.Flash) {
        $Snake.Flash--
        $glyph = if ($Snake.Flash % 2) { Get-SnakeGlyph $Snake.Body } else { [string] [char] 0x2800 * 2 }
        if (-not $Snake.Flash) { $Snake.Body = Get-SnakeEntry }
        return @($glyph, $true)
    }
    $moved = Step-Snake $Snake.Body $Snake.Rng
    if ($null -eq $moved) { $Snake.Flash = 6; return Get-SnakeFrame $Snake }
    $Snake.Body = $moved
    @((Get-SnakeGlyph $moved), $false)
}

function Get-Crc32([string] $Text) {
    [uint32] $crc = 0xFFFFFFFFu
    foreach ($b in [Text.Encoding]::UTF8.GetBytes($Text)) {
        $crc = $crc -bxor [uint32] $b
        for ($k = 0; $k -lt 8; $k++) {
            if ($crc -band 1u) { $crc = [uint32] (($crc -shr 1) -bxor 0xEDB88320u) } else { $crc = [uint32] ($crc -shr 1) }
        }
    }
    [uint32] ($crc -bxor 0xFFFFFFFFu)
}

function ConvertFrom-Oklch([double] $Lightness, [double] $Chroma, [double] $Hue) {
    $h = $Hue * [Math]::PI / 180
    $a = $Chroma * [Math]::Cos($h); $b = $Chroma * [Math]::Sin($h)
    $l = [Math]::Pow($Lightness + 0.3963377774 * $a + 0.2158037573 * $b, 3)
    $m = [Math]::Pow($Lightness - 0.1055613458 * $a - 0.0638541728 * $b, 3)
    $s = [Math]::Pow($Lightness - 0.0894841775 * $a - 1.2914855480 * $b, 3)
    $linear = @((4.0767416621 * $l - 3.3077115913 * $m + 0.2309699292 * $s),
        (-1.2684380046 * $l + 2.6097574011 * $m - 0.3413193965 * $s),
        (-0.0041960863 * $l - 0.7034186147 * $m + 1.7076147010 * $s))
    foreach ($x in $linear) {
        $x = [Math]::Min(1.0, [Math]::Max(0.0, $x))
        $e = if ($x -le 0.0031308) { 12.92 * $x } else { 1.055 * [Math]::Pow($x, 1 / 2.4) - 0.055 }
        [int][Math]::Round(255 * $e)
    }
}

function Format-Bar([double] $Fraction, [int] $Width) {
    $cells = [int][Math]::Round([Math]::Max(0.0, [Math]::Min(1.0, $Fraction)) * $Width * 8)
    $full = [int][Math]::Floor($cells / 8); $part = $cells % 8
    (('█' * $full) + $(if ($part) { $Eighths[$part] } else { '' })).PadRight($Width)
}

function Get-CellWidth([Text.Rune] $Rune) { if ($Rune.Value -ge 0x1F300 -or $Rune.Value -in 0x2728, 0x270B) { 2 } else { 1 } }

function Get-VisibleLength([string] $Text) {
    $n = 0
    foreach ($rune in ($Text -replace $AnsiPattern, '').EnumerateRunes()) { $n += Get-CellWidth $rune }
    $n
}

# Text cut to $Width visible cells, colour codes intact.
function Format-Fit([string] $Text, [int] $Width) {
    $out = [Text.StringBuilder]::new(); $used = 0; $i = 0
    $ansi = [regex] "\G$AnsiPattern"
    while ($i -lt $Text.Length) {
        $m = $ansi.Match($Text, $i)
        if ($m.Success) { [void] $out.Append($m.Value); $i += $m.Length; continue }
        $rune = [Text.Rune]::GetRuneAt($Text, $i)
        $w = Get-CellWidth $rune
        if ($used + $w -gt $Width) { break }
        [void] $out.Append($rune.ToString()); $used += $w; $i += $rune.Utf16SequenceLength
    }
    if ($Text.Contains("`e[")) { [void] $out.Append("`e[0m") }
    $out.ToString()
}

# Text as lines of at most $Width visible cells, broken at spaces, a word wider than a line broken
# across lines; continuation lines start with $Indent spaces.
function Format-Wrap([string] $Text, [int] $Width, [int] $Indent = 0) {
    $lines = [Collections.Generic.List[string]]::new()
    $current = [Collections.Generic.List[string]]::new()
    $used = 0; $room = $Width
    foreach ($word in (($Text -replace '[\t\n\r\v\f]', ' ') -split ' ')) {
        $size = Get-VisibleLength $word
        if ($current.Count -and $used + 1 + $size -gt $room) {
            $lines.Add(($current -join ' ').TrimEnd()); $current.Clear(); $used = 0; $room = [Math]::Max(1, $Width - $Indent)
        }
        if (-not $current.Count -and -not $word -and $lines.Count) { continue }
        while ($size -gt $room) {
            $head = (Format-Fit $word $room) -replace '\e\[0m$', ''
            if (-not $head) { $head = [Text.Rune]::GetRuneAt($word, 0).ToString() }
            $lines.Add($head)
            $word = $word.Substring($head.Length); $room = [Math]::Max(1, $Width - $Indent)
            $size = Get-VisibleLength $word
        }
        $used += $size + $(if ($current.Count) { 1 } else { 0 })
        $current.Add($word)
    }
    $lines.Add(($current -join ' ').TrimEnd())
    $pad = ' ' * $Indent
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($i) { $pad + $lines[$i] } else { $lines[$i] } }
}

function New-Handle([Random] $Rng) { "$($Adjectives[$Rng.Next($Adjectives.Count)])-$($Animals[$Rng.Next($Animals.Count)])" }

function Get-TimeLeft($Done, $Total, [double] $StartDone, [double] $Seconds) {
    if ($null -eq $Done -or $null -eq $Total -or $Done -le $StartDone) { return $null }
    [Math]::Max(0.0, ($Total - $Done) * $Seconds / ($Done - $StartDone))
}

# ---- style and console ------------------------------------------------------------------------

function Get-Width {
    if (-not $Style.Vt) { return 120 }
    try { [Math]::Min([Console]::WindowWidth, 120) } catch { 120 }
}

function Format-Paint([string] $Text, [int[]] $Codes) {
    if (-not $Style.Colour -or -not $Codes) { return $Text }
    "`e[$($Codes -join ';')m$Text`e[0m"
}

function Get-StageCode([string] $Name) {
    if (-not ($Style.Flair -and $Style.Colour)) { return }
    $key = Get-Crc32 $Name
    if ($Style.Depth -ge 8) { 38, 5, $Stage256[$key % $Stage256.Count] } else { $Stage16[$key % $Stage16.Count] }
}

function Get-RainbowCode([int] $Frame) {
    if (-not ($Style.Flair -and $Style.Colour)) { if ($Style.Colour) { 36 }; return }
    if ($Style.Depth -eq 24) { 38, 2; ConvertFrom-Oklch 0.75 0.12 ($Frame * 24 % 360); return }
    if ($Style.Depth -eq 8) { 38, 5, $Rainbow256[$Frame % $Rainbow256.Count]; return }
    $Rainbow16[$Frame % $Rainbow16.Count]
}

function Format-Meter([double] $Fraction, [int[]] $Codes, [int] $Width = 10) {
    if ($Style.Utf8) {
        $full = (Format-Bar $Fraction $Width).TrimEnd()
        $rest = '░' * ($Width - $full.Length)
    }
    else {
        $n = [int][Math]::Round([Math]::Max(0.0, [Math]::Min(1.0, $Fraction)) * $Width)
        $full = '#' * $n; $rest = '.' * ($Width - $n)
    }
    (Format-Paint $full $Codes) + (Format-Paint $rest 2)
}

function Get-Prefix([string] $Level, [string] $Event) {
    $now = Get-Date
    $moon = if ($Style.Emoji) { "$(Get-Moon $now) " } else { '' }
    $glyph = @{ debug = '.'; info = $Style.Dot; warning = '!'; error = 'x' }[$Level]
    $codes = $EventColours[$Event]
    if (-not $codes) { $codes = $LevelColours[$Level] }
    $text = "$moon$(Format-Paint $now.ToString('HH:mm:ss') 2) $glyph $(Format-Paint $Event.PadRight($EventWidth) $codes) "
    @($text, ($(if ($moon) { 3 } else { 0 }) + 9 + 2 + $EventWidth + 1))
}

function Save-Record([string] $Level, [string] $Event, [string] $Message) {
    if (-not $State.Log) { return }
    [Threading.Monitor]::Enter($State.SyncRoot)
    try { $State.Log.WriteLine(('{0:yyyy-MM-dd HH:mm:ss} {1,-7} {2} {3}' -f (Get-Date), $Level, $Event, $Message)) }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

function Format-Frame([string[]] $Lines) {
    $up = if ($State.LiveRows) { $State.LiveRows - 1 } else { 0 }
    $indent = (Get-Prefix 'info' 'progress')[1]
    $wrapped = @(foreach ($line in $Lines) { Format-Wrap $line ((Get-Width) - 1) $indent })
    $rows = ($wrapped | ForEach-Object { "$_`e[K" }) -join "`n"
    $hide = if ($State.LiveRows) { '' } else { "`e[?25l" }
    $State.LiveRows = $wrapped.Count
    "$hide$(if ($up) { "`e[${up}A" })`r$rows`e[J"
}

function Clear-Live {
    if (-not $State.LiveRows) { return }
    $up = $State.LiveRows - 1
    [Console]::Error.Write("`e[?2026h$(if ($up) { "`e[${up}A" })`r`e[J")
    $State.LiveRows = 0
}

function Restore-Live {
    if (-not $Style.Vt) { return }
    if ($State.LiveLines) { [Console]::Error.Write((Format-Frame $State.LiveLines)) }
    [Console]::Error.Write("`e[?2026l")
}

# One record: saved plain to the log, shown formatted on stderr above the live block.
function Write-Record([string] $Level, [string] $Event, [string] $Message, [string] $Flourish = '',
    [string[]] $Highlight = @(), [bool] $Show = $true, [bool] $Save = $true) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try {
        if ($Save) { Save-Record $Level $Event $Message }
        if (-not $Show -or ($Level -eq 'debug' -and -not $State.Verbose)) { return }
        $prefix, $indent = Get-Prefix $Level $Event
        $body = if ($Flourish) { "$Flourish $Message" } else { $Message }
        $lines = @(Format-Wrap $body ([Math]::Max(20, (Get-Width) - $indent)))
        foreach ($name in $Highlight) {
            $painted = Format-Paint $name @(Get-StageCode $name)
            $lines = @($lines | ForEach-Object { $_.Replace($name, $painted) })
        }
        Clear-Live
        [Console]::Error.Write($prefix + ($lines -join ("`n" + ' ' * $indent)) + "`n")
        Restore-Live
    }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

# Pre-formatted lines (a suite's output, the summary card), written as one unit.
function Write-Block([string[]] $Lines) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try { Clear-Live; [Console]::Error.Write(($Lines -join "`n") + "`n"); Restore-Live }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

function Write-Result([string] $Text) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try { Clear-Live; [Console]::Out.WriteLine($Text); [Console]::Out.Flush(); Restore-Live }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
    Save-Record 'info' 'result' $Text
}

function Write-Live([string[]] $Lines) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try { $State.LiveLines = $Lines; [Console]::Error.Write("`e[?2026h" + (Format-Frame $Lines) + "`e[?2026l") }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

function Close-Live {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try {
        if ($State.LiveRows) { [Console]::Error.Write("`n`e[?25h") }
        $State.LiveRows = 0; $State.LiveLines = $null
    }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

# ---- progress ---------------------------------------------------------------------------------

function Get-Now { $State.Clock.Elapsed.TotalSeconds }

function Set-Stage([string] $Name, $Done = $null, $Total = $null) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try {
        $now = Get-Now
        if ($Name -ne $State.Stage) {
            $State.Stage = $Name; $State.Since = $now; $State.MovedAt = $now
            $State.Deltas.Clear()
            $State.Sampled = $State.StartDone = $(if ($null -ne $Done) { $Done } else { 0 })
        }
        if ($Done -ne $State.Done) { $State.MovedAt = $now }
        $State.Done = $Done; $State.Total = $Total
    }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

function Set-Item([string] $Name) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try { $State.Item = $Name; $State.ItemSince = Get-Now }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

function Step-Sample {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try {
        if ($null -ne $State.Done) {
            $State.Deltas.Add([Math]::Max(0, $State.Done - $State.Sampled))
            if ($State.Deltas.Count -gt 16) { $State.Deltas.RemoveAt(0) }
            $State.Sampled = $State.Done
        }
    }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }
}

# The heartbeat's factual record and the live block's rows for one frame.
function Get-HeartbeatLine([int] $Frame, [double] $Now) {
    [Threading.Monitor]::Enter($State.SyncRoot)
    try {
        $name = $State.Stage; $done = $State.Done; $total = $State.Total; $since = $State.Since
        $movedAt = $State.MovedAt; $startDone = $State.StartDone; $deltas = [int[]] @($State.Deltas)
        $item = $State.Item; $itemSince = $State.ItemSince; $passed = $State.Passed; $failed = $State.Failed
    }
    finally { [Threading.Monitor]::Exit($State.SyncRoot) }

    $elapsed = $Now - $State.Started
    $left = Get-TimeLeft $done $total $startDone ($Now - $since)
    $counts = if ($null -eq $done) { '' } elseif ($total) { ': {0:N0}/{1:N0}' -f $done, $total } else { ': {0:N0}' -f $done }
    $idleFrom = if ($item) { $itemSince } else { [Math]::Max($since, $movedAt) }
    $idle = $Now - $idleFrom
    $factualLeft = if ($null -ne $left) { "; ~$(Format-Duration ([Math]::Ceiling($left))) left" } else { '' }
    $factualItem = if ($item) { "; $item" } else { '' }
    $factualIdle = if ($idle -ge $IdleAfter) { " ($([int][Math]::Floor($idle))s on this step)" } else { '' }
    $factual = "$Command $($State.Handle); $(Format-Duration $elapsed) elapsed; $name$counts$factualLeft$factualItem$factualIdle"
    if (-not $Style.Vt) { return @($factual, @()) }

    $prefix, $indent = Get-Prefix 'info' 'progress'
    if ($Style.Flair -and $Style.Utf8) {
        $glyph, $crashed = Get-SnakeFrame $State.Snake
        $spinner = Format-Paint $glyph $(if ($crashed) { @(1, 31) } else { @(Get-RainbowCode $Frame) })
    }
    else {
        $frames = if ($Style.Utf8) { $Braille } else { '|/-\' }
        $spinner = Format-Paint ([string] $frames[$Frame % $frames.Length]) @(Get-RainbowCode $Frame)
    }
    $drift = @()
    if ($Style.Flair -and $Style.Colour -and $State.Typical) {
        $ratio = $elapsed / $State.Typical
        $drift = if ($ratio -le 1.0) { @(32) } elseif ($ratio -le 1.5) { @(33) } else { @(31) }
    }
    $codes = @(Get-StageCode $name)
    $pad = ' ' * $indent
    $row = (Format-Paint $name $codes)
    if ($null -ne $done -and $total) { $row += " $(Format-Meter ($done / $total) $codes)" }
    if ($null -ne $done) { $row += if ($total) { ' {0:N0}/{1:N0}' -f $done, $total } else { ' {0:N0}' -f $done } }
    if ($Style.Emoji) {
        if ($passed) { $row += ' ' + (Format-Paint "✓ $passed" 32) }
        if ($failed) { $row += ' ' + (Format-Paint "✗ $failed" 31) }
    }
    else {
        if ($passed) { $row += " · $passed passed" }
        if ($failed) { $row += " · $failed failed" }
    }
    if ($null -ne $left) { $row += ' · ' + (Format-Paint "~$(Format-Duration ([Math]::Ceiling($left))) left" 1) }
    if ($Style.Flair) {
        if ($null -ne $done -and $deltas.Count) {
            $cells = @(Get-Sparkline $deltas $(if ($Style.Utf8) { $Spark } else { $SparkAscii }))
            $row += ' ' + (-join @(for ($i = 0; $i -lt $cells.Count; $i++) {
                        if ($deltas[$i] -eq 0) { Format-Paint $cells[$i] 2 } else { Format-Paint $cells[$i] $codes }
                    }))
        }
        elseif ($null -eq $done) {
            $scan = if ($Style.Utf8) { Get-Scanner $Frame } else { Get-AsciiScanner $Frame }
            $row += ' [' + (Format-Paint $scan $codes) + ']'
        }
    }
    $idleText = ''
    if ($idle -ge $IdleAfter) {
        $remark = if ($Style.Flair) { "; $($Remarks[[int][Math]::Floor(($idle - $IdleAfter) / 10) % $Remarks.Count])" } else { '' }
        $idleText = Format-Paint " ($([int][Math]::Floor($idle))s on this step$remark)" 2
    }
    $lines = @("$prefix$spinner $Command $(Format-Paint (Format-Duration $elapsed) $drift)")
    if ($item) {
        $lines += "$pad$row"
        $itemCodes = @(Get-StageCode $item)
        $itemRow = "$pad$(Format-Paint $item $itemCodes)"
        if ($Style.Flair) {
            $scan = if ($Style.Utf8) { Get-Scanner $Frame } else { Get-AsciiScanner $Frame }
            $itemRow += ' [' + (Format-Paint $scan $itemCodes) + ']'
        }
        $lines += "$itemRow$idleText"
    }
    else { $lines += "$pad$row$idleText" }
    @($factual, $lines)
}

$Heartbeat = {
    param($Shared)
    foreach ($entry in $Shared.GetEnumerator()) { Set-Variable -Name $entry.Key -Value $entry.Value -Scope Global }
    $frameMs = if ($Style.Flair) { [Math]::Max(1, [int]($TickMs / 8)) } else { $TickMs }
    $tick = $TickMs / 1000
    $frame = 0; $saved = 0.0; $nextSample = $tick
    try {
        while (-not $State.Stop) {
            Start-Sleep -Milliseconds $frameMs
            $frame++
            $now = Get-Now
            if ($now -ge $nextSample) { Step-Sample; $nextSample += $tick }
            $factual, $lines = Get-HeartbeatLine $frame $now
            if ($now - $saved -ge 5 * $tick) { $saved = $now; Write-Record 'info' 'progress' $factual -Show (-not $Style.Vt) }
            if ($Style.Vt -and -not $State.Stop) { Write-Live $lines }
        }
    }
    catch { Save-Record 'debug' 'heartbeat' ($_ | Out-String) }
}

# ---- manifest, suites, history ----------------------------------------------------------------

function Get-DataRoot {
    if ($env:DATA_ROOT) { return $env:DATA_ROOT }
    if ($IsWindows) { 'C:\data' } else { Join-Path $HOME 'data' }
}

function Read-Manifest([string] $Path) {
    if (-not (Get-Command yq -ErrorAction SilentlyContinue)) { throw 'yq not found on PATH; install mikefarah yq 4' }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "no manifest at $Path" }
    $json = yq -o=json '.' $Path 2>&1
    if ($LASTEXITCODE) { throw "yq could not parse ${Path}: $($json -join ' ')" }
    $raw = ($json -join "`n") | ConvertFrom-Json -AsHashtable
    if ($raw -isnot [hashtable] -or -not $raw.ContainsKey('suites') -or -not $raw.suites) { throw "$Path has no suites" }
    foreach ($entry in $raw.suites) {
        if ($entry -isnot [hashtable] -or -not $entry['name'] -or $entry['tier'] -notin 'quick', 'full' -or -not $entry['run']) {
            throw "$Path has a suite without a name, a quick or full tier, or a run command"
        }
        @{ name = [string] $entry['name']; tier = [string] $entry['tier']; run = [string] $entry['run']; local = [bool] $entry['local'] }
    }
}

# Runs one suite in a child pwsh from $Root, its output in $OutPath; returns its exit code.
function Invoke-Suite([hashtable] $Definition, [string] $Root, [string] $OutPath) {
    $psi = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($SuiteScript))
    foreach ($arg in '-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded) { $psi.ArgumentList.Add($arg) }
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.Environment['CI_SUITE_RUN'] = $Definition.run
    $psi.Environment['CI_SUITE_LOG'] = $OutPath
    $process = [Diagnostics.Process]::Start($psi)
    $State.Child = $process
    $process.StandardInput.Close()
    while (-not $process.WaitForExit(200)) { }
    $process.WaitForExit()
    $State.Child = $null
    $process.ExitCode
}

function Write-SuiteOutput([string] $Name, [string] $OutPath) {
    $lines = @(if (Test-Path -LiteralPath $OutPath) { Get-Content -LiteralPath $OutPath })
    $shown = @(if ($env:CI) { $lines } else { $lines | Select-Object -Last $OutputTail })
    $scope = if ($shown.Count -lt $lines.Count) { "last $($shown.Count) of $($lines.Count) lines" } else { "$($lines.Count) lines" }
    Write-Record 'warning' 'output' "$Name output, ${scope}: $OutPath" -Highlight @($Name)
    if ($shown) { Write-Block $shown }
}

function Invoke-Run {
    if ($ManifestError) { throw $ManifestError }
    if (-not $Selected.Count) { return 3, 'the selection holds no suite' }
    $root = $RepoRoot
    $worktree = $null
    if ($Clean) {
        Set-Stage 'creating clean worktree'
        $worktree = Join-Path ([IO.Path]::GetTempPath()) "ci-run-$($State.Handle)"
        $out = git -C $RepoRoot worktree add --detach --quiet $worktree HEAD 2>&1
        if ($LASTEXITCODE) { throw "git worktree add failed: $($out -join ' ')" }
        $root = $worktree
    }
    try {
        $total = $Selected.Count
        New-Item -ItemType Directory -Force -Path $SuiteLogDir | Out-Null
        Write-Record 'info' 'progress' ('suites: 0/{0:N0}' -f $total) -Highlight @('suites')
        Set-Stage 'suites' 0 $total
        $keyShown = $false
        for ($i = 0; $i -lt $total; $i++) {
            $definition = $Selected[$i]
            Set-Item $definition.name
            $outPath = Join-Path $SuiteLogDir "$($definition.name).log"
            $began = Get-Now
            $exit = Invoke-Suite $definition $root $outPath
            $seconds = (Get-Now) - $began
            $State.SuiteTimes[$definition.name] = $seconds
            if ($seconds -gt $State.Slowest[1]) { $State.Slowest = @($definition.name, $seconds) }
            if ($exit -eq 0) {
                $State.Passed++
                Write-Result "passed $($definition.name) $(Format-Duration $seconds)"
            }
            else {
                $State.Failed++
                Write-Result "failed $($definition.name) $(Format-Duration $seconds); exit=$exit"
                Write-SuiteOutput $definition.name $outPath
            }
            if ($Style.Emoji -and -not $keyShown) {
                Write-Record 'info' 'progress' 'key: ✓ suites passed, ✗ suites failed' -Save $false
                $keyShown = $true
            }
            Set-Stage 'suites' ($i + 1) $total
        }
        Set-Item ''
        Write-Record 'info' 'progress' ('suites complete: {0:N0}/{0:N0}' -f $total) -Highlight @('suites')
    }
    finally {
        if ($worktree) {
            Set-Item ''
            Set-Stage 'removing clean worktree'
            git -C $RepoRoot worktree remove --force $worktree 2>&1 | Out-Null
        }
    }
    if ($State.Failed) { return 3, "$($State.Failed) of $total suites failed" }
    0, ''
}

function Read-History([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return , @() }
    try { $lines = @(Get-Content -LiteralPath $Path -Tail 50) } catch { return $null }
    , @(foreach ($line in $lines) { try { $line | ConvertFrom-Json -AsHashtable } catch { } })
}

function Get-Comparable([object[]] $History, [int] $Total, [string] $Selection) {
    @($History | Where-Object {
            $_.exit -eq 0 -and -not $_.dry_run -and $_.selection -eq $Selection -and $_.total -and
            [Math]::Abs($_.total - $Total) -le 0.2 * $Total
        })
}

function Get-CardLine([string] $Handle, [System.Collections.Specialized.OrderedDictionary] $Counts,
    [System.Collections.Specialized.OrderedDictionary] $Stages, [object[]] $Slowest, [string] $Versus) {
    $width = [Math]::Min((Get-Width), 72)
    $inner = $width - 4
    $box = if ($Style.Utf8) { '╭─╮│╰╯' } else { '+-+|++' }
    $nameWidth = 18; $metaWidth = 13
    $barWidth = [Math]::Max(8, $inner - $nameWidth - $metaWidth - 1)
    $rows = [Collections.Generic.List[string]]::new()
    $rows.Add((($Counts.GetEnumerator() | ForEach-Object { '{0} {1:N0}' -f $_.Key, $_.Value }) -join ' · '))
    $rows.Add('')
    $longest = ($Stages.Values | Measure-Object -Maximum).Maximum; if (-not $longest) { $longest = 1 }
    $sum = ($Stages.Values | Measure-Object -Sum).Sum; if (-not $sum) { $sum = 1 }
    foreach ($stage in $Stages.GetEnumerator()) {
        $shape = if ($Style.Utf8) { Format-Bar ($stage.Value / $longest) $barWidth }
        else { ('#' * [int][Math]::Round($stage.Value / $longest * $barWidth)).PadRight($barWidth) }
        $label = $stage.Key.Substring(0, [Math]::Min($nameWidth, $stage.Key.Length)).PadRight($nameWidth)
        $rows.Add("$label$(Format-Paint $shape @(Get-StageCode $stage.Key)) $((Format-Duration $stage.Value).PadLeft(6)) $(('{0:P0}' -f ($stage.Value / $sum)).PadLeft(5))")
    }
    $rows.Add('')
    if ($Slowest[0]) { $rows.Add(('{0,-10}{1}  {2:N1}s' -f 'slowest', $Slowest[0], $Slowest[1])) }
    if ($Versus) { $rows.Add(('{0,-10}{1}' -f 'vs last', $Versus)) }
    $title = "$($box[1]) $Command · $Handle "
    "$($box[0])$title$([string] $box[1] * ($width - 2 - (Get-VisibleLength $title)))$($box[2])"
    foreach ($row in $rows) {
        if ((Get-VisibleLength $row) -gt $inner) { $row = Format-Fit $row $inner }
        "$($box[3]) $row$(' ' * ($inner - (Get-VisibleLength $row))) $($box[3])"
    }
    "$($box[4])$([string] $box[1] * ($width - 2))$($box[5])"
}

function Send-Notification([string] $Title, [string] $Body) {
    $env:CLI_NOTIFY_TITLE = $Title; $env:CLI_NOTIFY_BODY = $Body
    try {
        if ($IsWindows) {
            $psi = [Diagnostics.ProcessStartInfo]::new('powershell.exe')
            $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($ToastScript))
            foreach ($arg in '-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded) { $psi.ArgumentList.Add($arg) }
        }
        elseif ($IsMacOS) {
            $psi = [Diagnostics.ProcessStartInfo]::new('osascript')
            foreach ($arg in '-e', 'on run argv', '-e', 'display notification (item 2 of argv) with title (item 1 of argv)',
                '-e', 'end run', $Title, $Body) { $psi.ArgumentList.Add($arg) }
        }
        elseif (Get-Command notify-send -ErrorAction SilentlyContinue) {
            $psi = [Diagnostics.ProcessStartInfo]::new('notify-send')
            foreach ($arg in '--', $Title, $Body) { $psi.ArgumentList.Add($arg) }
        }
        else { Write-Record 'debug' 'notify' 'notify-send not found; no desktop notification'; return }
        $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
        [void] [Diagnostics.Process]::Start($psi)
    }
    catch { Write-Record 'debug' 'notify' "desktop notification failed: $($_.Exception.Message)" }
}

# Dot-sourcing loads the functions only (the tests do this).
if ($MyInvocation.InvocationName -eq '.') { return }

# ---- entry point ------------------------------------------------------------------------------

function Exit-Usage([string] $Message) { [Console]::Error.WriteLine("usage: $Message"); exit 2 }

if ($Rest) { Exit-Usage "unknown argument $($Rest[0])" }
if ($Tier -and $Suite) { Exit-Usage '-Tier and -Suite exclude each other' }
if (-not $Tier -and -not $Suite) { $Tier = 'quick' }
if ($Tier -and $Tier -notin 'quick', 'full') { Exit-Usage "-Tier must be quick or full, not '$Tier'" }
if ($LogFile) {
    $logFolder = Split-Path -Parent ([IO.Path]::GetFullPath($LogFile))
    if (-not (Test-Path -LiteralPath $logFolder -PathType Container)) { Exit-Usage "-LogFile folder $logFolder does not exist" }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$commonDir = git -C $RepoRoot rev-parse --path-format=absolute --git-common-dir 2>$null
$RepoName = if ($LASTEXITCODE -eq 0 -and $commonDir) { Split-Path -Leaf (Split-Path -Parent $commonDir) } else { Split-Path -Leaf $RepoRoot }

$ManifestError = $null
$Manifest = @()
try { $Manifest = @(Read-Manifest (Join-Path $RepoRoot 'ci/suites.yml')) } catch { $ManifestError = $_.Exception.Message }
if ($Suite) {
    $names = @($Suite -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if (-not $names) { Exit-Usage '-Suite names no suite' }
    if (-not $ManifestError) {
        $unknown = @($names | Where-Object { $_ -notin $Manifest.name })
        if ($unknown) { Exit-Usage "unknown suite $($unknown -join ', '); suites: $($Manifest.name -join ', ')" }
    }
    $Selected = @($Manifest | Where-Object { $_.name -in $names })
    $Selection = 'suites=' + (($names | Sort-Object) -join ',')
}
else {
    $Selected = @($Manifest | Where-Object { $Tier -eq 'full' -or $_.tier -eq 'quick' })
    $Selection = "tier=$Tier"
}

$vt = -not [Console]::IsErrorRedirected
$utf8 = [Console]::OutputEncoding.CodePage -eq 65001
$depth = if ($env:COLORTERM -in 'truecolor', '24bit' -or $env:WT_SESSION) { 24 }
elseif (($env:TERM -and $env:TERM.Contains('256')) -or $IsWindows) { 8 } else { 4 }
$Style = @{
    Vt = $vt; Flair = $vt -and -not $Plain; Colour = $vt -and -not $env:NO_COLOR; Utf8 = $utf8
    Emoji = $vt -and -not $Plain -and $utf8; Depth = $depth; Dot = $(if ($utf8) { '·' } else { '-' })
}
$rng = if ($env:CI_RUN_SEED) { [Random]::new([int] $env:CI_RUN_SEED) } else { [Random]::new() }
$handle = New-Handle $rng
$logDir = Join-Path (Get-DataRoot) $RepoName 'ci' 'logs' $Command
$logPath = if ($LogFile) { [IO.Path]::GetFullPath($LogFile) } else { Join-Path $logDir "$(Get-Date -Format 'yyyy-MM-dd')-$handle.log" }
$SuiteLogDir = Join-Path $logDir "$(Get-Date -Format 'yyyy-MM-dd')-$handle"
$historyPath = Join-Path $logDir 'history.jsonl'

$State = [hashtable]::Synchronized(@{
        Stage = 'starting'; Done = $null; Total = $null; Since = 0.0; MovedAt = 0.0; StartDone = 0; Sampled = 0
        Deltas = [Collections.Generic.List[int]]::new(); Item = ''; ItemSince = 0.0; Passed = 0; Failed = 0
        SuiteTimes = [ordered] @{}; Slowest = @('', 0.0); Stop = $false; LiveRows = 0; LiveLines = $null; Log = $null
        Verbose = $PSBoundParameters.ContainsKey('Verbose'); Clock = [Diagnostics.Stopwatch]::StartNew(); Started = 0.0
        Handle = $handle; Typical = $null; Child = $null
        Snake = @{ Body = (Get-SnakeEntry); Flash = 0; Rng = [Random]::new() }
    })

try {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
    $State.Log = [IO.StreamWriter]::new($logPath, $true, [Text.UTF8Encoding]::new($false))
    $State.Log.AutoFlush = $true
}
catch { Write-Record 'warning' 'warning' "log not saved: $($_.Exception.Message)" '' @() $true $false }

$history = Read-History $historyPath
if ($null -eq $history) { Write-Record 'warning' 'warning' "history not readable: $historyPath"; $history = @() }
$peers = @(Get-Comparable $history $Selected.Count $Selection)
if ($peers) {
    $recent = @($peers | Select-Object -Last 5 | ForEach-Object { [double] $_.elapsed_s } | Sort-Object)
    $State.Typical = if ($recent.Count % 2) { $recent[[Math]::Floor($recent.Count / 2)] }
    else { ($recent[$recent.Count / 2 - 1] + $recent[$recent.Count / 2]) / 2 }
}

$startedAt = Get-Date
Write-Record 'info' 'start' "$Command $handle; log=$logPath"
$flags = "clean=$Clean plain=$Plain flair=$($Style.Flair) colour=$($Style.Colour)".ToLowerInvariant()
Write-Record 'debug' 'settings' "selection=$Selection $flags root=$RepoRoot"

$shared = @{ State = $State; Style = $Style; TickMs = $(if ($env:CI_RUN_TICK_MS) { [int] $env:CI_RUN_TICK_MS } else { 1000 }) }
foreach ($name in 'Command', 'Remarks', 'Moons', 'Spark', 'SparkAscii', 'Braille', 'Eighths', 'Stage256', 'Stage16',
    'Rainbow256', 'Rainbow16', 'EventColours', 'LevelColours', 'SnakeBits', 'EventWidth', 'IdleAfter', 'AnsiPattern') {
    $shared[$name] = Get-Variable -Name $name -ValueOnly
}
$iss = [initialsessionstate]::CreateDefault2()
foreach ($function in Get-ChildItem function: | Where-Object { $_.ScriptBlock.File -eq $PSCommandPath }) {
    $iss.Commands.Add([Management.Automation.Runspaces.SessionStateFunctionEntry]::new($function.Name, $function.Definition))
}
$beat = [powershell]::Create($iss)
[void] $beat.AddScript($Heartbeat).AddArgument($shared)
$beatHandle = $beat.BeginInvoke()

$code, $reason, $completed = 1, 'stopped', $false
try {
    $code, $reason = Invoke-Run
    $completed = $true
}
catch {
    Write-Record 'debug' 'traceback' ($_ | Out-String)
    $code, $reason, $completed = 1, $_.Exception.Message, $true
}
finally {
    if (-not $completed) { $code, $reason = 130, 'rerun to start again' }
    if ($State.Child -and -not $State.Child.HasExited) { try { $State.Child.Kill($true) } catch { } }
    $State.Stop = $true
    try { [void] $beat.EndInvoke($beatHandle) } catch { }
    $beat.Dispose()
    if ($State.LiveRows) { Write-Live (Get-HeartbeatLine 0 (Get-Now))[1] }
    Close-Live
    $elapsed = Get-Now

    if ($code -eq 0 -and $peers -and $elapsed -lt ($peers.elapsed_s | Measure-Object -Minimum).Minimum) {
        $best = ($peers.elapsed_s | Measure-Object -Minimum).Minimum
        Write-Record 'info' 'milestone' "fastest $Command yet: $(Format-Duration $elapsed) (previous best $(Format-Duration $best))" `
            $(if ($Style.Emoji) { '🎉' } else { '' })
    }

    $counts = [ordered] @{ passed = $State.Passed; failed = $State.Failed }
    if ($elapsed -ge $CardAfter -and $State.SuiteTimes.Count) {
        $versus = ''
        if ($peers) {
            $last = $peers[-1]
            $change = ($last.elapsed_s - $elapsed) / $last.elapsed_s
            $word = if ($change -ge 0) { 'faster' } else { 'slower' }
            $versus = "$(Format-Duration $elapsed), $('{0:P0}' -f [Math]::Abs($change)) $word than $(Format-Duration $last.elapsed_s) ($($last.handle))"
        }
        $sum = ($State.SuiteTimes.Values | Measure-Object -Sum).Sum
        $summary = @("counts passed=$($State.Passed) failed=$($State.Failed)")
        $summary += @($State.SuiteTimes.GetEnumerator() | ForEach-Object {
                "stage $($_.Key) $(Format-Duration $_.Value) $('{0:P0}' -f ($_.Value / [Math]::Max($sum, 0.001)))"
            })
        if ($State.Slowest[0]) { $summary += 'slowest {0} {1:N1}s' -f $State.Slowest[0], $State.Slowest[1] }
        if ($versus) { $summary += "vs-last $versus" }
        if ($Style.Flair) {
            foreach ($line in $summary) { Save-Record 'info' 'summary' $line }
            Write-Block @(Get-CardLine $handle $counts $State.SuiteTimes $State.Slowest $versus)
        }
        else { foreach ($line in $summary) { Write-Record 'info' 'summary' $line } }
    }

    $event = switch ($code) { 0 { 'done' } 3 { 'partial' } 130 { 'interrupted' } default { 'failed' } }
    $level = switch ($event) { 'done' { 'info' } 'failed' { 'error' } default { 'warning' } }
    $because = if ($reason) { ": $reason" } else { '' }
    Write-Record $level $event "$Command $handle$because; exit=$code; elapsed=$(Format-Duration $elapsed)" `
        $(if ($Style.Emoji) { $Moods[$event] } else { '' })

    $record = [ordered] @{
        handle = $handle; started = $startedAt.ToString('yyyy-MM-ddTHH:mm:sszzz'); elapsed_s = [Math]::Round($elapsed, 1)
        exit = $code; dry_run = $false; total = $Selected.Count; selection = $Selection; counts = $counts
        stages = [ordered] @{}
    }
    foreach ($entry in $State.SuiteTimes.GetEnumerator()) { $record.stages[$entry.Key] = [Math]::Round($entry.Value, 1) }
    try { Add-Content -LiteralPath $historyPath -Value ($record | ConvertTo-Json -Compress -Depth 4) }
    catch { Write-Record 'warning' 'warning' "history not saved: $($_.Exception.Message)" }

    if ($Style.Flair -and -not $NoNotify -and -not $env:CI -and $elapsed -ge $NotifyAfter) {
        Send-Notification "$Command $event" "$handle · $(Format-Duration $elapsed) · passed $($State.Passed) · failed $($State.Failed)"
    }
    if ($State.Log) { $State.Log.Dispose(); $State.Log = $null }
}
exit $code
