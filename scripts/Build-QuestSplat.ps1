param(
    [Parameter(Mandatory = $true)]
    [guid] $SessionId,
    [switch] $ReviewedNativeOrientation,
    [switch] $FreshBaseline,
    [switch] $Candidate,
    [guid] $FrozenSourceJobId,
    [int] $TargetTrainingStep,
    [switch] $IncrementalNewCaptures,
    [ValidateSet('quick', 'final', 'house')]
    [string] $Quality = 'final',
    [string] $CaptureSnapshot,
    [string] $DataDir = (Join-Path $env:LOCALAPPDATA 'QuestSplatBridge')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $DataDir 'tools'
$colmap = Join-Path $tools 'colmap-4.2.0\bin'
$nerfstudio = Join-Path $tools 'nerfstudio310-venv\Scripts'
$bridge = Join-Path $root '.venv\Scripts\quest-splat-bridge.exe'
foreach ($required in @((Join-Path $colmap 'colmap.exe'),
                       (Join-Path $nerfstudio 'ns-process-data.exe'),
                       (Join-Path $nerfstudio 'ns-train.exe'),
                       (Join-Path $nerfstudio 'ns-export.exe'), $bridge)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing local reconstruction dependency: $required"
    }
}

$env:QUEST_SPLAT_COLMAP_EXE = Join-Path $colmap 'colmap.exe'
$compat = Join-Path $root 'tools\colmap-compat'
$env:PATH = "$compat;$colmap;$nerfstudio;$env:PATH"
$buildArgs = @('--data-dir', $DataDir, '--build-session', $SessionId.ToString(), '--quality', $Quality)
if ($ReviewedNativeOrientation) { $buildArgs += '--reviewed-native-orientation' }
if ($FreshBaseline) { $buildArgs += '--fresh-baseline' }
if ($Candidate) { $buildArgs += '--candidate' }
if ($PSBoundParameters.ContainsKey('FrozenSourceJobId')) { $buildArgs += @('--frozen-source-job-id', $FrozenSourceJobId.ToString()) }
if ($PSBoundParameters.ContainsKey('TargetTrainingStep')) { $buildArgs += @('--target-training-step', [string]$TargetTrainingStep) }
if ($IncrementalNewCaptures) { $buildArgs += '--incremental-new-captures' }
if ($CaptureSnapshot) { $buildArgs += @('--capture-snapshot', $CaptureSnapshot) }
& $bridge @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
