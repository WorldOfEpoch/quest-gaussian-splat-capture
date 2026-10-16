[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string] $Destination,
    [Parameter(Mandatory = $true)]
    [string] $PlatformSdkArchive
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$questRoot = Join-Path $repositoryRoot 'quest'
$dependencySpec = Get-Content -LiteralPath (Join-Path $questRoot 'dependencies.json') -Raw | ConvertFrom-Json
$destinationPath = [IO.Path]::GetFullPath($Destination)
$archivePath = [IO.Path]::GetFullPath($PlatformSdkArchive)
if ($destinationPath.TrimEnd('\','/') -eq [IO.Path]::GetPathRoot($destinationPath).TrimEnd('\','/')) {
    throw 'Choose a new named build directory, not a drive root.'
}
if ($destinationPath.Equals($repositoryRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $destinationPath.StartsWith($repositoryRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Keep the generated Unity project and vendor dependencies outside the public source checkout.'
}
if (Test-Path -LiteralPath $destinationPath) {
    throw 'Destination already exists. No files were changed; use a new build directory.'
}
if (-not (Test-Path -LiteralPath (Split-Path $destinationPath -Parent) -PathType Container)) {
    throw 'Create the intended parent directory first.'
}
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw 'Supply the separately downloaded Meta XR Platform SDK 85.0.0 archive.'
}
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $dependencySpec.metaPlatformSdk.archiveSha256) {
    throw 'Platform SDK archive differs from the pinned dependency. Review the dependency update before using it.'
}
$gitExecutable = (Get-Command git -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
if (-not $PSCmdlet.ShouldProcess($destinationPath, 'Fetch pinned public dependencies and prepare an isolated Unity project')) {
    return
}

function Invoke-CheckedGit {
    param([string[]] $GitArguments)
    & $gitExecutable @GitArguments
    if ($LASTEXITCODE -ne 0) { throw 'Git dependency preparation failed; the partial build directory is preserved for inspection.' }
}

function Get-PinnedDependency {
    param([string] $Url, [string] $Commit, [string] $Path)
    if ($Url -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$' -or
        $Commit -notmatch '^[a-f0-9]{40}$') {
        throw 'Invalid pinned dependency identity.'
    }
    Invoke-CheckedGit -GitArguments @('clone', '--filter=blob:none', '--no-checkout', '--', $Url, $Path)
    Invoke-CheckedGit -GitArguments @('-C', $Path, 'checkout', '--detach', $Commit)
    $head = & $gitExecutable -C $Path rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $Commit) { throw 'Dependency commit verification failed.' }
}

New-Item -ItemType Directory -Path $destinationPath | Out-Null
$dependenciesPath = Join-Path $destinationPath 'dependencies'
$unityProjectPath = Join-Path $destinationPath 'UnityProject'
New-Item -ItemType Directory -Path $dependenciesPath | Out-Null
New-Item -ItemType Directory -Path $unityProjectPath | Out-Null
$samplePath = Join-Path $dependenciesPath 'meta-camera-sample'
$rendererPath = Join-Path $dependenciesPath 'UnitySplats'

Get-PinnedDependency -Url $dependencySpec.metaCameraSample.repository -Commit $dependencySpec.metaCameraSample.commit -Path $samplePath
Get-PinnedDependency -Url $dependencySpec.unitySplats.repository -Commit $dependencySpec.unitySplats.commit -Path $rendererPath

foreach ($folder in @('Assets', 'Packages', 'ProjectSettings')) {
    $sampleFolder = Join-Path $samplePath $folder
    if (-not (Test-Path -LiteralPath $sampleFolder -PathType Container)) { throw 'Pinned sample project is incomplete.' }
    Copy-Item -LiteralPath $sampleFolder -Destination $unityProjectPath -Recurse
}
Copy-Item -LiteralPath (Join-Path $samplePath 'LICENSE.txt') -Destination (Join-Path $unityProjectPath 'META-SAMPLE-LICENSE.txt')
Copy-Item -LiteralPath $archivePath -Destination (Join-Path $dependenciesPath 'com.meta.xr.sdk.platform-85.0.0.tgz')

$projectAssets = Join-Path $unityProjectPath 'Assets'
foreach ($overlayItem in Get-ChildItem -LiteralPath (Join-Path $questRoot 'Assets') -Force) {
    Copy-Item -LiteralPath $overlayItem.FullName -Destination $projectAssets -Recurse -Force
}
foreach ($manifestName in @('manifest.json', 'packages-lock.json')) {
    Copy-Item -LiteralPath (Join-Path $questRoot ('Packages\' + $manifestName)) -Destination (Join-Path $unityProjectPath ('Packages\' + $manifestName)) -Force
}

$editorVersion = Get-Content -LiteralPath (Join-Path $unityProjectPath 'ProjectSettings\ProjectVersion.txt') -Raw
if ($editorVersion -notmatch ('(?m)^m_EditorVersion: ' + [regex]::Escape($dependencySpec.unityEditor) + '\s*$')) {
    throw 'Pinned sample Editor version differs from the dependency record.'
}
$receipt = [ordered]@{
    schemaVersion = 1
    preparedAtUtc = [DateTime]::UtcNow.ToString('o')
    unityEditor = $dependencySpec.unityEditor
    cameraSampleCommit = $dependencySpec.metaCameraSample.commit
    rendererCommit = $dependencySpec.unitySplats.commit
    platformSdkSha256 = $actualHash
    appRegistrationConfigured = $false
    unityBuildRun = $false
    headsetTestRun = $false
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destinationPath 'preparation-receipt.json') -Encoding UTF8
Write-Output ('Prepared isolated project: ' + $unityProjectPath)
Write-Output 'Next: supply your own package/app registration and run the documented Unity build. No app was built or installed.'
