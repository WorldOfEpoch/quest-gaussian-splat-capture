[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$bootstrapSource = Join-Path $repositoryRoot 'scripts\Prepare-QuestUnity.ps1'
$temporaryParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixtureRoot = Join-Path $temporaryParent ('quest-bootstrap-fixture-' + [guid]::NewGuid().ToString('N'))
$previousPath = $env:PATH
$previousFixtureRoot = $env:QUEST_SPLAT_BOOTSTRAP_FIXTURE
$previousFixtureLog = $env:QUEST_SPLAT_BOOTSTRAP_GIT_LOG
$results = [Collections.Generic.List[string]]::new()

function Assert-Check {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
    $results.Add($Message)
}

function Write-FixtureFile {
    param([string] $Path, [string] $Text)
    New-Item -ItemType Directory -Path (Split-Path $Path -Parent) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

try {
    New-Item -ItemType Directory -Path $fixtureRoot | Out-Null
    $fixtureRepository = Join-Path $fixtureRoot 'Public Source'
    $fixtureQuest = Join-Path $fixtureRepository 'quest'
    $fixtureBootstrap = Join-Path $fixtureRepository 'scripts\Prepare-QuestUnity.ps1'
    New-Item -ItemType Directory -Path (Split-Path $fixtureBootstrap -Parent) -Force | Out-Null
    New-Item -ItemType Directory -Path $fixtureQuest -Force | Out-Null
    Copy-Item -LiteralPath $bootstrapSource -Destination $fixtureBootstrap
    Copy-Item -LiteralPath (Join-Path $repositoryRoot 'quest\Assets') -Destination $fixtureQuest -Recurse
    Copy-Item -LiteralPath (Join-Path $repositoryRoot 'quest\Packages') -Destination $fixtureQuest -Recurse

    $sdkFixture = Join-Path $fixtureRoot 'SDK archive.tgz'
    Write-FixtureFile -Path $sdkFixture -Text 'Synthetic SDK bytes for offline bootstrap testing only.'
    $dependencySpec = Get-Content -LiteralPath (Join-Path $repositoryRoot 'quest\dependencies.json') -Raw | ConvertFrom-Json
    $dependencySpec.metaPlatformSdk.archiveSha256 = (Get-FileHash -LiteralPath $sdkFixture -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-FixtureFile -Path (Join-Path $fixtureQuest 'dependencies.json') -Text ($dependencySpec | ConvertTo-Json -Depth 6)

    $sampleFixture = Join-Path $fixtureRoot 'Sample'
    Write-FixtureFile -Path (Join-Path $sampleFixture 'Assets\Editor\KeepFromSample.txt') -Text 'Preserve this upstream sample file.'
    Write-FixtureFile -Path (Join-Path $sampleFixture 'Assets\Scripts\QuestCaptureController.cs') -Text 'This fixture must be replaced by the actual public overlay.'
    Write-FixtureFile -Path (Join-Path $sampleFixture 'Packages\manifest.json') -Text '{"dependencies":{}}'
    Write-FixtureFile -Path (Join-Path $sampleFixture 'ProjectSettings\ProjectVersion.txt') -Text "m_EditorVersion: 6000.0.66f2`n"
    Write-FixtureFile -Path (Join-Path $sampleFixture 'LICENSE.txt') -Text 'Synthetic sample license marker; no vendor code in this test.'
    Write-FixtureFile -Path (Join-Path $fixtureRoot 'Renderer\package.json') -Text '{"name":"com.arloopa.unitysplats","version":"1.2.0"}'

    $fakeGitSource = @'
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
public static class FixtureGit {
    public static int Main(string[] arguments) {
        try {
            var escaped = new List<string>();
            foreach (var argument in arguments)
                escaped.Add("\"" + argument.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"");
            File.AppendAllText(Environment.GetEnvironmentVariable("QUEST_SPLAT_BOOTSTRAP_GIT_LOG"),
                "{\"arguments\":[" + String.Join(",", escaped.ToArray()) + "]}" + Environment.NewLine);
            if (arguments.Length == 6 && arguments[0] == "clone" &&
                arguments[1] == "--filter=blob:none" && arguments[2] == "--no-checkout" && arguments[3] == "--") {
                var fixture = arguments[4].EndsWith("Unity-PassthroughCameraApiSamples.git") ? "Sample" :
                    arguments[4].EndsWith("UnitySplats.git") ? "Renderer" : null;
                if (fixture == null) throw new InvalidOperationException("Unexpected fixture remote");
                CopyDirectory(Path.Combine(Environment.GetEnvironmentVariable("QUEST_SPLAT_BOOTSTRAP_FIXTURE"), fixture), arguments[5]);
                return 0;
            }
            if (arguments.Length == 5 && arguments[0] == "-C" && arguments[2] == "checkout" && arguments[3] == "--detach") {
                File.WriteAllText(Path.Combine(arguments[1], ".fixture-head"), arguments[4]);
                return 0;
            }
            if (arguments.Length == 4 && arguments[0] == "-C" && arguments[2] == "rev-parse" && arguments[3] == "HEAD") {
                Console.WriteLine(File.ReadAllText(Path.Combine(arguments[1], ".fixture-head")));
                return 0;
            }
            throw new InvalidOperationException("Incomplete or altered Git arguments");
        } catch (Exception error) {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }
    private static void CopyDirectory(string source, string destination) {
        Directory.CreateDirectory(destination);
        foreach (var file in Directory.GetFiles(source))
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)));
        foreach (var directory in Directory.GetDirectories(source))
            CopyDirectory(directory, Path.Combine(destination, Path.GetFileName(directory)));
    }
}
'@
    $fakeBin = Join-Path $fixtureRoot 'Fake Git'
    $fakeSourcePath = Join-Path $fakeBin 'FixtureGit.cs'
    Write-FixtureFile -Path $fakeSourcePath -Text $fakeGitSource
    $frameworkCompiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if (-not (Test-Path -LiteralPath $frameworkCompiler -PathType Leaf)) {
        throw 'This Windows test requires the installed .NET Framework C# compiler.'
    }
    & $frameworkCompiler '/nologo' '/target:exe' ('/out:' + (Join-Path $fakeBin 'git.exe')) $fakeSourcePath
    if ($LASTEXITCODE -ne 0) { throw 'Could not compile the isolated fake Git executable.' }
    $gitLog = Join-Path $fixtureRoot 'git-arguments.jsonl'
    $env:QUEST_SPLAT_BOOTSTRAP_FIXTURE = $fixtureRoot
    $env:QUEST_SPLAT_BOOTSTRAP_GIT_LOG = $gitLog
    $env:PATH = $fakeBin + [IO.Path]::PathSeparator + $previousPath

    $buildDestination = Join-Path $fixtureRoot 'Build Workspace'
    & $fixtureBootstrap -Destination $buildDestination -PlatformSdkArchive $sdkFixture | Out-Null
    $projectPath = Join-Path $buildDestination 'UnityProject'
    Assert-Check (Test-Path -LiteralPath (Join-Path $projectPath 'Assets\Editor\KeepFromSample.txt')) 'Existing sample Editor subfolder content is preserved.'
    Assert-Check (-not (Test-Path -LiteralPath (Join-Path $projectPath 'Assets\Assets'))) 'Overlay does not create a nested Assets directory.'
    Assert-Check (-not (Test-Path -LiteralPath (Join-Path $projectPath 'Assets\Editor\Editor'))) 'Overlay does not create a nested Editor directory.'
    $assetFiles = @(Get-ChildItem -LiteralPath (Join-Path $fixtureQuest 'Assets') -Recurse -File)
    foreach ($sourceFile in $assetFiles) {
        $relative = $sourceFile.FullName.Substring($fixtureQuest.Length + 1)
        $copiedFile = Join-Path $projectPath $relative
        if (-not (Test-Path -LiteralPath $copiedFile -PathType Leaf) -or
            (Get-FileHash -LiteralPath $sourceFile.FullName).Hash -ne (Get-FileHash -LiteralPath $copiedFile).Hash) {
            throw 'Overlay source or GUID changed during preparation.'
        }
    }
    Assert-Check ($assetFiles.Count -gt 0) 'Every overlay source and metadata file matches its prepared copy.'

    $manifestPath = Join-Path $projectPath 'Packages\manifest.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($packageName in @('com.arloopa.unitysplats','com.meta.xr.sdk.platform')) {
        $uri = $manifest.dependencies.$packageName
        Assert-Check ($uri.StartsWith('file:')) ($packageName + ' uses an explicit local dependency URI.')
        $resolvedDependency = [IO.Path]::GetFullPath((Join-Path (Split-Path $manifestPath -Parent) $uri.Substring(5)))
        Assert-Check (Test-Path -LiteralPath $resolvedDependency) ($packageName + ' resolves to the prepared dependency.')
    }
    $receipt = Get-Content -LiteralPath (Join-Path $buildDestination 'preparation-receipt.json') -Raw | ConvertFrom-Json
    Assert-Check ($receipt.unityEditor -eq '6000.0.66f2' -and -not $receipt.unityBuildRun -and -not $receipt.headsetTestRun) 'Receipt distinguishes preparation from compilation and headset testing.'
    $gitRecords = @(Get-Content -LiteralPath $gitLog | ForEach-Object { $_ | ConvertFrom-Json })
    Assert-Check ($gitRecords.Count -eq 6) 'Both dependencies receive complete clone, checkout, and revision-verification arguments.'

    $refusedExisting = $false
    try { & $fixtureBootstrap -Destination $buildDestination -PlatformSdkArchive $sdkFixture | Out-Null }
    catch { $refusedExisting = $_.Exception.Message -like 'Destination already exists*' }
    Assert-Check $refusedExisting 'An existing build workspace is refused without replacement.'

    $wrongSdk = Join-Path $fixtureRoot 'Wrong SDK.tgz'
    Write-FixtureFile -Path $wrongSdk -Text 'Wrong synthetic SDK bytes.'
    $rejectedDestination = Join-Path $fixtureRoot 'Rejected Build'
    $refusedHash = $false
    try { & $fixtureBootstrap -Destination $rejectedDestination -PlatformSdkArchive $wrongSdk | Out-Null }
    catch { $refusedHash = $_.Exception.Message -like 'Platform SDK archive differs*' }
    Assert-Check ($refusedHash -and -not (Test-Path -LiteralPath $rejectedDestination)) 'A mismatched SDK hash is rejected before creating output.'

    $previewDestination = Join-Path $fixtureRoot 'Preview Only'
    & $fixtureBootstrap -Destination $previewDestination -PlatformSdkArchive $sdkFixture -WhatIf | Out-Null
    Assert-Check (-not (Test-Path -LiteralPath $previewDestination)) 'WhatIf creates no build workspace.'
    Assert-Check (@(Get-Content -LiteralPath $gitLog).Count -eq 6) 'Rejected and WhatIf runs launch no dependency fetch.'

    [pscustomobject]@{status='pass'; checks=$results.Count; fixture='synthetic offline Git, SDK, and sample project'; overlayFiles=$assetFiles.Count; details=@($results)} | ConvertTo-Json -Depth 4
}
finally {
    $env:PATH = $previousPath
    $env:QUEST_SPLAT_BOOTSTRAP_FIXTURE = $previousFixtureRoot
    $env:QUEST_SPLAT_BOOTSTRAP_GIT_LOG = $previousFixtureLog
    $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
    if ($resolvedFixture.StartsWith($temporaryParent.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
        [IO.Path]::GetFileName($resolvedFixture).StartsWith('quest-bootstrap-fixture-') -and
        (Test-Path -LiteralPath $resolvedFixture -PathType Container)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
