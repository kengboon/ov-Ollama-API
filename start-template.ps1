#requires -Version 7.0
# start.ps1 — NoLlama launcher
# Activates the venv and runs nollama.py. nollama.py prints its own
# device-detection, per-model loading progress, and the "NoLlama ready"
# banner with the URL — the launcher does not poll /health or auto-open
# the browser. Open the URL from the banner yourself.
#
# Args are set by install.ps1 in the generated start.ps1.

param(
    [string]$ServerArgs = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv (Scripts on Windows, bin on POSIX)
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
& (Join-Path $ScriptDir "venv" $VenvBinDir "Activate.ps1")

$AllArgs = @((Join-Path $ScriptDir "nollama.py"))
if ($ServerArgs) {
    $AllArgs += $ServerArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
}

& python @AllArgs
