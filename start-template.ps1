#requires -Version 7.0
# start.ps1 — NoLlama launcher
# Starts the server, waits for models to load, then opens the browser.
# Args are set by install.ps1 in the generated start.ps1

param(
    [string]$ServerArgs = ""
)

function Open-Url($url) {
    # Best-effort cross-platform browser open; tolerate headless / no handler.
    try { Start-Process $url } catch { Write-Host "  Open $url in your browser" -ForegroundColor DarkGray }
}

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Default port is 8000. If the user passed --port N in -ServerArgs (or
# edited the generated start.ps1 to do so), use that for the browser-open
# and /health polling so they match what nollama.py will bind to.
# Auto-detect was removed: on Windows, processes can share 0.0.0.0:N at
# the socket layer in ways that make bind-tests unreliable. Better to
# leave port handling explicit and let the user override on the CLI.
$Port = 8000
if ($ServerArgs -match '--port\s+(\d+)') {
    $Port = [int]$Matches[1]
}
$Url = "http://localhost:$Port"
Write-Host "  Launcher polling: $Url" -ForegroundColor DarkCyan

# Activate venv (Scripts on Windows, bin on POSIX)
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
& (Join-Path $ScriptDir "venv" $VenvBinDir "Activate.ps1")

# Start server in background. $ServerArgs is pass-through verbatim —
# nollama.py owns argparse, including --port (default 8000).
$AllArgs = @((Join-Path $ScriptDir "nollama.py"))
if ($ServerArgs) {
    $AllArgs += $ServerArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
}
$Server = Start-Process -FilePath python -ArgumentList $AllArgs `
    -NoNewWindow -PassThru

Write-Host ""
Write-Host "  NoLlama starting..." -ForegroundColor Cyan
Write-Host ""

# Poll /health until ready (or error/timeout)
$Spinner = @("|", "/", "-", "\")
$MaxWait = 120
$Elapsed = 0
$LastStatus = ""
$SpinIdx = 0

while ($Elapsed -lt $MaxWait) {
    Start-Sleep -Milliseconds 500
    $Elapsed += 0.5

    if ($Server.HasExited) {
        Write-Host ""
        Write-Host "  ERROR: Server process exited unexpectedly." -ForegroundColor Red
        exit 1
    }

    try {
        $resp = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2 -ErrorAction Stop
        $Status = $resp.status

        if ($Status -ne $LastStatus) {
            $LastStatus = $Status
            $DeviceInfo = ""
            if ($resp.devices) {
                $parts = @()
                $resp.devices.PSObject.Properties | ForEach-Object {
                    $devName = $_.Name.ToUpper()
                    $st = $_.Value.status
                    $modelName = $_.Value.model
                    if ($st -and $st -ne "not_configured") {
                        $parts += "${devName}: ${modelName} (${st})"
                    }
                }
                $DeviceInfo = $parts -join "  |  "
            }
            Write-Host ""
            Write-Host "  $DeviceInfo" -ForegroundColor DarkGray
        }

        if ($Status -eq "ready") {
            Write-Host ""
            Write-Host "  Ready! Opening browser..." -ForegroundColor Green
            Write-Host ""
            Open-Url $Url
            break
        }

        $spin = $Spinner[$SpinIdx % 4]
        $SpinIdx++
        $bar = "#" * [math]::Min([int]($Elapsed / 2), 40)
        Write-Host "`r  [$spin] Loading models on $Url ... $bar" -NoNewline
    } catch {
        $spin = $Spinner[$SpinIdx % 4]
        $SpinIdx++
        Write-Host "`r  [$spin] Waiting for server at $Url ... $($_.Exception.Message)        " -NoNewline
    }
}

if ($Elapsed -ge $MaxWait) {
    Write-Host ""
    Write-Host "  WARNING: Server did not become ready within ${MaxWait}s" -ForegroundColor Yellow
    Write-Host "  Opening browser anyway..."
    Open-Url $Url
}

Write-Host "  Server running at $Url"
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

try {
    $Server.WaitForExit()
} catch {}

if (-not $Server.HasExited) {
    $Server.Kill()
}
