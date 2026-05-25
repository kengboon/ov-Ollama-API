#requires -Version 7.0
# install.ps1 — NoLlama setup: venv, dependencies, model selection
#
# Usage:
#     .\install.ps1              # interactive setup
#     .\install.ps1 -SkipModel   # venv + deps only
#
# Detects available devices (NPU, GPU, CPU), then walks the user
# through model selection. NPU-first: if you have an NPU, that's
# your primary chat device.

param(
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsRoot = Join-Path $HOME "models"
Push-Location $ScriptDir

Write-Host ""
Write-Host "=== NoLlama Install ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Create venv
# ---------------------------------------------------------------------------

$VenvDir = Join-Path $ScriptDir "venv"

# Validate existing venv. Script launchers (pip.exe, hf.exe, ...) bake the
# absolute path to python.exe into themselves at install time. If the venv
# folder is moved or renamed, every launcher fails with "Unable to create
# process". Catch that here and recreate, rather than failing mid-install.
if (Test-Path $VenvDir) {
    $venvPip = Join-Path $VenvDir "Scripts\pip.exe"
    $venvOk = $false
    if (Test-Path $venvPip) {
        & $venvPip --version 2>&1 | Out-Null
        $venvOk = ($LASTEXITCODE -eq 0)
    }
    if ($venvOk) {
        Write-Host "[OK] venv already exists"
    } else {
        Write-Host "[!] venv at $VenvDir is broken (likely moved from another path). Recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvDir
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python venv..."
    python -m venv $VenvDir
    if (-not $?) { Write-Host "ERROR: Failed to create venv. Is Python installed?" -ForegroundColor Red; Pop-Location; exit 1 }
    Write-Host "[OK] venv created"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip wheel setuptools 2>&1 | Out-Null
python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
if (-not $?) { Write-Host "ERROR: pip install failed" -ForegroundColor Red; Pop-Location; exit 1 }
Write-Host "[OK] Dependencies installed"
Write-Host ""

# ---------------------------------------------------------------------------
# 2. Detect devices
# ---------------------------------------------------------------------------

Write-Host "Detecting devices..." -ForegroundColor Cyan
# Mirror nollama.py's detect_devices(): canonical-keyed {kind: {id, name}}.
# Filter non-Intel GPUs (NVIDIA/AMD enumerated via OpenCL are unusable —
# crash with CL_INVALID_VALUE at warmup). Normalize multi-GPU enumeration
# (GPU.0/GPU.1) to a single canonical "GPU" entry pointing at the first
# Intel GPU; "id" preserves the real OpenVINO device id for --device.
$DeviceInfo = python -c @"
import openvino as ov, json
core = ov.Core()
out = {}
for dev in core.get_available_devices():
    try: full = core.get_property(dev, 'FULL_DEVICE_NAME')
    except: full = dev
    if dev.startswith('GPU'):
        if 'intel' not in full.lower(): continue
        if 'GPU' not in out: out['GPU'] = {'id': dev, 'name': full}
    elif dev in ('NPU', 'CPU'):
        out[dev] = {'id': dev, 'name': full}
print(json.dumps(out))
"@ | ConvertFrom-Json

$HasNPU = $null -ne $DeviceInfo.NPU
$HasGPU = $null -ne $DeviceInfo.GPU

Write-Host ""
if ($HasNPU) { Write-Host "  [+] NPU: $($DeviceInfo.NPU.name)" -ForegroundColor Green }
else         { Write-Host "  [-] NPU: not found" -ForegroundColor DarkGray }
if ($HasGPU) {
    $gpuSuffix = if ($DeviceInfo.GPU.id -ne "GPU") { " [$($DeviceInfo.GPU.id)]" } else { "" }
    Write-Host "  [+] GPU$($gpuSuffix): $($DeviceInfo.GPU.name)" -ForegroundColor Green
} else {
    Write-Host "  [-] GPU: not found (non-Intel GPUs are filtered)" -ForegroundColor DarkGray
}
Write-Host "  [+] CPU: $($DeviceInfo.CPU.name)" -ForegroundColor DarkGray
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Scan existing local models in ~/models/
# ---------------------------------------------------------------------------

$LocalModels = @()
if (Test-Path $ModelsRoot) {
    $LocalModels = @(Get-ChildItem -Path $ModelsRoot -Directory | Where-Object {
        (Test-Path (Join-Path $_.FullName "openvino_language_model.bin")) -or
        (Test-Path (Join-Path $_.FullName "openvino_model.bin"))
    } | ForEach-Object {
        $vlmBin = Join-Path $_.FullName "openvino_language_model.bin"
        $llmBin = Join-Path $_.FullName "openvino_model.bin"
        $binPath = if (Test-Path $vlmBin) { $vlmBin } else { $llmBin }
        $binSize = (Get-Item $binPath).Length
        $sizeGB = [math]::Round($binSize / 1GB, 1)
        $mtype = "llm"
        $cfgPath = Join-Path $_.FullName "config.json"
        if (Test-Path $cfgPath) {
            try {
                $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
                $arch = ""
                if ($cfg.architectures -and $cfg.architectures.Count -gt 0) { $arch = $cfg.architectures[0].ToLower() }
                $mt = if ($cfg.model_type) { $cfg.model_type.ToLower() } else { "" }
                if ($arch -match "vl|vision|llava|qwen2vl|internvl|minicpm" -or $mt -match "vl|vision") {
                    $mtype = "vlm"
                }
            } catch {}
        }
        # Detect NPU compatibility: needs int4-cw quantization and reasonable size
        $npuOk = ($_.Name -match "int4-cw" -or $_.Name -match "cw-ov") -and $sizeGB -lt 10
        [PSCustomObject]@{ Name = $_.Name; Path = $_.FullName; SizeGB = $sizeGB; Type = $mtype; NpuOk = $npuOk }
    })
}

if ($LocalModels.Count -gt 0) {
    Write-Host "  Local models ($ModelsRoot):" -ForegroundColor DarkGray
    foreach ($lm in $LocalModels) {
        Write-Host "    $($lm.Name)  ($($lm.SizeGB) GB, $($lm.Type.ToUpper()))" -ForegroundColor DarkGray
    }
    Write-Host ""
}

if ($SkipModel) {
    Write-Host "Skipping model selection (-SkipModel)"
    Write-Host ""
    Write-Host "=== Install complete (no model) ===" -ForegroundColor Yellow
    Pop-Location; exit 0
}

# ---------------------------------------------------------------------------
# Helper: show a model menu and return the selection
# ---------------------------------------------------------------------------

$Registry = Get-Content (Join-Path $ScriptDir "models.json") -Raw | ConvertFrom-Json

function Show-ModelMenu {
    param(
        [string]$Title,
        [array]$RegistryModels,
        [array]$LocalModels,
        [string]$LocalLabel = "Already on disk (instant)",
        [bool]$AllowSkip = $false
    )

    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ""

    $items = @()

    # Local models first
    if ($LocalModels.Count -gt 0) {
        Write-Host "  $LocalLabel" -ForegroundColor Yellow
        foreach ($lm in $LocalModels) {
            $items += [PSCustomObject]@{
                Action = "local"; Name = $lm.Name; Path = $lm.Path
                HfId = $null; Source = $null; Weight = $null; Trust = $false
                SizeGB = $lm.SizeGB; Notes = "Already on disk"
            }
            $i = $items.Count
            Write-Host "    $i. $($lm.Name)" -NoNewline
            Write-Host "  ($($lm.SizeGB) GB)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  Already on disk" -ForegroundColor DarkGray
        }
        Write-Host ""
    }

    # Registry models — skip any already on disk
    $localNames = @($LocalModels | ForEach-Object { $_.Name.ToLower() })
    $filteredRegistry = @($RegistryModels | Where-Object {
        $repoName = ($_.hf_id -split '/')[-1].ToLower()
        $repoName -notin $localNames
    })
    if ($filteredRegistry.Count -gt 0) {
        Write-Host "  Download from HuggingFace:" -ForegroundColor Yellow
        foreach ($dm in $filteredRegistry) {
            $dlTag = if ($dm.source -eq "pre-exported") { "download" } else { "convert" }
            $items += [PSCustomObject]@{
                Action = $dm.source; Name = $dm.name; Path = $null
                HfId = $dm.hf_id; Source = $dm.source
                Weight = $dm.weight_format; Trust = $dm.trust_remote_code
                SizeGB = $dm.est_size_gb; Notes = $dm.notes
            }
            $i = $items.Count
            Write-Host "    $i. $($dm.name)" -NoNewline
            Write-Host "  (~$($dm.est_size_gb) GB, $dlTag)" -ForegroundColor DarkGray -NoNewline
            Write-Host "  $($dm.notes)" -ForegroundColor DarkGray
        }
    }

    Write-Host ""

    if ($AllowSkip) {
        $prompt = "Pick a model [1-$($items.Count)] or press Enter to skip"
    } else {
        $prompt = "Pick a model [1-$($items.Count)]"
    }

    while ($true) {
        $choice = Read-Host $prompt
        if ($AllowSkip -and [string]::IsNullOrWhiteSpace($choice)) {
            return $null
        }
        $num = 0
        if ([int]::TryParse($choice, [ref]$num) -and $num -ge 1 -and $num -le $items.Count) {
            return $items[$num - 1]
        }
        Write-Host "Enter a number between 1 and $($items.Count)" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# Helper: download or link a model into a target directory
# ---------------------------------------------------------------------------

function Test-ModelCacheValid {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    return (Test-Path (Join-Path $Path "openvino_language_model.bin")) -or
           (Test-Path (Join-Path $Path "openvino_model.bin"))
}

function New-ModelJunction {
    param([string]$TargetDir, [string]$CachePath)
    if (Test-Path $TargetDir) {
        $item = Get-Item $TargetDir
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            cmd /c rmdir "`"$TargetDir`""
        } else {
            Remove-Item -Recurse -Force $TargetDir
        }
    }
    cmd /c mklink /J "`"$TargetDir`"" "`"$CachePath`"" | Out-Null
}

function Install-Model {
    param(
        [PSCustomObject]$Selected,
        [string]$TargetDir
    )

    if ($Selected.Action -eq "local") {
        Write-Host "Linking to: $($Selected.Path)" -ForegroundColor Green
        New-ModelJunction -TargetDir $TargetDir -CachePath $Selected.Path
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    # pre-exported and convert both cache into ~/models/<name>/ first, then
    # junction $TargetDir → cache. Lets re-installs detect the existing
    # model (scan looks at ~/models/) and skip the download.
    if ($Selected.Action -eq "pre-exported") {
        $cacheName = ($Selected.HfId -split '/')[-1]
        $cachePath = Join-Path $ModelsRoot $cacheName

        if (Test-ModelCacheValid -Path $cachePath) {
            Write-Host "Using cached $($Selected.Name) at $cachePath" -ForegroundColor Green
        } else {
            if (Test-Path $cachePath) {
                Write-Host "  Found incomplete cache at $cachePath, removing." -ForegroundColor DarkGray
                Remove-Item -Recurse -Force $cachePath
            }
            New-Item -ItemType Directory -Path $ModelsRoot -Force | Out-Null
            Write-Host "Downloading $($Selected.Name)..." -ForegroundColor Cyan
            Write-Host "  From: $($Selected.HfId)"
            Write-Host "  To:   $cachePath"
            Write-Host ""
            $env:PYTHONIOENCODING = "utf-8"
            hf download $Selected.HfId --local-dir $cachePath
            if (-not $?) {
                Write-Host "ERROR: Download failed." -ForegroundColor Red
                Write-Host "  If 401/403: run 'huggingface-cli login' first" -ForegroundColor Yellow
                return $false
            }
        }

        New-ModelJunction -TargetDir $TargetDir -CachePath $cachePath
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    if ($Selected.Action -eq "convert") {
        # Include weight format in cache name so int4 and int8 conversions
        # of the same model don't collide.
        $cacheName = "$(($Selected.HfId -split '/')[-1])-$($Selected.Weight)"
        $cachePath = Join-Path $ModelsRoot $cacheName

        if (Test-ModelCacheValid -Path $cachePath) {
            Write-Host "Using cached $($Selected.Name) at $cachePath" -ForegroundColor Green
        } else {
            if (Test-Path $cachePath) {
                Write-Host "  Found incomplete cache at $cachePath, removing." -ForegroundColor DarkGray
                Remove-Item -Recurse -Force $cachePath
            }
            New-Item -ItemType Directory -Path $ModelsRoot -Force | Out-Null
            Write-Host "Converting $($Selected.Name)..." -ForegroundColor Cyan
            Write-Host "  From: $($Selected.HfId)"
            Write-Host "  To:   $cachePath"
            Write-Host "  This may take 5-20 minutes."
            Write-Host ""
            $args = @("export", "openvino", "--model", $Selected.HfId, "--weight-format", $Selected.Weight)
            if ($Selected.Trust) { $args += "--trust-remote-code" }
            $args += $cachePath
            Write-Host "Running: optimum-cli $($args -join ' ')" -ForegroundColor DarkGray
            & optimum-cli @args
            if (-not $?) {
                Write-Host "ERROR: Conversion failed." -ForegroundColor Red
                Write-Host "  If unsupported architecture: needs newer optimum-intel" -ForegroundColor Yellow
                return $false
            }
        }

        New-ModelJunction -TargetDir $TargetDir -CachePath $cachePath
        Write-Host "[OK] $($Selected.Name)" -ForegroundColor Green
        return $true
    }

    Write-Host "ERROR: Unknown action '$($Selected.Action)'" -ForegroundColor Red
    return $false
}

# ---------------------------------------------------------------------------
# 4. Model selection — NPU-first
# ---------------------------------------------------------------------------

$ModelDir = Join-Path $ScriptDir "model"
$GpuModelDir = Join-Path $ScriptDir "gpu-model"
$StartArgs = @()  # collect args for start.ps1

if ($HasNPU) {
    # --- Step 1: NPU chat model ---
    # Only show local models that are NPU-compatible (int4-cw, reasonable size)
    $npuLocal = @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.NpuOk })
    $sel = Show-ModelMenu -Title "Step 1: Chat Model (NPU)" `
        -RegistryModels $Registry.npu `
        -LocalModels $npuLocal `
        -LocalLabel "Already converted (instant)"

    $NpuSelectedName = $null
    if ($sel) {
        $NpuSelectedName = $sel.Name
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Write-Host ""; Write-Host "Model installation failed. You can re-run install.ps1 to try again." -ForegroundColor Yellow; Pop-Location; exit 1 }
        $StartArgs += @("--device", "NPU")
        Write-Host ""
    }

    # --- Step 2: GPU model (optional) ---
    if ($HasGPU) {
        Write-Host ""
        Write-Host "=== Step 2: GPU Model (optional) ===" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  You also have an Intel ARC GPU. What do you want to use it for?"
        Write-Host ""
        Write-Host "    A. Vision model  — image understanding alongside NPU chat"
        Write-Host "    B. Bigger LLM    — much smarter chat than the NPU model"
        Write-Host "    C. Skip          — NPU chat only"
        Write-Host ""
        while ($true) {
            $gpuChoice = (Read-Host "  [A/B/C]").ToUpper()
            if ($gpuChoice -in @("A", "B", "C", "")) { break }
            Write-Host "  Enter A, B, or C" -ForegroundColor Red
        }

        if ($gpuChoice -eq "A") {
            $vlmLocal = @($LocalModels | Where-Object { $_.Type -eq "vlm" })
            $sel = Show-ModelMenu -Title "GPU Vision Model" `
                -RegistryModels $Registry.gpu_vlm `
                -LocalModels $vlmLocal
            if ($sel) {
                $ok = Install-Model -Selected $sel -TargetDir $GpuModelDir
                if ($ok) { $StartArgs += @("--gpu-model-dir", "gpu-model") }
                Write-Host ""
            }
        } elseif ($gpuChoice -eq "B") {
            $llmLocal = @($LocalModels | Where-Object { $_.Type -eq "llm" -and $_.Name -ne $NpuSelectedName })
            $sel = Show-ModelMenu -Title "GPU LLM (bigger chat model)" `
                -RegistryModels $Registry.gpu_llm `
                -LocalModels $llmLocal
            if ($sel) {
                $ok = Install-Model -Selected $sel -TargetDir $GpuModelDir
                if ($ok) { $StartArgs += @("--gpu-model-dir", "gpu-model") }
                Write-Host ""
            }
        }
    }
} elseif ($HasGPU) {
    # --- No NPU, GPU only ---
    Write-Host "No NPU detected. Selecting a GPU model." -ForegroundColor Yellow
    Write-Host ""
    $allGpu = @($Registry.gpu_vlm) + @($Registry.gpu_llm)
    $sel = Show-ModelMenu -Title "GPU Model" `
        -RegistryModels $allGpu `
        -LocalModels $LocalModels
    if ($sel) {
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Pop-Location; exit 1 }
        $StartArgs += @("--device", "GPU")
        Write-Host ""
    }
} else {
    # --- No NPU, no GPU — CPU fallback ---
    Write-Host "No NPU or GPU detected. Models will run on CPU (slower)." -ForegroundColor Yellow
    Write-Host ""
    $sel = Show-ModelMenu -Title "CPU Model" `
        -RegistryModels $Registry.npu `
        -LocalModels @($LocalModels | Where-Object { $_.Type -eq "llm" })
    if ($sel) {
        $ok = Install-Model -Selected $sel -TargetDir $ModelDir
        if (-not $ok) { Pop-Location; exit 1 }
        $StartArgs += @("--device", "CPU")
        Write-Host ""
    }
}

# ---------------------------------------------------------------------------
# 5. Generate start.ps1
# ---------------------------------------------------------------------------

$StartScript = Join-Path $ScriptDir "start.ps1"
$TemplateScript = Join-Path $ScriptDir "start-template.ps1"
$ArgsStr = $StartArgs -join " "

# Generate start.ps1 — a one-liner that calls the template with the right args
$Content = "# Auto-generated by install.ps1`n"
$Content += "& '$(Join-Path $ScriptDir "start-template.ps1")' -ServerArgs '$ArgsStr'"
Set-Content -Path $StartScript -Value $Content -Encoding UTF8
Write-Host "[OK] Generated start.ps1" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=== NoLlama install complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server:"
Write-Host "  .\start.ps1"
Write-Host ""

Pop-Location
