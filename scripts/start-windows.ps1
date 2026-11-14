<#
  Start Syntra AI on Windows: backend + interface, each in its own window.

    powershell -ExecutionPolicy Bypass -File scripts\start-windows.ps1

  Options: -ApiPort 8765 -WebPort 5291 -CondaEnv syntra
  First run creates the conda environment "syntra" if it does not exist
  (rasterio comes from conda-forge, which is the reliable route on Windows).
#>
param([int]$ApiPort = 8765, [int]$WebPort = 5291, [string]$CondaEnv = "syntra")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# conda is often installed without being on PATH (the Anaconda Prompt adds it,
# a plain PowerShell does not), so look for it in the usual install folders.
function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($b in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\anaconda3",
                     "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\anaconda3", "C:\ProgramData\miniconda3")) {
        foreach ($c in @("$b\Scripts\conda.exe", "$b\condabin\conda.bat")) { if (Test-Path $c) { return $c } }
    }
    return $null
}

$conda = Find-Conda

function Find-EnvPython {
    $bases = @()
    if ($conda) {
        $ErrorActionPreference = "Continue"
        $b = & $conda info --base 2>$null
        $ErrorActionPreference = "Stop"
        if ($b) { $bases += $b }
    }
    $bases += "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\anaconda3", "C:\ProgramData\anaconda3"
    foreach ($b in $bases | Where-Object { $_ }) {
        $p = "$b\envs\$CondaEnv\python.exe"
        if (Test-Path $p) { return $p }
    }
    foreach ($p in @("$env:USERPROFILE\.conda\envs\$CondaEnv\python.exe", "$env:USERPROFILE\conda-envs\$CondaEnv\python.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Test-PortFree([int]$port) {
    # Something answering on IPv4 or IPv6 localhost means taken (catches Docker/WSL too).
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { return $false }
    foreach ($h in @("127.0.0.1", "::1")) {
        $c = New-Object System.Net.Sockets.TcpClient
        try {
            if ($c.ConnectAsync($h, $port).Wait(300) -and $c.Connected) { return $false }
        } catch {} finally { $c.Close() }
    }
    return $true
}

function Get-FreePort([int]$start) {
    for ($p = $start; $p -lt $start + 50; $p++) { if (Test-PortFree $p) { return $p } }
    throw "No free port between $start and $($start + 49)."
}

$py = Find-EnvPython
if (-not $py) {
    if (-not $conda) { throw "Anaconda/Miniconda was not found. Install Miniconda, or pass -CondaEnv with an existing environment." }
    Write-Host "Creating conda environment '$CondaEnv' with $conda (first run only, a few minutes)..."
    & $conda create -n $CondaEnv -c conda-forge python=3.11 rasterio shapely pyproj -y
    if ($LASTEXITCODE -ne 0) { throw "conda create failed; see the messages above." }
    $py = Find-EnvPython
    if (-not $py) { throw "Could not find the '$CondaEnv' environment after creating it. Is Anaconda installed?" }
}
Write-Host "Python: $py"

# Windows PowerShell 5.1 turns a redirected stderr line into a terminating error
# under "Stop", so this probe runs with "Continue".
$ErrorActionPreference = "Continue"
& $py -c "import fastapi, uvicorn, multipart, rasterio, cv2, onnxruntime, yaml" 2>$null
$missing = $LASTEXITCODE -ne 0
$ErrorActionPreference = "Stop"
if ($missing) {
    Write-Host "Installing Python packages..."
    & $py -m pip install fastapi "uvicorn[standard]" python-multipart numpy opencv-python-headless pillow onnxruntime pyyaml python-dotenv
    if ($LASTEXITCODE -ne 0) { throw "pip install failed; see the messages above." }
}

$ApiPort = Get-FreePort $ApiPort
$WebPort = Get-FreePort $WebPort

$weights = "$env:USERPROFILE\Downloads\IMG-VHR\models\yolo11n-obb.onnx"
$setW = if (Test-Path $weights) { "`$env:SYNTRA_AERIAL_OBB_WEIGHTS='$weights'; " } else { "" }

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root\backend'; $setW& '$py' -m uvicorn app.main:app --port $ApiPort"

Set-Location "$root\frontend"
if (-not (Test-Path node_modules)) {
    Write-Host "Installing interface packages (first run only)..."
    npm install
}
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root\frontend'; `$env:VITE_API_PORT='$ApiPort'; npx vite --port $WebPort --strictPort"

# Open the browser only once this port really serves Syntra AI (not another app).
$url = "http://localhost:$WebPort"
$ok = $false
for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
    Start-Sleep -Seconds 1
    try { $ok = (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).Content -match "Syntra AI" } catch {}
}
if ($ok) {
    Start-Process $url
    Write-Host "Syntra AI: $url  (API on port $ApiPort)"
} else {
    Write-Host "The interface did not start on $url. Check the interface window for the error." -ForegroundColor Red
}
