<#
  Start Syntra AI on Windows: backend + interface, each in its own window.

    powershell -ExecutionPolicy Bypass -File scripts\start-windows.ps1

  Options: -ApiPort 8765 -WebPort 5180 -CondaEnv syntra
  First run creates the conda environment "syntra" if it does not exist
  (rasterio comes from conda-forge, which is the reliable route on Windows).
#>
param([int]$ApiPort = 8765, [int]$WebPort = 5180, [string]$CondaEnv = "syntra")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Find-EnvPython {
    $bases = @()
    try { $bases += (& conda info --base 2>$null) } catch {}
    $bases += "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:LOCALAPPDATA\anaconda3", "C:\ProgramData\anaconda3"
    foreach ($b in $bases | Where-Object { $_ }) {
        foreach ($p in @("$b\envs\$CondaEnv\python.exe")) { if (Test-Path $p) { return $p } }
    }
    $p = "$env:USERPROFILE\.conda\envs\$CondaEnv\python.exe"
    if (Test-Path $p) { return $p }
    return $null
}

function Test-PortFree([int]$port) {
    -not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

$py = Find-EnvPython
if (-not $py) {
    Write-Host "Creating conda environment '$CondaEnv' (first run only, a few minutes)..."
    conda create -n $CondaEnv -c conda-forge python=3.11 rasterio shapely pyproj -y
    $py = Find-EnvPython
    if (-not $py) { throw "Could not find the '$CondaEnv' environment after creating it. Is Anaconda installed?" }
}
Write-Host "Python: $py"

& $py -c "import fastapi, uvicorn, multipart, rasterio, cv2, onnxruntime, yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Python packages..."
    & $py -m pip install fastapi "uvicorn[standard]" python-multipart numpy opencv-python-headless pillow onnxruntime pyyaml python-dotenv
    if ($LASTEXITCODE -ne 0) { throw "pip install failed; see the messages above." }
}

foreach ($port in @($ApiPort, $WebPort)) {
    if (-not (Test-PortFree $port)) { throw "Port $port is already in use. Run again with another port, e.g. -ApiPort 8766 -WebPort 5181" }
}

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

Start-Sleep -Seconds 6
Start-Process "http://localhost:$WebPort"
Write-Host "Syntra AI: http://localhost:$WebPort  (API on port $ApiPort)"
