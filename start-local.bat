@echo off
rem Starts Syntra on this computer only (http://localhost:5291).
cd /d "%~dp0"
set SYNTRA_PORT=8011
rem Stop any Syntra still running from before: by port, and any engine by its command line
rem (an old engine can keep answering on a shared port after its window is closed).
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010,8011,5291 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*' -and $_.CommandLine -like '*uvicorn*app.main:app*') -or ($_.Name -like 'node*' -and $_.CommandLine -like '*Syntra-AI*vite*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
rem And by PID from netstat, with its child processes (npm > node > esbuild), in case PowerShell could not.
for %%P in (8010 8011 5291) do for /f "tokens=5" %%I in ('netstat -ano ^| findstr /r /c:":%%P .*LISTENING"') do taskkill /F /T /PID %%I >nul 2>&1
timeout /t 2 /nobreak >nul
start "Syntra engine" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --port %SYNTRA_PORT%"
rem The interface no longer reloads itself, so an old one would keep serving old files: always start a fresh one.
netstat -ano | findstr /r /c:":5291 .*LISTENING" >nul
if %errorlevel%==0 (
  echo Port 5291 is still held by another program. Close every "Syntra interface" window, then run start-local.bat again.
  pause
  exit /b 1
)
rem A new library in the interface (package-lock.json newer than the installed set): install it first.
powershell -NoProfile -Command "$l='frontend\package-lock.json'; $m='frontend\node_modules\.package-lock.json'; if (!(Test-Path $m) -or (Get-Item $l).LastWriteTime -gt (Get-Item $m).LastWriteTime) { exit 1 }"
if errorlevel 1 (
  echo Installing the interface's libraries...
  pushd frontend && call npm install --no-audit --no-fund && popd
)
start "Syntra interface" cmd /k "cd /d "%~dp0frontend" && set VITE_API_PORT=%SYNTRA_PORT%&& npm run dev"
