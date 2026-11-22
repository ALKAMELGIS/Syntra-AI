@echo off
rem Starts Syntra on this computer only (http://localhost:5291).
cd /d "%~dp0"
set SYNTRA_PORT=8011
rem Stop any Syntra still running from before: by port, and any engine by its command line
rem (an old engine can keep answering on a shared port after its window is closed).
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010,8011,5291 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*' -and $_.CommandLine -like '*uvicorn*app.main:app*') -or ($_.Name -like 'node*' -and $_.CommandLine -like '*Syntra-AI*vite*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
start "Syntra engine" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --port %SYNTRA_PORT%"
rem An interface that could not be stopped still serves the current files (Vite reloads them): keep it.
netstat -ano | findstr /r /c:":5291 .*LISTENING" >nul
if %errorlevel%==0 (
  echo The interface is already running: http://localhost:5291
) else (
  start "Syntra interface" cmd /k "cd /d "%~dp0frontend" && set VITE_API_PORT=%SYNTRA_PORT%&& npm run dev"
)
