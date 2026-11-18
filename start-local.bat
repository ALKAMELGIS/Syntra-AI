@echo off
rem Starts Syntra on this computer only (http://localhost:5291).
cd /d "%~dp0"
rem stop a Syntra that is still running from before, so the ports are free
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010,5291 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
start "Syntra engine" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --port 8010"
start "Syntra interface" cmd /k "cd /d "%~dp0frontend" && npm run dev"
