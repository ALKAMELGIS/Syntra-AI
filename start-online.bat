@echo off
rem Starts Syntra and opens it through a Cloudflare link.
rem Three windows open: the engine (port 8010), the interface (port 5291) and
rem the link. Keep them open; close them to stop. The link is printed in the
rem "Syntra link" window as https://....trycloudflare.com
cd /d "%~dp0"
rem stop a Syntra that is still running from before, so the ports are free
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8010,5291 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
start "Syntra engine" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --port 8010"
start "Syntra interface" cmd /k "cd /d "%~dp0frontend" && npm run dev"
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared is not installed. Run:  winget install --id Cloudflare.cloudflared
  echo then open a new window and run this file again.
  pause
  exit /b 1
)
timeout /t 8 /nobreak >nul
start "Syntra link" cmd /k "cloudflared tunnel --url http://localhost:5291"
