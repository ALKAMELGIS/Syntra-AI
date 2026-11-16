@echo off
rem Starts Syntra on this computer only (http://localhost:5291).
cd /d "%~dp0"
start "Syntra engine" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --port 8010"
start "Syntra interface" cmd /k "cd /d "%~dp0frontend" && npm run dev"
