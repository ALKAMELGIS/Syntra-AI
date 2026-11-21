@echo off
rem Builds the interface and uploads Syntra to its Hugging Face Space (Alkamelgis/SYNTRA-AI).
cd /d "%~dp0frontend"
call npm run build || goto :fail
cd /d "%~dp0"
python scripts\deploy_space.py || goto :fail
pause
exit /b 0
:fail
echo.
echo Deploy stopped: see the message above.
pause
