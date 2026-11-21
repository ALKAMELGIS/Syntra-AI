@echo off
rem Updates Syntra from GitHub (branch main), installs what changed, then starts it.
cd /d "%~dp0"
git checkout main || goto :fail
git pull --ff-only origin main || goto :fail
pushd backend
python -m pip install -q -r requirements.txt || echo Some Python packages did not install: see above.
popd
pushd frontend
call npm install --no-fund --no-audit || echo npm install failed: see above.
popd
call start-local.bat
exit /b 0
:fail
echo.
echo Update stopped: git could not switch to main or pull. Local changes? Run "git status" here.
pause
