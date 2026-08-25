@echo off
REM Install project dependencies from local packages\ folder (no internet).
REM Requires Python 3.14 (Windows amd64) matching the downloaded wheels.

set ROOT=%~dp0
cd /d "%ROOT%"

if not exist "packages\" (
  echo ERROR: packages\ folder not found.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo Creating venv...
  python -m venv venv
)

echo Upgrading pip/setuptools/wheel from packages\ ...
venv\Scripts\python.exe -m pip install --no-index --find-links=packages --upgrade pip setuptools wheel
if errorlevel 1 (
  echo WARNING: could not upgrade pip from packages; continuing...
)

echo Installing from packages\ (offline)...
venv\Scripts\python.exe -m pip install --no-index --find-links=packages -r requirements-offline.txt
if errorlevel 1 (
  echo Install failed.
  exit /b 1
)

echo.
echo Done. Activate and run:
echo   venv\Scripts\activate
echo   python main.py
exit /b 0
