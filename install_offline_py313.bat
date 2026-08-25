@echo off
REM Offline install for Python 3.13.5 (Windows amd64) from packages-py313\
set ROOT=%~dp0
cd /d "%ROOT%"

if not exist "packages-py313\" (
  echo ERROR: packages-py313\ folder not found.
  exit /b 1
)

where py >nul 2>&1
if %errorlevel%==0 (
  py -3.13 --version >nul 2>&1
  if %errorlevel%==0 (
    set PYLAUNCH=py -3.13
  ) else (
    set PYLAUNCH=python
  )
) else (
  set PYLAUNCH=python
)

echo Using: %PYLAUNCH%
%PYLAUNCH% --version
if errorlevel 1 (
  echo ERROR: Python 3.13 not found. Install Python 3.13.5 first.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo Creating venv with Python 3.13...
  %PYLAUNCH% -m venv venv
)

echo Upgrading pip/setuptools/wheel from packages-py313\ ...
venv\Scripts\python.exe -m pip install --no-index --find-links=packages-py313 --upgrade pip setuptools wheel

echo Installing requirements-offline.txt ...
venv\Scripts\python.exe -m pip install --no-index --find-links=packages-py313 -r requirements-offline.txt
if errorlevel 1 (
  echo Install failed.
  exit /b 1
)

echo Installing TensorFlow (optional Keras CNN-LSTM path)...
venv\Scripts\python.exe -m pip install --no-index --find-links=packages-py313 "tensorflow>=2.15.0"
if errorlevel 1 (
  echo WARNING: TensorFlow install failed; NumPy models still work.
)

echo.
echo Done. Activate and run:
echo   venv\Scripts\activate
echo   python main.py
exit /b 0
