@echo off
setlocal EnableDelayedExpansion
REM Offline install for Python 3.13.5 (Windows amd64) from packages-py313\
REM Wheels may live in packages-py313\ or in part1\ part2\ part3\ (GitHub Release zips).
set ROOT=%~dp0
cd /d "%ROOT%"

if not exist "packages-py313\" (
  echo ERROR: packages-py313\ folder not found.
  echo Download all three Release zips (part1, part2, part3) and extract them into packages-py313\
  exit /b 1
)

set FINDLINKS=--find-links="%ROOT%packages-py313"
for /d %%D in ("%ROOT%packages-py313\part*") do (
  set FINDLINKS=!FINDLINKS! --find-links="%%D"
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
venv\Scripts\python.exe -m pip install --no-index %FINDLINKS% --upgrade pip setuptools wheel

echo Installing requirements-offline.txt ...
venv\Scripts\python.exe -m pip install --no-index %FINDLINKS% -r requirements-offline.txt
if errorlevel 1 (
  echo Install failed. Make sure you extracted ALL three part zips into packages-py313\
  exit /b 1
)

echo Installing TensorFlow (optional Keras CNN-LSTM path)...
venv\Scripts\python.exe -m pip install --no-index %FINDLINKS% "tensorflow>=2.15.0"
if errorlevel 1 (
  echo WARNING: TensorFlow install failed; NumPy models still work.
  echo If you skipped a Release zip, TensorFlow is usually in the largest part.
)

echo.
echo Done. Activate and run:
echo   venv\Scripts\activate
echo   python main.py
exit /b 0
