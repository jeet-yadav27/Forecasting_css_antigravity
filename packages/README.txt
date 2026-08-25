Offline dependency wheels for Forecastingv2cursor
=================================================
Platform : Windows amd64
Python   : 3.14 (cp314)

On a machine WITH internet (refresh this folder):
  venv\Scripts\python.exe -m pip download -r requirements-offline.txt -d packages --prefer-binary
  venv\Scripts\python.exe -m pip download pip setuptools wheel -d packages --prefer-binary

On a machine WITHOUT internet:
  1. Copy this whole project folder (including packages\)
  2. Run install_offline.bat
     OR:
        python -m venv venv
        venv\Scripts\python.exe -m pip install --no-index --find-links=packages -r requirements-offline.txt
  3. venv\Scripts\activate
  4. python main.py

Notes
-----
- TensorFlow is NOT included (no wheel for Python 3.14). Optional Keras CNN-LSTM
  path is skipped; NumPy models still run.
- These wheels are for Python 3.14 on Windows. Other OS/Python versions need a
  fresh download on a matching machine.
