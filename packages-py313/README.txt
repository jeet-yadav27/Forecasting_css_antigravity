Offline dependency wheels for Forecastingv2cursor
=================================================
Platform : Windows amd64
Python   : 3.13.5 (cp313)

Contents
--------
- All packages from requirements-offline.txt
- TensorFlow + Keras (available for Python 3.13)
- pip / setuptools / wheel

On a machine WITH internet (refresh this folder):
  python -m pip download -r requirements-offline.txt -d packages-py313 --prefer-binary ^
    --python-version 3.13.5 --implementation cp --abi cp313 --platform win_amd64 --only-binary=:all:
  python -m pip download "tensorflow>=2.15.0" -d packages-py313 --prefer-binary ^
    --python-version 3.13.5 --implementation cp --abi cp313 --platform win_amd64 --only-binary=:all:
  python -m pip download pip setuptools wheel -d packages-py313 --prefer-binary ^
    --python-version 3.13.5 --implementation cp --abi cp313 --platform win_amd64 --only-binary=:all:

On a machine WITHOUT internet:
  1. Install Python 3.13.5 (Windows amd64)
  2. Copy project + packages-py313\
  3. Run install_offline_py313.bat
     OR:
        py -3.13 -m venv venv
        venv\Scripts\python.exe -m pip install --no-index --find-links=packages-py313 -r requirements-offline.txt
        venv\Scripts\python.exe -m pip install --no-index --find-links=packages-py313 "tensorflow>=2.15.0"
  4. venv\Scripts\activate
  5. python main.py

Do not mix with packages\ (that folder is for Python 3.14).
