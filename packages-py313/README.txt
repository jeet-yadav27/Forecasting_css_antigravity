Offline dependency wheels for Forecasting (Python 3.13)
=======================================================
Platform : Windows amd64
Python   : 3.13.x (cp313)

Download (GitHub Release)
-------------------------
Wheels are split into THREE zips so they are easier to download:

  packages-py313-part1.zip
  packages-py313-part2.zip
  packages-py313-part3.zip

1. Open the repo Releases page and download all three zips.
2. Extract each zip into this folder (packages-py313\).
   You should end up with:

     packages-py313\part1\   (wheels + MANIFEST.txt)
     packages-py313\part2\
     packages-py313\part3\

3. From the project root run:  install_offline_py313.bat

pip --find-links searches part1, part2, and part3 automatically.
You need ALL three parts; splitting is only for download size.

On a machine WITH internet (refresh wheels, then re-split):
  python -m pip download -r requirements-offline.txt -d packages-py313 --prefer-binary ^
    --python-version 3.13 --implementation cp --abi cp313 --platform win_amd64 --only-binary=:all:
  python -m pip download "tensorflow>=2.15.0" pip setuptools wheel -d packages-py313 --prefer-binary ^
    --python-version 3.13 --implementation cp --abi cp313 --platform win_amd64 --only-binary=:all:
  python scripts\split_offline_py313.py

Do not mix with packages\ (that folder is for Python 3.14).
