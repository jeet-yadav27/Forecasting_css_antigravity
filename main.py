"""
main.py — Entry point for the Automotive Warranty Claims Forecasting System.

Usage
-----
    # Activate the virtual environment first:
    #   Windows: .\\venv\\Scripts\\Activate.ps1

    # Opens Gradio immediately — NO training until you upload data:
    python main.py

    # Then in the UI:
    #   1. Upload CSV/Excel
    #   2. Select Part Name + enter monthly Production / Cost
    #   3. Optional countermeasure month
    #   4. Click Train & Forecast
    # Best hyperparameters are locked under outputs/best_params.json

    python main.py --horizon 6
    python main.py --port 8080 --no-browser
    python main.py --share

Output
------
    Gradio web app on http://localhost:<port>  (default 7860)
"""

import argparse
import sys


def _ensure_pptx() -> None:
    """Fail fast with an install hint if python-pptx is missing."""
    try:
        import pptx  # noqa: F401
    except ModuleNotFoundError:
        print(
            "\n[ERROR] Missing dependency: python-pptx\n"
            f"  Python in use: {sys.executable}\n"
            "  Install with:\n"
            f"    \"{sys.executable}\" -m pip install python-pptx\n"
            "  Or activate the project venv and install requirements:\n"
            "    .\\venv\\Scripts\\Activate.ps1\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)


import forecasting.config as cfg
from forecasting.pipeline.runner import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Automotive Warranty Claims Forecasting — Upload-first Gradio UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python main.py
  python main.py --horizon 6
  python main.py --port 8080 --no-browser
  python main.py --share
""",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=cfg.FORECAST_HORIZON,
        metavar="N",
        help=f"Forecast horizon in months (default: {cfg.FORECAST_HORIZON})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=cfg.LOOKBACK_WINDOW,
        metavar="N",
        help=f"Sliding-window lookback in months (default: {cfg.LOOKBACK_WINDOW})",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        metavar="PART",
        default=None,
        help="Optional filter: only show these part names after upload.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        metavar="N",
        help="Local port for the Gradio server (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share URL",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        dest="no_browser",
        help="Do not automatically open a browser tab on startup",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _ensure_pptx()
    args = _parse_args()
    cfg.FORECAST_HORIZON = args.horizon
    cfg.LOOKBACK_WINDOW = args.lookback

    main(
        parts_filter=args.parts,
        port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )
