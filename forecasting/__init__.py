"""
=============================================================================
  AUTOMOTIVE WARRANTY CLAIMS FORECASTING SYSTEM
  Models: CNN-LSTM · N-BEATS · Transformer · SARIMA · Ensemble
  Output: Interactive Gradio Dashboard
=============================================================================
"""

__version__ = "1.1.0"

from forecasting.pipeline.runner import main  # noqa: F401

__all__ = ["main", "__version__"]
