"""
forecasting/models/base.py
--------------------------
Shared activation functions and the minimal Adam optimizer used by all
pure-NumPy deep-learning models.
"""

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Clipped hyperbolic tangent."""
    return np.tanh(np.clip(x, -30, 30))


def epoch_bar(n_epochs: int, model_name: str):
    """tqdm progress bar over training epochs (visible in terminal + Gradio)."""
    return tqdm(
        range(n_epochs),
        desc=f"{model_name} epochs",
        unit="ep",
        leave=True,
        dynamic_ncols=True,
    )


def apply_dropout(x: np.ndarray, rate: float, training: bool) -> np.ndarray:
    """Inverted dropout — active only when *training* is True."""
    if (not training) or rate <= 0.0:
        return x
    keep = 1.0 - float(rate)
    mask = (np.random.rand(*x.shape) < keep).astype(x.dtype)
    return x * mask / keep


class EarlyStopTracker:
    """
    Stop training when epoch loss stops decreasing.

    If loss does not improve by *min_delta* for *patience* consecutive
    epochs, ``step`` returns True and the caller should restore
    ``best_state`` then break out of the epoch loop.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-6) -> None:
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best_loss = np.inf
        self.bad_epochs = 0
        self.best_state = None

    def step(self, loss: float, state: dict | None = None) -> bool:
        """
        Record *loss* for this epoch.

        Returns
        -------
        bool
            True → no improvement for *patience* epochs (stop training).
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = float(loss)
            self.bad_epochs = 0
            if state is not None:
                self.best_state = {k: np.copy(v) for k, v in state.items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """
    Minimal, per-key stateful Adam optimiser (Kingma & Ba, 2015).

    Each parameter tensor is identified by a string *key*; state (m, v)
    is stored in dictionaries so the same instance can manage multiple
    independent parameter tensors.

    Parameters
    ----------
    lr  : float  Learning rate (α).
    b1  : float  Exponential decay rate for 1st-moment estimate.
    b2  : float  Exponential decay rate for 2nd-moment estimate.
    eps : float  Numerical stability constant.
    """

    def __init__(self, lr: float = 1e-3, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t: int = 0
        self.m: dict = {}
        self.v: dict = {}

    def update(self, key: str, param: np.ndarray,
               grad: np.ndarray) -> np.ndarray:
        """
        Compute and return the updated parameter tensor.

        Parameters
        ----------
        key   : str        Unique name identifying this parameter.
        param : np.ndarray Current parameter value.
        grad  : np.ndarray Gradient w.r.t. *param*.

        Returns
        -------
        np.ndarray  Updated parameter.
        """
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

        self.t     += 1
        self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grad
        self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * grad ** 2

        m_hat = self.m[key] / (1 - self.b1 ** self.t)
        v_hat = self.v[key] / (1 - self.b2 ** self.t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
