"""
forecasting/models/nbeats.py
----------------------------
Pure-NumPy N-BEATS forecaster with polynomial basis expansion.

Reference: Oreshkin et al. (2020) — N-BEATS: Neural basis expansion
analysis for interpretable time series forecasting.
"""

import numpy as np

from .base import relu, AdamOptimizer, epoch_bar, apply_dropout, EarlyStopTracker


# ---------------------------------------------------------------------------
# N-BEATS building block
# ---------------------------------------------------------------------------

class NBeatsBlock:
    """
    Single N-BEATS block: 3-layer fully-connected stack producing
    backcast and forecast via polynomial basis projections.

    Parameters
    ----------
    in_dim    : int   Input dimensionality (lookback * n_features).
    theta_dim : int   Polynomial basis degree.
    H         : int   Forecast horizon.
    hidden    : int   Width of FC hidden layers.
    seed      : int   Random seed.
    dropout   : float Dropout rate on FC hidden activations (train only).
    """

    def __init__(self, in_dim: int, theta_dim: int, H: int,
                 hidden: int = 64, seed: int = 0,
                 dropout: float = 0.2) -> None:
        np.random.seed(seed)
        self.dropout = float(dropout)
        # FC stack
        self.W1  = np.random.randn(in_dim, hidden) * 0.05
        self.b1  = np.zeros(hidden)
        self.W2  = np.random.randn(hidden, hidden) * 0.05
        self.b2  = np.zeros(hidden)
        self.W3  = np.random.randn(hidden, hidden) * 0.05
        self.b3  = np.zeros(hidden)
        # Basis projection heads
        self.Wtb = np.random.randn(hidden, theta_dim) * 0.05
        self.Wtf = np.random.randn(hidden, theta_dim) * 0.05
        self.btb = np.zeros(theta_dim)
        self.btf = np.zeros(theta_dim)
        # Vandermonde basis matrices
        t_b      = np.linspace(-1,  0, in_dim)
        t_f      = np.linspace( 0,  1, H)
        self.Vb  = np.vstack([t_b ** k for k in range(theta_dim)]).T
        self.Vf  = np.vstack([t_f ** k for k in range(theta_dim)]).T

    def forward(self, x: np.ndarray, training: bool = False):
        """
        Parameters
        ----------
        x : np.ndarray  Shape (in_dim,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(backcast, forecast)`` both 1-D.
        """
        h  = relu(x  @ self.W1 + self.b1)
        h  = apply_dropout(h, self.dropout, training)
        h  = relu(h  @ self.W2 + self.b2)
        h  = apply_dropout(h, self.dropout, training)
        h  = relu(h  @ self.W3 + self.b3)
        h  = apply_dropout(h, self.dropout, training)
        bc = (h @ self.Wtb + self.btb) @ self.Vb.T
        fc = (h @ self.Wtf + self.btf) @ self.Vf.T
        return bc, fc


# ---------------------------------------------------------------------------
# N-BEATS stack forecaster
# ---------------------------------------------------------------------------

class NBeatsForecaster:
    """
    N-BEATS forecaster: 2 stacks × 3 blocks each.

    Training uses sparse numerical gradients on the forecast-head weights
    (``Wtf``, ``btf``) only. Epochs stop early when loss stops decreasing.

    Parameters
    ----------
    lookback       : int   Lookback window length.
    n_features     : int   Features per time step.
    horizon        : int   Forecast horizon.
    lr             : float Adam learning rate.
    epochs         : int   Max training epochs.
    seed           : int   Random seed.
    dropout        : float Dropout rate on FC hidden layers.
    early_patience : int   Stop after this many epochs with no loss decrease.
    """

    def __init__(self, lookback: int = 12, n_features: int = 1,
                 horizon: int = 12, lr: float = 5e-3,
                 epochs: int = 100, seed: int = 42,
                 dropout: float = 0.2, early_patience: int = 5) -> None:
        np.random.seed(seed)
        self.W       = lookback * n_features
        self.H       = horizon
        self.lr      = lr
        self.epochs  = epochs
        self.dropout = float(dropout)
        self.early_patience = int(early_patience)
        self.blocks  = [
            NBeatsBlock(
                self.W, 8, horizon, hidden=64,
                seed=s * 10 + b, dropout=self.dropout,
            )
            for s in range(2) for b in range(3)
        ]

    def _forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        residual = x.copy()
        forecast = np.zeros(self.H)
        for blk in self.blocks:
            bc, fc   = blk.forward(residual, training=training)
            residual  = residual - bc
            forecast  = forecast + fc
        return forecast

    def _snapshot(self) -> dict:
        state = {}
        for i, blk in enumerate(self.blocks):
            for attr in ("W1", "b1", "W2", "b2", "W3", "b3",
                         "Wtb", "Wtf", "btb", "btf"):
                state[f"b{i}_{attr}"] = getattr(blk, attr)
        return state

    def _restore(self, state: dict) -> None:
        for i, blk in enumerate(self.blocks):
            for attr in ("W1", "b1", "W2", "b2", "W3", "b3",
                         "Wtb", "Wtf", "btb", "btf"):
                key = f"b{i}_{attr}"
                if key in state:
                    setattr(blk, attr, np.copy(state[key]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NBeatsForecaster":
        """
        Parameters
        ----------
        X : np.ndarray  Shape (N, W, F).
        y : np.ndarray  Shape (N, horizon).
        """
        N   = X.shape[0]
        Xf  = X.reshape(N, -1)
        opt = AdamOptimizer(self.lr)
        eps = 1e-3
        stopper = EarlyStopTracker(patience=self.early_patience)
        bar = epoch_bar(self.epochs, "N-BEATS")

        for _ in bar:
            epoch_loss = 0.0
            for i in np.random.permutation(N):
                xi, yi = Xf[i], y[i]
                pred = self._forward(xi, training=True)
                epoch_loss += float(np.mean((pred - yi) ** 2))
                for bi, blk in enumerate(self.blocks):
                    for attr in ["Wtf", "btf"]:
                        W_   = getattr(blk, attr)
                        grad = np.zeros_like(W_)
                        n_up = min(W_.size, 40)
                        idxs = np.random.choice(W_.size, n_up, replace=False)
                        flat = W_.ravel()
                        for j in idxs:
                            orig    = flat[j]
                            flat[j] = orig + eps
                            setattr(blk, attr, flat.reshape(W_.shape))
                            pp = self._forward(xi, training=False)
                            flat[j] = orig - eps
                            setattr(blk, attr, flat.reshape(W_.shape))
                            pm = self._forward(xi, training=False)
                            flat[j] = orig
                            setattr(blk, attr, flat.reshape(W_.shape))
                            grad.ravel()[j] = (
                                np.mean((pp - yi) ** 2) - np.mean((pm - yi) ** 2)
                            ) / (2 * eps)
                        setattr(blk, attr,
                                opt.update(f"b{bi}_{attr}",
                                           getattr(blk, attr), grad))

            mean_loss = epoch_loss / max(N, 1)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{mean_loss:.4f}")
            if stopper.step(mean_loss, self._snapshot()):
                if stopper.best_state is not None:
                    self._restore(stopper.best_state)
                break
        else:
            if stopper.best_state is not None:
                self._restore(stopper.best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray  Shape (N, W, F).

        Returns
        -------
        np.ndarray  Shape (N, horizon).
        """
        Xf = X.reshape(X.shape[0], -1)
        return np.array([self._forward(xi, training=False) for xi in Xf])
