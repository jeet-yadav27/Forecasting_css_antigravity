"""
forecasting/models/transformer.py
----------------------------------
Pure-NumPy multi-head self-attention Transformer forecaster.
"""

import numpy as np

from .base import relu, AdamOptimizer, epoch_bar, apply_dropout, EarlyStopTracker


class TransformerForecaster:
    """
    Lightweight Transformer encoder: positional embedding → 2 × multi-head
    self-attention + FFN → mean pool → Dense forecast.

    Training uses analytic gradients for the Dense layer and sparse
    numerical gradients (50 random elements per step) for key matrices.
    Epochs stop early when loss stops decreasing; dropout regularises
    attention / FFN activations during training.

    Parameters
    ----------
    lookback       : int   Lookback window length.
    n_features     : int   Features per time step.
    horizon        : int   Forecast horizon.
    d_model        : int   Embedding / model dimension.
    n_heads        : int   Number of attention heads (d_model must be divisible).
    lr             : float Adam learning rate.
    epochs         : int   Max training epochs.
    seed           : int   Random seed.
    dropout        : float Dropout rate (train only).
    early_patience : int   Stop after this many epochs with no loss decrease.
    """

    def __init__(self, lookback: int = 12, n_features: int = 1,
                 horizon: int = 12, d_model: int = 16, n_heads: int = 2,
                 lr: float = 3e-3, epochs: int = 80,
                 seed: int = 42, dropout: float = 0.2,
                 early_patience: int = 5) -> None:
        np.random.seed(seed)
        self.W  = lookback
        self.F  = n_features
        self.H  = horizon
        self.dm = d_model
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.lr = lr
        self.epochs = epochs
        self.dropout = float(dropout)
        self.early_patience = int(early_patience)
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        dm = self.dm

        # Input projection + sinusoidal positional encoding
        self.We = np.random.randn(self.F, dm) * 0.1
        self.be = np.zeros(dm)

        pos       = np.arange(self.W)[:, None]
        div       = np.exp(np.arange(0, dm, 2) * (-np.log(10_000) / dm))
        self.PE   = np.zeros((self.W, dm))
        self.PE[:, 0::2] = np.sin(pos * div)
        cols_cos  = np.arange(1, dm, 2)
        self.PE[:, cols_cos] = np.cos(pos * div[:len(cols_cos)])

        # Multi-head attention weights
        self.Wq = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wk = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wv = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wo = np.random.randn(self.nh * self.dh, dm) * 0.05

        # Feed-forward network
        self.Wff1 = np.random.randn(dm, 32) * 0.05
        self.bff1 = np.zeros(32)
        self.Wff2 = np.random.randn(32, dm) * 0.05
        self.bff2 = np.zeros(dm)

        # Output Dense
        self.Wd = np.random.randn(dm, self.H) * 0.1
        self.bd = np.zeros(self.H)

    def _snapshot(self) -> dict:
        return {
            "We": self.We, "be": self.be,
            "Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv, "Wo": self.Wo,
            "Wff1": self.Wff1, "bff1": self.bff1,
            "Wff2": self.Wff2, "bff2": self.bff2,
            "Wd": self.Wd, "bd": self.bd,
        }

    def _restore(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, np.copy(v))

    # ------------------------------------------------------------------
    # Forward primitives
    # ------------------------------------------------------------------

    def _attn(self, x: np.ndarray) -> np.ndarray:
        """Multi-head scaled dot-product self-attention."""
        heads = []
        for h in range(self.nh):
            Q  = x @ self.Wq[h]
            K  = x @ self.Wk[h]
            V  = x @ self.Wv[h]
            sc = Q @ K.T / np.sqrt(self.dh)
            sc -= sc.max(axis=-1, keepdims=True)
            A  = np.exp(sc) / (np.exp(sc).sum(axis=-1, keepdims=True) + 1e-9)
            heads.append(A @ V)
        return np.concatenate(heads, axis=-1) @ self.Wo

    def _forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray  Shape (W, F).

        Returns
        -------
        np.ndarray  Shape (horizon,).
        """
        e  = x @ self.We + self.be + self.PE
        e  = e + apply_dropout(self._attn(e), self.dropout, training)
        ff = relu(e @ self.Wff1 + self.bff1)
        ff = apply_dropout(ff, self.dropout, training)
        ff = ff @ self.Wff2 + self.bff2
        e  = e + apply_dropout(ff, self.dropout, training)
        e  = e + apply_dropout(self._attn(e), self.dropout, training)
        return e.mean(axis=0) @ self.Wd + self.bd

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerForecaster":
        """
        Parameters
        ----------
        X : np.ndarray  Shape (N, W, F).
        y : np.ndarray  Shape (N, horizon).
        """
        opt = AdamOptimizer(self.lr)
        N   = X.shape[0]
        eps = 1e-3
        stopper = EarlyStopTracker(patience=self.early_patience)
        bar = epoch_bar(self.epochs, "Transformer")

        for _ in bar:
            epoch_loss = 0.0
            for i in np.random.permutation(N):
                xi, yi = X[i], y[i]
                pred   = self._forward(xi, training=True)
                epoch_loss += float(np.mean((pred - yi) ** 2))
                dL     = 2 * (pred - yi) / self.H

                # Analytic Dense gradient (approximate via mean-pool)
                pooled  = (xi @ self.We + self.be + self.PE).mean(axis=0)
                self.Wd = opt.update("Wd", self.Wd, np.outer(pooled, dL))
                self.bd = opt.update("bd", self.bd, dL)

                # Sparse numerical gradients for key matrices
                for name in ["Wo", "Wff2", "We"]:
                    W_   = getattr(self, name)
                    grad = np.zeros_like(W_)
                    n_up = min(W_.size, 50)
                    idxs = np.random.choice(W_.size, n_up, replace=False)
                    flat = W_.ravel()
                    for j in idxs:
                        orig    = flat[j]
                        flat[j] = orig + eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pp = self._forward(xi, training=False)
                        flat[j] = orig - eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pm = self._forward(xi, training=False)
                        flat[j] = orig
                        setattr(self, name, flat.reshape(W_.shape))
                        grad.ravel()[j] = (
                            np.mean((pp - yi) ** 2) - np.mean((pm - yi) ** 2)
                        ) / (2 * eps)
                    setattr(self, name,
                            opt.update(name, getattr(self, name), grad))

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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray  Shape (N, W, F).

        Returns
        -------
        np.ndarray  Shape (N, horizon).
        """
        return np.array([self._forward(xi, training=False) for xi in X])
