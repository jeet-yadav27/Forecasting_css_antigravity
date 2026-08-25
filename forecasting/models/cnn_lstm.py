"""
forecasting/models/cnn_lstm.py
------------------------------
Pure-NumPy 1-D CNN feature extractor + LSTM memory → Dense output forecaster.
"""

import numpy as np

from .base import (
    relu, sigmoid, tanh, AdamOptimizer, epoch_bar,
    apply_dropout, EarlyStopTracker,
)


class CnnLstmForecaster:
    """
    1-D CNN feature extractor stacked before an LSTM, producing a
    multi-step forecast via a single Dense output layer.

    Architecture
    ------------
    Input  : (W, F)  — lookback window of F features
    Conv1  : kernel=3, 16 filters  → ReLU → Dropout
    Conv2  : kernel=3,  8 filters  → ReLU → Dropout
    LSTM   : hidden size 32 → Dropout
    Dense  : 32 → horizon

    Training uses analytic gradients for the Dense layer and sparse
    numerical gradients (128 random elements per step) for the LSTM
    weights. Epochs stop early when loss stops decreasing.

    Parameters
    ----------
    lookback       : int   Number of historical time steps in the input window.
    n_features     : int   Number of features per time step.
    horizon        : int   Number of future steps to forecast.
    lr             : float Adam learning rate.
    epochs         : int   Max training epochs.
    seed           : int   NumPy random seed for reproducibility.
    dropout        : float Dropout rate applied after Conv / LSTM (train only).
    early_patience : int   Stop after this many epochs with no loss decrease.
    """

    def __init__(self, lookback: int = 12, n_features: int = 1,
                 horizon: int = 12, lr: float = 1e-2,
                 epochs: int = 80, seed: int = 42,
                 dropout: float = 0.2, early_patience: int = 5) -> None:
        np.random.seed(seed)
        self.W      = lookback
        self.F      = n_features
        self.H      = horizon
        self.lr     = lr
        self.epochs = epochs
        self.dropout = float(dropout)
        self.early_patience = int(early_patience)
        self.H_lstm = 32
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Conv1: kernel=3, F_in=F, F_out=16
        self.Wc1 = np.random.randn(3, self.F, 16) * 0.1
        self.bc1 = np.zeros(16)
        # Conv2: kernel=3, F_in=16, F_out=8
        self.Wc2 = np.random.randn(3, 16, 8) * 0.1
        self.bc2 = np.zeros(8)
        # LSTM: input_size=8, hidden=32
        H, L_in    = self.H_lstm, 8
        self.Wlstm = np.random.randn(L_in + H, 4 * H) * 0.05
        self.blstm = np.zeros(4 * H)
        self.blstm[H:2 * H] = 1.0   # forget-gate bias initialised to 1
        # Dense: H -> horizon
        self.Wd = np.random.randn(H, self.H) * 0.1
        self.bd = np.zeros(self.H)

    def _snapshot(self) -> dict:
        return {
            "Wc1": self.Wc1, "bc1": self.bc1,
            "Wc2": self.Wc2, "bc2": self.bc2,
            "Wlstm": self.Wlstm, "blstm": self.blstm,
            "Wd": self.Wd, "bd": self.bd,
        }

    def _restore(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, np.copy(v))

    # ------------------------------------------------------------------
    # Forward primitives
    # ------------------------------------------------------------------

    def _conv1d(self, x: np.ndarray, W: np.ndarray,
                b: np.ndarray) -> np.ndarray:
        """Valid-padded 1-D convolution."""
        k, _, C_out = W.shape
        T           = x.shape[0]
        out         = np.zeros((T - k + 1, C_out))
        for i in range(T - k + 1):
            out[i] = x[i:i + k].reshape(-1) @ W.reshape(-1, C_out) + b
        return out

    def _lstm_step(self, x_seq: np.ndarray) -> np.ndarray:
        """Run LSTM over *x_seq* and return final hidden state."""
        H   = self.H_lstm
        h   = np.zeros(H)
        c   = np.zeros(H)
        for xt in x_seq:
            combined = np.concatenate([xt, h])
            gates    = combined @ self.Wlstm + self.blstm
            ig = sigmoid(gates[:H])
            fg = sigmoid(gates[H:2 * H])
            g  = tanh   (gates[2 * H:3 * H])
            og = sigmoid(gates[3 * H:])
            c  = fg * c + ig * g
            h  = og * tanh(c)
        return h

    def _forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        c1 = relu(self._conv1d(x, self.Wc1, self.bc1))
        c1 = apply_dropout(c1, self.dropout, training)
        c2 = relu(self._conv1d(c1, self.Wc2, self.bc2))
        c2 = apply_dropout(c2, self.dropout, training)
        h  = self._lstm_step(c2)
        h  = apply_dropout(h, self.dropout, training)
        return h @ self.Wd + self.bd

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CnnLstmForecaster":
        """
        Train on sliding-window dataset.

        Parameters
        ----------
        X : np.ndarray  Shape (N, W, F).
        y : np.ndarray  Shape (N, horizon).
        """
        opt = AdamOptimizer(self.lr)
        N   = X.shape[0]
        stopper = EarlyStopTracker(patience=self.early_patience)
        bar = epoch_bar(self.epochs, "CNN-LSTM")

        for _ in bar:
            epoch_loss = 0.0
            for i in np.random.permutation(N):
                xi, yi = X[i], y[i]
                pred   = self._forward(xi, training=True)
                epoch_loss += float(np.mean((pred - yi) ** 2))
                dL     = 2 * (pred - yi) / self.H

                # Analytic Dense gradient (eval path, no dropout noise)
                c1  = relu(self._conv1d(xi, self.Wc1, self.bc1))
                c2  = relu(self._conv1d(c1, self.Wc2, self.bc2))
                h   = self._lstm_step(c2)
                self.Wd = opt.update("Wd", self.Wd, np.outer(h, dL))
                self.bd = opt.update("bd", self.bd, dL)

                # Sparse numerical gradient for LSTM parameters
                eps = 1e-3
                for name in ["Wlstm", "blstm"]:
                    W_    = getattr(self, name)
                    grad  = np.zeros_like(W_)
                    n_upd = min(W_.size, 128)
                    idxs  = np.random.choice(W_.size, n_upd, replace=False)
                    flat  = W_.ravel()
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
