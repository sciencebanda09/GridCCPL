"""Minimal numpy MLP with manual backprop and Adam.

Kept dependency-free on purpose: the long-term programme needs exact, fast,
CPU-only experiments (millions of steps, no framework noise in latency numbers).
"""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 clip: float = 5.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.clip = clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], lr: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr is None else lr
        lr_t = lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.clip(np.nan_to_num(g), -self.clip, self.clip)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


class MLP:
    """Tanh MLP with an optional linear output head."""

    def __init__(self, dims: list[int], seed: int = 0, out_scale: float = 1.0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            s = np.sqrt(2.0 / dims[i])
            self.W.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float64))
            self.b.append(np.zeros(dims[i + 1]))
        self.W[-1] *= out_scale
        self.params = [x for pair in zip(self.W, self.b) for x in pair]
        self._cache: list | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        acts = [h]
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            acts.append(h)
        out = h @ self.W[-1] + self.b[-1]
        self._cache = acts
        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = self._cache
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (1.0 - acts[i] ** 2)
        grads = [x for pair in zip(gW, gb) for x in pair]
        return d, grads

    # convenience ------------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q


def batch_gradient_step(net: MLP, X: np.ndarray, Y: np.ndarray,
                        optim: Adam, loss: str = "mse") -> float:
    pred = net.forward(X).ravel()
    err = pred - Y
    loss_val = float(np.mean(err ** 2))
    dout = (2.0 * err / len(err))[:, None]
    _, grads = net.backward(dout)
    optim.step(grads)
    return loss_val
