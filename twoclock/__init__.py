"""Two-Clock Dispatch — shadow-price control for grid dispatch.

Stage 1 (existence):  learn the price of a slow physical constraint
Stage 2 (reduction):  does a scalar price recover the optimal controller?
Stage 3 (discovery):  recover unpriced-constraint prices from market data

The package is deliberately dependency-light: numpy + scipy only.
"""

__version__ = "0.1.0"
ENV_VERSION = "twoclock-env-1"

__all__ = ["ENV_VERSION"]
