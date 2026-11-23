"""DC power flow on a small meshed network.

Conventions
-----------
* ``theta`` is the vector of voltage angles (radians), reduced modulo a shift.
* Line flow (MW, positive i -> j) is ``f_l = (theta_i - theta_j) / x_l``.
* Injections are MW: positive = generation into the bus.
* All impedances are in per-unit on a 100 MVA base; because the flow relation is
  linear, per-unit reactance absorbs the base and flows come out in MW directly
  when injections are in MW.  (We keep the 1/x coefficient in "MW per radian".)

There is intentionally no AC/voltage model here: at a 15-minute dispatch step the
physically correct statement is a *balance constraint*, not a swing equation.
That modelling decision is discussed in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Line:
    i: int
    j: int
    x: float
    limit_mw: float

    @property
    def b(self) -> float:
        """Series susceptance, MW per radian."""
        return 1.0 / self.x


@dataclass
class DCNetwork:
    n_bus: int
    lines: Sequence[Line]
    slack: int = 0
    bus_names: Sequence[str] = field(default_factory=tuple)

    # cached structure -----------------------------------------------------
    _B: np.ndarray = field(init=False, repr=False)
    _B_pinv: np.ndarray = field(init=False, repr=False)
    _A_inc: np.ndarray = field(init=False, repr=False)   # (n_bus, n_line)

    def __post_init__(self) -> None:
        self.lines = list(self.lines)
        B = np.zeros((self.n_bus, self.n_bus))
        A = np.zeros((self.n_bus, len(self.lines)))
        for k, ln in enumerate(self.lines):
            b = ln.b
            B[ln.i, ln.i] += b
            B[ln.j, ln.j] += b
            B[ln.i, ln.j] -= b
            B[ln.j, ln.i] -= b
            A[ln.i, k] += 1.0     # flow leaves i
            A[ln.j, k] -= 1.0     # flow enters j
        self._B = B
        self._B_pinv = np.linalg.pinv(B)
        self._A_inc = A

    # ------------------------------------------------------------------ props
    @property
    def n_line(self) -> int:
        return len(self.lines)

    @property
    def limits(self) -> np.ndarray:
        return np.array([ln.limit_mw for ln in self.lines], dtype=float)

    @property
    def incidence(self) -> np.ndarray:
        """(n_bus, n_line) incidence; ``A[b, l] = +1`` if line l leaves b."""
        return self._A_inc

    # ------------------------------------------------------------------ solve
    def theta(self, p_inj: np.ndarray) -> np.ndarray:
        """Angles consistent with an injection vector (min-norm solution)."""
        return self._B_pinv @ np.asarray(p_inj, dtype=float)

    def flows(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        return np.array([(th[ln.i] - th[ln.j]) / ln.x for ln in self.lines])

    def flows_from_injection(self, p_inj: np.ndarray) -> np.ndarray:
        return self.flows(self.theta(p_inj))

    def overload(self, flows: np.ndarray) -> np.ndarray:
        """Positive part of line loading beyond limits (MW)."""
        return np.maximum(np.abs(flows) - self.limits, 0.0)

    def describe(self) -> str:
        rows = [f"DCNetwork  buses={self.n_bus}  lines={self.n_line}  slack={self.slack}"]
        for k, ln in enumerate(self.lines):
            rows.append(f"  L{k}: bus{ln.i} -> bus{ln.j}  x={ln.x:<5} limit={ln.limit_mw:6.0f} MW")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# A canonical 5-bus test system used throughout the package
# ---------------------------------------------------------------------------

GENERATOR_NAMES = ("gas", "coal", "nuclear", "renewable")


def build_5bus() -> DCNetwork:
    """Meshed 5-bus system.

    bus 0  gas (slack)      bus 1  coal
    bus 2  nuclear          bus 3  renewable
    bus 4  battery + load centre
    """
    lines = [
        Line(0, 1, x=0.050, limit_mw=350.0),
        Line(1, 2, x=0.060, limit_mw=300.0),
        Line(2, 3, x=0.080, limit_mw=250.0),
        Line(3, 4, x=0.050, limit_mw=300.0),
        Line(4, 0, x=0.070, limit_mw=300.0),
        Line(1, 3, x=0.100, limit_mw=200.0),
    ]
    return DCNetwork(
        n_bus=5,
        lines=lines,
        slack=0,
        bus_names=("gas", "coal", "nuclear", "renewable", "centre"),
    )


# Load sharing across buses (fractions of system load)
LOAD_SHARE = np.array([0.15, 0.25, 0.20, 0.15, 0.25])

# Generator fleet: bus, pmin, pmax, ramp MW/step, fuel $/MWh, thermal rating MW
GEN_BUS = np.array([0, 1, 2, 3])
GEN_PMIN = np.array([0.0, 0.0, 170.0, 0.0])
GEN_PMAX = np.array([350.0, 300.0, 200.0, 150.0])
GEN_RAMP = np.array([80.0, 30.0, 5.0, 0.0])
GEN_COST = np.array([65.0, 45.0, 20.0, 5.0])
GEN_RATING = np.array([350.0, 300.0, 200.0, 150.0])
GEN_DISPATCHABLE = np.array([True, True, True, False])

BATTERY_BUS = 4
BATTERY_POWER_MW = 100.0
BATTERY_CAPACITY_MWH = 400.0
IMPORT_BUS = 4
IMPORT_CAP_MW = 100.0
