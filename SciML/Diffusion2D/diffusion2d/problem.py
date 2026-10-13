"""Cell-centred discretization of the homework's unit-square problem."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    """Velocity is fixed at (1, 0); array layout is (ny, nx), x fastest."""

    nx: int = 40
    ny: int = 40
    diffusion: float = 0.01

    def __post_init__(self):
        for size in (self.nx, self.ny):
            if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 2:
                raise ValueError("nx and ny must be integers >= 2")
        if not np.isfinite(self.diffusion) or self.diffusion < 0:
            raise ValueError("diffusion must be finite and nonnegative")

    @property
    def dx(self):
        return 1.0 / self.nx

    @property
    def dy(self):
        return 1.0 / self.ny

    @property
    def ndof(self):
        return self.nx * self.ny

    @property
    def x(self):
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self):
        return (np.arange(self.ny) + 0.5) * self.dy

    def inlet_values(self):
        """Strict band inequality evaluated at boundary-face centres."""
        return ((self.y > 1 / 3) & (self.y < 2 / 3)).astype(float)
