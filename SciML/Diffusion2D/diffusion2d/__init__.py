"""Finite-volume, POD and affine augmented DMD teaching implementations."""

from .problem import Problem
from .fvm import assemble_system, solve_fvm
from .pod import fit_pod, solve_pod
from .dmd import fit_dmd, predict_dmd

__all__ = ["Problem", "assemble_system", "solve_fvm", "fit_pod", "solve_pod",
           "fit_dmd", "predict_dmd"]
