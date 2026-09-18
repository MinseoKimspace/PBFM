"""Compiled sequential sparse PGS for single-world stress comparisons.

The update order/formula and projected KKT stopping criterion match pgs().
Only zero entries of the fixed D matrix are omitted. No coloring, Jacobi,
relaxation, warm starts or altered contact constraints are introduced.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _sparse_pgs(indptr, columns, values, column_ptr, rows, column_values,
                diagonal, c, eta, start, tolerance, max_sweeps):
    lam = start.copy()
    g = c.copy()
    sweeps = 0
    zero = c[0]-c[0]  # Keep float32 arithmetic float32 (avoid integer promotion).
    while True:
        # Recompute residuals each sweep, as in the original dense PGS.
        for i in range(len(c)):
            total = c[i]
            for k in range(indptr[i], indptr[i+1]):
                total += values[k]*lam[columns[k]]
            g[i] = total
        solved = True
        for i in range(len(c)):
            projected = max(lam[i]-eta*g[i], zero)
            if (not np.isfinite(lam[i]) or not np.isfinite(g[i])
                    or abs((lam[i]-projected)/eta) > tolerance
                    or -g[i] > tolerance or -lam[i] > tolerance):
                solved = False
                break
        if solved or sweeps == max_sweeps:
            return lam, sweeps, solved
        for i in range(len(c)):
            increment = max(lam[i]-g[i]/diagonal[i], zero)-lam[i]
            lam[i] += increment
            # Keep actual columns: floating-point D may not be bitwise symmetric.
            for k in range(column_ptr[i], column_ptr[i+1]):
                g[rows[k]] += column_values[k]*increment
        sweeps += 1


class SparsePGS:
    """QP transfer/CSR construction is setup; state transfers are solve work."""

    def __init__(self, problem):
        try:
            from numba import njit
        except ImportError as error:
            raise RuntimeError("Compiled stress PGS needs numba: pip install -r requirements.txt; or use --pgs-backend torch") from error
        if len(problem["c"]) != 1:
            raise ValueError("Sparse stress PGS expects exactly one world")
        self.count = int(problem["mask"].sum())
        self.device, self.dtype = problem["c"].device, problem["c"].dtype
        matrix = problem["D"][0, :self.count, :self.count].cpu().numpy()
        rows, columns = np.nonzero(matrix)
        self.columns = np.ascontiguousarray(columns, dtype=np.int64)
        self.indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=self.count)))).astype(np.int64)
        self.values = np.ascontiguousarray(matrix[rows, columns])
        order = np.lexsort((rows, columns))
        self.column_ptr = np.concatenate(([0], np.cumsum(np.bincount(columns, minlength=self.count)))).astype(np.int64)
        self.rows = np.ascontiguousarray(rows[order], dtype=np.int64)
        self.column_values = np.ascontiguousarray(self.values[order])
        self.diagonal = np.ascontiguousarray(matrix.diagonal())
        self.c = problem["c"][0, :self.count].cpu().numpy().copy()
        self.eta = problem["eta"][0, 0].cpu().numpy()[()]
        self.problem_shape = problem["c"].shape
        # One lazy compiled dispatcher is shared across QPs; no disk cache files.
        global _compiled
        if _compiled is None:
            _compiled = njit(_sparse_pgs, fastmath=False)

    def solve(self, start, tolerance, max_sweeps):
        if not math.isfinite(tolerance) or tolerance <= 0 or type(max_sweeps) is not int or max_sweeps < 1:
            raise ValueError("Positive tolerance and integer sweep budget required")
        if start.shape != self.problem_shape:
            raise ValueError("Start shape differs from the prepared QP")
        if self.count:
            value = start[0, :self.count].detach().cpu().numpy()
            solution, sweeps, solved = _compiled(self.indptr, self.columns, self.values,
                self.column_ptr, self.rows, self.column_values,
                self.diagonal, self.c, self.eta, value, tolerance, max_sweeps)
            final = torch.from_numpy(solution).to(device=self.device, dtype=self.dtype)[None]
        else:
            final, sweeps, solved = torch.zeros_like(start), 0, True
        return dict(final=final, sweeps=torch.tensor([sweeps], device=self.device),
                    contact_evals=torch.tensor([sweeps*self.count], device=self.device),
                    converged=torch.tensor([solved], device=self.device))


_compiled = None
