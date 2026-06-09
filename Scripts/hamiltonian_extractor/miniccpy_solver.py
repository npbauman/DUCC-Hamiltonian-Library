"""
miniccpy solver for correlated quantum chemistry calculations.

Provides routines to run correlated methods (MP2, CCSD, CCSDT, etc.) using
the miniccpy library, driven by the Fock (F) and antisymmetrized two-electron
integrals (G) extracted from a HamiltonianData object.

Author: ExaChem Development Team
Note: Requires miniccpy to be installed/available on the Python path.
"""

import sys
import types
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Pure-numpy shims injected into sys.modules before any miniccpy method module
# is loaded, so that h5py (diis) and pyscf (printing) are never required.
# ──────────────────────────────────────────────────────────────────────────────

def _make_diis_shim():
    """Return a module that replaces miniccpy.diis without needing h5py.
    Only the in-memory path (out_of_core=False) is supported, which is the
    default used by run_cc_calc."""
    mod = types.ModuleType("miniccpy.diis")

    def _solve_gauss(A, b):
        n = A.shape[0]
        for i in range(n - 1):
            for j in range(i + 1, n):
                m = A[j, i] / A[i, i]
                A[j, :] -= m * A[i, :]
                b[j] -= m * b[i]
        x = np.zeros(n)
        k = n - 1
        x[k] = b[k] / A[k, k]
        while k >= 0:
            x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]
            k -= 1
        return x

    class DIIS:
        def __init__(self, ndim, diis_size, out_of_core,
                     vecfile="t.npy", residfile="dt.npy"):
            if out_of_core:
                raise RuntimeError(
                    "miniccpy_solver: out_of_core DIIS requires h5py which is "
                    "not available in this environment."
                )
            self.diis_size = diis_size
            self.ndim = ndim
            self.T_list = np.zeros((diis_size, ndim))
            self.T_residuum_list = np.zeros((diis_size, ndim))

        def cleanup(self):
            pass

        def push(self, T_tuple, T_residuum_tuple, iteration):
            idx = iteration % self.diis_size
            self.T_list[idx, :] = np.hstack([t.flatten() for t in T_tuple])
            self.T_residuum_list[idx, :] = np.hstack(
                [dt.flatten() for dt in T_residuum_tuple]
            )

        def extrapolate(self, niter=None):
            m = self.diis_size if niter is None else min(self.diis_size, niter)
            B_dim = m + 1
            B = -1.0 * np.ones((B_dim, B_dim))
            for i in range(m):
                for j in range(i, m):
                    B[i, j] = np.dot(
                        self.T_residuum_list[i, :], self.T_residuum_list[j, :]
                    )
                    B[j, i] = B[i, j]
            B[-1, -1] = 0.0
            rhs = np.zeros(B_dim)
            rhs[-1] = -1.0
            coeff = _solve_gauss(B, rhs)
            x = np.zeros(self.ndim)
            for i in range(m):
                x += coeff[i] * self.T_list[i, :]
            return x

    mod.DIIS = DIIS
    mod.solve_gauss = _solve_gauss
    return mod


def _make_printing_shim():
    """Return a module that replaces miniccpy.printing without needing pyscf.
    Only print_amplitudes is called by run_cc_calc; everything else is a no-op."""
    mod = types.ModuleType("miniccpy.printing")
    mod.WHITESPACE = "  "

    def print_amplitudes(t1, t2, print_threshold, rhf=False):
        nu, no = t1.shape
        spin_label = lambda p: ("a" if p % 2 == 1 else "b")
        spatial_index = lambda p: (p + 1) // 2
        n = 1
        print("          i -> a")
        for a in range(nu):
            for i in range(no):
                if abs(t1[a, i]) <= print_threshold:
                    continue
                if rhf:
                    print(f"     [{n}]  {i+1} -> {a+no+1}    {t1[a, i]}")
                else:
                    print(
                        f"     [{n}]  {spatial_index(i+1)}{spin_label(i+1)}"
                        f" -> {spatial_index(a+no+1)}{spin_label(a+no+1)}"
                        f"    {t1[a, i]}"
                    )
                n += 1
        print("          i j -> a b")
        for a in range(nu):
            b_range = range(nu) if rhf else range(a + 1, nu)
            for b in b_range:
                for i in range(no):
                    j_range = range(i, no) if rhf else range(i + 1, no)
                    for j in j_range:
                        if abs(t2[a, b, i, j]) <= print_threshold:
                            continue
                        if rhf:
                            print(
                                f"     [{n}]  {i+1} {j+1}"
                                f" -> {a+no+1} {b+no+1}    {t2[a, b, i, j]}"
                            )
                        else:
                            print(
                                f"     [{n}]  {spatial_index(i+1)}{spin_label(i+1)}"
                                f" {spatial_index(j+1)}{spin_label(j+1)}"
                                f" -> {spatial_index(a+no+1)}{spin_label(a+no+1)}"
                                f" {spatial_index(b+no+1)}{spin_label(b+no+1)}"
                                f"    {t2[a, b, i, j]}"
                            )
                        n += 1

    def _noop(*args, **kwargs):
        pass

    mod.print_amplitudes = print_amplitudes
    mod.print_system_information = _noop
    mod.print_custom_system_information = _noop
    mod.print_custom_system_information_rhf = _noop
    mod.get_memory_usage = lambda: 0.0
    return mod


def _inject_shims(miniccpy_path=None):
    """Insert pure-numpy shims into sys.modules so that h5py and pyscf are not
    required when loading miniccpy method modules."""
    if miniccpy_path is not None:
        p = str(miniccpy_path)
        if p not in sys.path:
            sys.path.insert(0, p)

    # Only inject if the real module would fail or has not been loaded yet
    if "miniccpy.diis" not in sys.modules:
        try:
            import h5py  # noqa: F401
        except ImportError:
            sys.modules["miniccpy.diis"] = _make_diis_shim()

    if "miniccpy.printing" not in sys.modules:
        try:
            from pyscf import symm  # noqa: F401
        except ImportError:
            sys.modules["miniccpy.printing"] = _make_printing_shim()


# ──────────────────────────────────────────────────────────────────────────────
# Integral conversion
# ──────────────────────────────────────────────────────────────────────────────

def build_spinorb_integrals(data):
    """Convert spatial Fock (F) and two-electron integrals (G) from
    HamiltonianData into the spin-orbital antisymmetrized form expected by
    miniccpy.

    data.fock[p,q]          : F_pq, closed-shell spatial Fock matrix
    data.two_body[p,q,r,s]  : (pq|rs) chemist notation

    Returns g[p,q,r,s] = <pq||rs> = <pq|rs> - <pq|sr>
    with alpha/beta interleaved spin-orbital ordering.
    """
    if data.fock is None:
        raise RuntimeError("data.fock is None - call data.compute_fock_matrix() first.")

    norb = data.n_orbitals
    n_so = 2 * norb
    n_occ = 2 * data.n_occ_alpha

    # Spin-orbital Fock (block-diagonal for RHF)
    fockso = np.zeros((n_so, n_so))
    fockso[0::2, 0::2] = data.fock
    fockso[1::2, 1::2] = data.fock

    # Spin-orbital chemist integrals (pq|rs)
    tb = data.two_body
    stei = np.zeros((n_so, n_so, n_so, n_so))
    stei[0::2, 0::2, 0::2, 0::2] = tb
    stei[1::2, 1::2, 1::2, 1::2] = tb
    stei[0::2, 0::2, 1::2, 1::2] = tb
    stei[1::2, 1::2, 0::2, 0::2] = tb

    # Antisymmetrize: <pq||rs> = <pq|rs> - <pq|sr>
    # <pq|rs> = stei[p,r,q,s]  => transpose(0,2,1,3)
    # <pq|sr> = stei[p,s,q,r]  => transpose(0,2,3,1)  [NOT (0,3,1,2)]
    g = stei.transpose(0, 2, 1, 3) - stei.transpose(0, 2, 3, 1)

    o = slice(0, n_occ)
    v = slice(n_occ, n_so)
    return fockso, g, o, v


# ──────────────────────────────────────────────────────────────────────────────
# miniccpy solver driver
# ──────────────────────────────────────────────────────────────────────────────

_MPN_METHODS = {"mp2"}


def solve_miniccpy(data, method,
                   maxit=80,
                   convergence=1.0e-07,
                   level_shift=0.0,
                   diis_size=6,
                   n_start_diis=0,
                   miniccpy_path=None):
    """Run a correlated calculation using miniccpy with integrals from HamiltonianData.

    Supported methods: mp2, ccsd, ccsdt, ccsdtq, ccsdt_p, lrccsd, lccd, ccd,
    rccsd, rccd, rccsdt, cc3, cct3, and any other method in miniccpy.

    Args:
        data           : HamiltonianData (fock and two_body must be set)
        method         : miniccpy method name
        maxit          : max CC iterations (default 80)
        convergence    : convergence threshold (default 1e-7)
        level_shift    : amplitude denominator shift (default 0.0)
        diis_size      : DIIS subspace size (default 6)
        n_start_diis   : iteration to start DIIS (default 0)
        miniccpy_path  : optional path to miniccpy source directory

    Returns:
        dict with keys: method, e_corr, e_total, T
    """
    # Inject shims before any miniccpy method module is imported
    _inject_shims(miniccpy_path)

    try:
        from miniccpy.driver import run_cc_calc, run_mpn_calc
    except ImportError as exc:
        raise ImportError(
            f"miniccpy could not be imported: {exc}\n"
            "Ensure miniccpy is installed or pass miniccpy_path= to solve_miniccpy()."
        ) from exc

    fockso, g, o, v = build_spinorb_integrals(data)
    method_lower = method.lower()

    T = None
    if method_lower in _MPN_METHODS:
        e_corr = run_mpn_calc(fockso, g, o, v, method_lower)
    else:
        T, e_corr = run_cc_calc(
            fockso, g, o, v, method_lower,
            maxit=maxit,
            convergence=convergence,
            energy_shift=level_shift,
            diis_size=diis_size,
            n_start_diis=n_start_diis,
        )

    e_total = data.scf_energy + e_corr
    print("")
    print(f"    {method.upper()} Correlation Energy: {e_corr: 20.12f}")
    print(f"    {method.upper()} Total Energy:       {e_total: 20.12f}")
    print("")

    return {"method": method_lower, "e_corr": e_corr, "e_total": e_total, "T": T}
