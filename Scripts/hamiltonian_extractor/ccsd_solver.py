"""
Coupled Cluster Singles and Doubles (CCSD) solver implementation.
"""
import numpy as np
from typing import Dict, Tuple

def solve_ccsd(data, threshold: float = 0.001) -> Dict:
    """Solve the CCSD problem for given one and two electron integrals.

    Args:
        data: HamiltonianData object containing one_body and two_body integrals
        threshold: Threshold for printing T1/T2 amplitudes (default: 0.001)

    Returns:
        Dictionary with CCSD energy, T1 amplitudes, and T2 amplitudes
    """
    try:
        from pyscf import gto, scf, cc, ao2mo
    except ImportError:
        raise ImportError("PySCF is required for CCSD calculations. Please install it using 'pip install pyscf'.")

    n_occ = data.n_occ_alpha
    n_orb = data.n_orbitals
    n_electrons = 2 * n_occ  # closed shell

    # Build a fake molecule with the correct number of electrons and no basis
    mol = gto.M()
    mol.nelectron = n_electrons
    mol.spin = 0
    mol.verbose = 0

    # Use a fake mean-field object backed by the extracted integrals
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: data.one_body
    mf.get_ovlp = lambda *args: np.eye(n_orb)
    mf._eri = ao2mo.restore(8, data.two_body, n_orb)

    # Build the MO coefficients (identity: integrals already in MO basis)
    mf.mo_coeff = np.eye(n_orb)
    mf.mo_occ = np.array([2.0] * n_occ + [0.0] * (n_orb - n_occ))
    mf.mo_energy = np.diag(data.fock) if data.fock is not None else np.zeros(n_orb)
    mf.e_tot = data.scf_energy if data.scf_energy is not None else 0.0

    # Run CCSD
    mycc = cc.RCCSD(mf)
    mycc.verbose = 5
    e_corr, t1, t2 = mycc.kernel()

    total_energy = (mf.e_tot + e_corr +
                    data.coulomb_repulsion['value'] +
                    data.energy_shift)

    # Print T1 amplitudes above threshold
    print("\n--- CCSD T1 Amplitudes (i -> a) ---")
    for i in range(n_occ):
        for a in range(n_orb - n_occ):
            val = t1[i, a]
            if abs(val) > threshold:
                print(f"  t1[{i},{n_occ + a}] = {val: .8f}")

    # Print T2 amplitudes above threshold
    print("\n--- CCSD T2 Amplitudes (i,j -> a,b) ---")
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_orb - n_occ):
                for b in range(n_orb - n_occ):
                    val = t2[i, j, a, b]
                    if abs(val) > threshold:
                        print(f"  t2[{i},{j},{n_occ + a},{n_occ + b}] = {val: .8f}")

    return {
        'energy': total_energy,
        'e_corr': e_corr,
        't1': t1,
        't2': t2,
    }
