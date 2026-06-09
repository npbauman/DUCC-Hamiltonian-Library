"""
Data structures and utility functions for Hamiltonian extraction.
"""
import numpy as np
from numpy import linalg as LA


class HamiltonianData:
    """Container for extracted Hamiltonian data and analysis results."""
    
    def __init__(self):
        # Raw data from files
        self.geometry = None
        self.coulomb_repulsion = None
        self.scf_energy = None
        self.n_occ_alpha = None
        self.n_occ_beta = None
        self.n_virt_alpha = None
        self.n_virt_beta = None
        self.n_frozen_core = None
        self.n_orbitals = None
        self.energy_shift = None
        self.note = None
        
        # Integral matrices
        self.one_body = None
        self.two_body = None
        self.fock = None
        self.fockso = None  # Spin-orbital Fock matrix
        
        # Processed data
        self.scf_energy = None
        # self.orbital_energies = None
        
        # Parameters
        self.printthresh = 0.00000000005
        
    def compute_energies(self):
        """Compute Hartree-Fock and core energies."""
        # Safety guard: treat missing energy shift as zero
        if self.energy_shift is None:
            self.energy_shift = 0.0
        # Compute the Hartree-Fock energy
        self.scf_energy = self.coulomb_repulsion['value'] + self.energy_shift
        for x in range(self.n_occ_alpha):
            self.scf_energy += 2 * self.one_body[x, x]
            for y in range(self.n_occ_alpha):
                self.scf_energy += 2 * self.two_body[x, x, y, y] - self.two_body[x, y, y, x]


    def compute_fock_matrix(self):
        """Form the Fock operator matrix (spatial and spin-orbital)."""
        self.fock = np.zeros((self.n_orbitals, self.n_orbitals))
        for x in range(self.n_orbitals):
            for y in range(self.n_orbitals):
                self.fock[x, y] += self.one_body[x, y]
                for z in range(self.n_occ_alpha):
                    self.fock[x, y] += 2 * self.two_body[z, z, x, y] - self.two_body[z, y, x, z]

        # Populate spin-orbital Fock matrix (block diagonal: alpha == beta)
        self.fockso = np.zeros((2 * self.n_orbitals, 2 * self.n_orbitals))
        self.fockso[0::2, 0::2] = self.fock
        self.fockso[1::2, 1::2] = self.fock

    def compute_MP2_energy(self):
        """Compute MP2 energy with verification of Hartree-Fock energy."""
        num_spin_orbs = 2 * self.n_orbitals
        num_electrons = self.n_occ_alpha + self.n_occ_beta

        # Build spin-orbital integrals
        soei, stei = self._build_spin_orbital_integrals(num_spin_orbs)

        # Verify HF energy with spin-orbital formulation
        self._verify_hf_energy_spin_orbitals(soei, stei, num_electrons)

        # Build antisymmetrized two-electron integrals
        atei = self._build_antisymmetrized_integrals(stei, num_spin_orbs)

        # Verify HF energy with antisymmetrized integrals
        self._verify_hf_energy_antisymmetrized(soei, atei, num_electrons)

        # Compute MP2 energy
        mp2_energy = self._compute_mp2_correlation(atei, num_electrons, num_spin_orbs)

        print(f"MP2 Correction = {mp2_energy:.8f}")
        return mp2_energy

    def _build_spin_orbital_integrals(self, num_spin_orbs):
        """Build spin-orbital one- and two-electron integrals."""
        soei = np.zeros((num_spin_orbs, num_spin_orbs))
        stei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))

        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                # Populate 1-body coefficients (same spin only)
                soei[2 * p, 2 * q] = self.one_body[p, q]
                soei[2 * p + 1, 2 * q + 1] = self.one_body[p, q]

                # Populate 2-body coefficients
                for r in range(self.n_orbitals):
                    for s in range(self.n_orbitals):
                        # [ij|kl] format - same spin terms
                        stei[2 * p, 2 * q, 2 * r, 2 * s] = self.two_body[p, q, r, s]
                        stei[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = self.two_body[p, q, r, s]
                        # Mixed spin terms
                        stei[2 * p, 2 * q, 2 * r + 1, 2 * s + 1] = self.two_body[p, q, r, s]
                        stei[2 * p + 1, 2 * q + 1, 2 * r, 2 * s] = self.two_body[p, q, r, s]

        return soei, stei

    def _verify_hf_energy_spin_orbitals(self, soei, stei, num_electrons):
        """Verify Hartree-Fock energy using spin-orbital formulation."""
        print("\nRe-computing Hartree-Fock energy (Spin-Orbitals)...")
        hf_energy = self.coulomb_repulsion['value'] + self.energy_shift

        for p in range(num_electrons):
            hf_energy += soei[p, p]
            for q in range(num_electrons):
                hf_energy += 0.5 * stei[p, p, q, q] - 0.5 * stei[p, q, q, p]

        print(f"Hartree-Fock energy = {hf_energy:.8f}")

    def _build_antisymmetrized_integrals(self, stei, num_spin_orbs):
        """Build antisymmetrized two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>."""
        print("\nRe-computing Hartree-Fock energy (Antisymmetrized Spin-Orbitals)...")
        atei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))
        for p in range(num_spin_orbs):
            for q in range(num_spin_orbs):
                for r in range(num_spin_orbs):
                    for s in range(num_spin_orbs):
                        atei[p, q, r, s] = stei[p, r, q, s] - stei[p, s, q, r]
        return atei

    def _verify_hf_energy_antisymmetrized(self, soei, atei, num_electrons):
        """Verify Hartree-Fock energy using antisymmetrized integrals."""
        hf_energy = self.coulomb_repulsion['value'] + self.energy_shift

        for p in range(num_electrons):
            hf_energy += soei[p, p]
            for q in range(num_electrons):
                hf_energy += 0.5 * atei[p, q, p, q]

        print(f"Hartree-Fock energy (antisymmetrized) = {hf_energy:.8f}")

    def _compute_mp2_correlation(self, atei, num_electrons, num_spin_orbs):
        """Compute MP2 correlation energy."""
        mp2_energy = 0.0

        for i in range(num_electrons):
            for j in range(num_electrons):
                for a in range(num_electrons, num_spin_orbs):
                    for b in range(num_electrons, num_spin_orbs):
                        denominator = (self.fockso[i, i] + self.fockso[j, j] -
                                       self.fockso[a, a] - self.fockso[b, b])
                        mp2_energy += (atei[i, j, a, b] ** 2) / denominator

        return mp2_energy / 4
