Welcome to the Double Unitary Coupled Cluster (DUCC) Hamiltonian Library. This repo also contains bare active-space Hamiltonians.

Each molecule has various file formats with various information. For further details about the files, please see below.

- *.json - Input for the ExaChem calculations that generates the Hamiltonians.
- *.out - Output of the ExaChem calculation. 
- *.yaml - YAML format of the Hamiltonian based on the Broombridge Schema.
- *.out-FCIDUMP - FCIDUMP-like format of the Hamiltonian.
- *.out-xacc - Fermionic representation of the Hamiltonian for the XACC software.
- *.out-info - Some information about the active-space effective Hamiltonian. 
- *.FCI - output of the FCI calculation for the Hamiltonian. 
- *_files/ - Directory with raw data from the calculation.

**Hamiltonian Symmetries**
Hamiltonians are Hermitian. While effective Hamiltonians can contain higher-body terms, we only consider up to two-body interactions:

H = *scalar* + h(i, j) + v(i, j, k, l)

Scalar terms refer to a collective scalar for the nuclear repulsion energy, dropped core energy, and, in the case of DUCC Hamiltonians, fully contracted terms from the theory.

Since the one-electron integrals are Hermitian,

h(i, j) = h(j, i)

For two-electron integrals, consider the following format:

v(electron 1, electron 1, electron 2, electron 2)

Bare Hamiltonians have the standard 8-fold symmetry:

v(i, j, k, l) = v(k, l, i, j) = v(j, i, l, k) = v(l, k, j, i) = v(j, i, k, l) = v(i, j, l, k) = v(k, l, j, i) = v(l, k, i, j)

where i, j, k, l are orbital indices (as opposed to spin-orbital indices)

The DUCC effective Hamiltonians have a lower 4-fold symmetry:

v(i, j, k, l) = v(k, l, i, j) = v(j, i, l, k) = v(l, k, j, i)

**File Formats**

*.yaml*

*out-FCIDUMP*

This is a unique nonstandard FCIDUMP format. The file is separated into five sections of 'orbital' integrals (separated by a "0.0 0 0 0 0" line) and a scalar value.

- The first section is the alpha-alpha terms of the two-electron integrals. They are antisymmetrized and labeled with orbital (not spin-orbital) indices.
- The second section is the beta-beta terms of the two-electron integrals. They are antisymmetrized and labeled with orbital indices.
- The third section is the alpha-beta terms of the two-electron integrals, labeled with orbital indices.
- The fourth section is the alpha terms of the one-electron integrals labeled with orbital indices.
- The fifth section is the beta terms of the one-electron integrals labeled with orbital indices.
- Then the last line is the scalar term, which accounts for the nuclear repulsion, dropped core, and the fully contracted terms from the DUCC method.

**Scripts**

Compute_FCI_Energy.py - Computed the FCI energy using the *.yaml file.

ExtractDUCC.py - Script used to transform the raw Hamiltonian data into the various file formats.
