# DUCC Hamiltonian Library

Welcome to the Double Unitary Coupled Cluster (DUCC) Hamiltonian Library. This repository contains DUCC effective Hamiltonians and bare active-space Hamiltonians.

## Repository Contents

Each molecular system may include several file types:

- `*.json`: Input file for the ExaChem calculation that generates the Hamiltonian
- `*.out`: Main ExaChem output file
- `*.yaml`: Hamiltonian in YAML format based on the Broombridge schema
- `*.out-FCIDUMP`: FCIDUMP-like Hamiltonian format
- `*.out-xacc`: Fermionic Hamiltonian format for XACC workflows
- `*.out-info`: Summary and validation information for the active-space/effective Hamiltonian
- `*_files/`: Directory with raw calculation artifacts

## Tutorials

- H2O DUCC extraction walkthrough: `Tutorial/H2O_Tutorial.md`

## Hamiltonian Symmetries

Hamiltonians in this repository are Hermitian. While effective Hamiltonians can contain higher-body terms, this repository currently provides up to two-body interactions:

$$
H = \text{scalar} + h(i, j) + v(i, j, k, l)
$$

The scalar term contains nuclear repulsion, dropped-core contributions, and for DUCC Hamiltonians, fully contracted DUCC terms.

Because the one-electron integrals are Hermitian:

$$
h(i, j) = h(j, i)
$$

For two-electron integrals, index ordering is written as:

$$
v(\text{electron 1}, \text{electron 1}, \text{electron 2}, \text{electron 2})
$$

Bare Hamiltonians have standard 8-fold symmetry:

$$
v(i, j, k, l) = v(k, l, i, j) = v(j, i, l, k) = v(l, k, j, i) = v(j, i, k, l) = v(i, j, l, k) = v(k, l, j, i) = v(l, k, i, j)
$$

where $i, j, k, l$ are orbital indices (not spin-orbital indices).

DUCC effective Hamiltonians have 4-fold symmetry:

$$
v(i, j, k, l) = v(k, l, i, j) = v(j, i, l, k) = v(l, k, j, i)
$$

## File Format Notes

### `*.out-FCIDUMP`

This is a nonstandard FCIDUMP-style format. The file is separated into five orbital-integral sections (each section separated by a `0.0 0 0 0 0` line), followed by a scalar value:

- Section 1: alpha-alpha two-electron integrals (antisymmetrized, orbital indices)
- Section 2: beta-beta two-electron integrals (antisymmetrized, orbital indices)
- Section 3: alpha-beta two-electron integrals (orbital indices)
- Section 4: alpha one-electron integrals (orbital indices)
- Section 5: beta one-electron integrals (orbital indices)
- Final scalar line: nuclear repulsion + dropped-core + DUCC contracted terms

### `*.yaml`

The YAML output follows the Broombridge-style schema used by the extraction scripts.

## Scripts

- `Scripts/hamiltonian_extractor/extract_hamiltonian.py`: Converts raw DUCC data into FCIDUMP, YAML, XACC, and info outputs
- `Scripts/hamiltonian_extractor/README.md`: Detailed script usage and command-line options
