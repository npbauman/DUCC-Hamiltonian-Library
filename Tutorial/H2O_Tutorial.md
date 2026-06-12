# H2O DUCC Hamiltonian Extraction Tutorial

This tutorial walks through the H2O equilibrium example in this repository and shows how to extract the downfolded DUCC Hamiltonian into common formats.

## What this example is

The H2O equilibrium case is located in:

- `H2O/Eq`

This calculation uses a cc-pVTZ basis and downfolds correlation into an effective active-space Hamiltonian (10 electrons in 10 orbitals).

In `H2O/Eq`, you should find:

- `H2O.json` (input)
- `H2O.out` (main ExaChem output)
- `H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt` (raw DUCC effective Hamiltonian text)

## Script used

Hamiltonian extraction is handled by:

- `Scripts/hamiltonian_extractor/extract_hamiltonian.py`

The script can generate:

- FCIDUMP
- YAML
- XACC
- an info summary file (always written)

## Prerequisites

- Python 3.6+
- NumPy
- Optional for YAML output: PyYAML or ruamel.yaml
- Optional for FCI: PySCF
- Optional for PySCF CCSD mode: PySCF
- Optional for miniccpy correlated methods: miniccpy

## Recommended run location

Run commands from `H2O/Eq`.

## Basic extraction

From `H2O/Eq`, run:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt
```

By default, this writes all three formats plus an info file.

## Selecting output format(s)

Only FCIDUMP:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --format fcidump
```

YAML and XACC only:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --format yaml xacc
```

Use a custom output prefix:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --output-prefix H2O_ducc_ccpvtz
```

## Output files to expect

With default prefix (`H2O`), the script writes:

- `H2O-info`
- `H2O-FCIDUMP` (if FCIDUMP requested)
- `H2O.yaml` (if YAML requested)
- `H2O-xacc` (if XACC requested)

## Validation checks

The most useful quick validation is in `H2O-info`:

- `Original SCF Energy`

Use this value to verify your downstream workflow read the Hamiltonian correctly.

The info file also records:

- nuclear repulsion
- energy shift
- active orbital/electron partitioning
- Fock matrix elements above the print threshold

## Optional: FCI verification

If PySCF is available, run an exact diagonalization check:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --fci
```

For multiple roots:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --fci --n-roots 3
```

## Optional: connect to existing PySCF workflows

If your pipeline already uses PySCF, the extractor can run a CCSD calculation directly on the extracted effective Hamiltonian:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --pyscf-ccsd
```

Implementation details are in:

- `Scripts/hamiltonian_extractor/ccsd_solver.py`

That solver builds a PySCF RHF object using the extracted one- and two-electron integrals in MO form, which is a good reference for integrating DUCC-extracted Hamiltonians into a PySCF-based workflow.

## Optional: correlated methods via miniccpy

If miniccpy is installed, you can run methods such as CCSD from the same extracted Hamiltonian:

```bash
python ../../Scripts/hamiltonian_extractor/extract_hamiltonian.py \
  H2O.out \
  H2O.cc-pvtz_files/restricted/ducc/H2O.cc-pvtz.ducc.results.txt \
  --cc-method ccsd
```

If miniccpy is not on your Python path, add:

```bash
--miniccpy-path /path/to/miniccpy
```

## Notes

- The extractor is currently set up for closed-shell systems.
- If YAML writing fails, install one of: `PyYAML` or `ruamel.yaml`.
- If FCI or PySCF-CCSD fails, verify that `pyscf` is installed in the Python environment you are using.
