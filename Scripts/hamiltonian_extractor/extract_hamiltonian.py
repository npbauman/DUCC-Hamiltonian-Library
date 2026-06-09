#!/usr/bin/env python3
"""
Hamiltonian Extractor Utility

This utility extracts Hamiltonian data from DUCC calculation output files
and converts it to various formats for different quantum chemistry and
quantum computing applications.

Author: ExaChem Development Team
Note: This is only set up for closed shell systems.
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from grab_data import extract_hamiltonian_data
    from format_writer import write_info_file
    from fcidump_writer import FCIDUMPWriter
    from yaml_writer import YAMLWriter
    from xacc_writer import XACCWriter
    from fci_solver import solve_fci
    from miniccpy_solver import solve_miniccpy
    from ccsd_solver import solve_ccsd
except ImportError:
    # Try relative imports if absolute imports fail
    from .grab_data import extract_hamiltonian_data
    from .format_writer import write_info_file
    from .fcidump_writer import FCIDUMPWriter
    from .yaml_writer import YAMLWriter
    from .xacc_writer import XACCWriter
    from .fci_solver import solve_fci
    from .miniccpy_solver import solve_miniccpy
    from .ccsd_solver import solve_ccsd


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract Hamiltonian data from DUCC calculations and convert to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all formats
  python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt

  # Extract only FCIDUMP format
  python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --format fcidump

  # Extract multiple specific formats
  python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --format yaml xacc

Available formats:
  - fcidump: FCIDUMP format (modified for DMRG calculations)
  - yaml: YAML format based on Broombridge version 0.3
  - xacc: XACC format for quantum computing applications
  - info: Information file about the extracted Hamiltonian (always generated)
        """
    )
    
    parser.add_argument(
        'output_file',
        help='Path to the output file'
    )
    
    parser.add_argument(
        'integral_path',
        help='Path to the integral file (typically ends with .ducc.results.txt)'
    )
    
    parser.add_argument(
        '--format',
        choices=['fcidump', 'yaml', 'xacc', 'all'],
        nargs='*',
        default=['all'],
        help='Output format(s) to generate. Use "all" to generate all formats (default: all)'
    )
    
    parser.add_argument(
        '--output-prefix',
        help='Prefix for output files (default: use input output_file name)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.00000000005,
        help='Threshold for printing integrals (default: 5e-11)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--fci',
        action='store_true',
        help='Perform Full Configuration Interaction (FCI) calculation'
    )
    
    parser.add_argument(
        '--n-roots',
        type=int,
        default=1,
        help='Number of FCI roots to compute (default: 1)'
    )
    
    parser.add_argument(
        '--fci-threshold',
        type=float,
        default=0.001,
        help='Threshold for printing CI vector components (default: 0.001)'
    )

    # ── MP2 (built-in, no extra dependency) ──────────────────────────────────
    parser.add_argument(
        '--mp2',
        action='store_true',
        help='Compute MP2 energy using built-in spin-orbital implementation'
    )

    # ── miniccpy correlated methods ───────────────────────────────────────────
    parser.add_argument(
        '--cc-method',
        metavar='METHOD',
        help=(
            'Run a correlated calculation via miniccpy using the extracted '
            'Fock and two-electron integrals. '
            'Examples: ccsd, ccsdt, ccsdtq, mp2, cc3, lrccsd. '
            'Requires miniccpy to be installed.'
        )
    )

    parser.add_argument(
        '--miniccpy-path',
        metavar='PATH',
        default=None,
        help='Path to prepend to sys.path so that miniccpy can be imported '
             '(e.g. /path/to/miniccpy). Optional if miniccpy is already installed.'
    )

    parser.add_argument(
        '--cc-maxit',
        type=int,
        default=80,
        help='Maximum CC iterations for miniccpy solver (default: 80)'
    )

    parser.add_argument(
        '--cc-convergence',
        type=float,
        default=1.0e-7,
        help='Convergence threshold for miniccpy CC solver (default: 1e-7)'
    )

    parser.add_argument(
        '--cc-level-shift',
        type=float,
        default=0.0,
        help='Amplitude denominator level shift for miniccpy convergence '
             '(default: 0.0, distinct from the DUCC energy shift)'
    )

    parser.add_argument(
        '--cc-diis-size',
        type=int,
        default=6,
        help='DIIS subspace size for miniccpy CC solver (default: 6)'
    )

    # ── PySCF CCSD ───────────────────────────────────────────────────────────
    parser.add_argument(
        '--pyscf-ccsd',
        action='store_true',
        help='Perform CCSD calculation using PySCF'
    )

    parser.add_argument(
        '--pyscf-ccsd-threshold',
        type=float,
        default=0.001,
        help='Threshold for printing T1/T2 amplitudes (default: 0.001)'
    )

    return parser


def get_format_writers(output_prefix):
    """Get dictionary of available format writers."""
    return {
        'fcidump': FCIDUMPWriter(output_prefix),
        'yaml': YAMLWriter(output_prefix),
        'xacc': XACCWriter(output_prefix)
    }


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.output_file).exists():
        print(f"Error: Output file '{args.output_file}' does not exist.")
        sys.exit(1)
    
    if not Path(args.integral_path).exists():
        print(f"Error: Integral file '{args.integral_path}' does not exist.")
        sys.exit(1)
    
    # Determine output prefix
    output_prefix = args.output_prefix or Path(args.output_file).stem
    
    # Extract data
    if args.verbose:
        print(f"Extracting data from {args.output_file} and {args.integral_path}")
    
    try:
        data = extract_hamiltonian_data(args.output_file, args.integral_path)
        data.printthresh = args.threshold
        
        if args.verbose:
            print(f"Successfully extracted data:")
            print(f"  - {data.n_orbitals} orbitals")
            print(f"  - {data.n_occ_alpha} occupied alpha orbitals")
            print(f"  - {data.n_occ_beta} occupied beta orbitals")
            print(f"  - {data.n_virt_alpha} virtual alpha orbitals")
            print(f"  - {data.n_virt_beta} virtual beta orbitals")
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        sys.exit(1)
    
    # Always write info file
    if args.verbose:
        print("Writing info file...")
    write_info_file(data, output_prefix)
    
    # ── Built-in MP2 ──────────────────────────────────────────────────────────
    if args.mp2:
        if args.verbose:
            print("\nComputing MP2 energy (built-in spin-orbital implementation)...")
        mp2_corr = data.compute_MP2_energy()
        mp2_total = data.scf_energy + mp2_corr
        print(f"MP2 Total Energy = {mp2_total:.12f}")

    # ── Run FCI if requested ──────────────────────────────────────────────────
    if args.fci:
        if args.verbose:
            print(f"\nPerforming FCI calculation with {args.n_roots} root(s)...")
        
        # Run FCI
        fci_results = solve_fci(
            data,
            n_roots=args.n_roots,
            threshold=args.fci_threshold
        )
        
        # Print FCI results
        for i, result in enumerate(fci_results):
            print(f"\nRoot {i}:")
            print(f"FCI Energy = {result['energy']:.12f}")
            print(f"2S+1 = {result['multiplicity']:.7f}")
            print(f"\nCI vector components (|coef| > {args.fci_threshold}):")
            for occ_str, coef in sorted(result['ci_vector'].items(), 
                                      key=lambda x: (abs(x[1]), x[0]),
                                      reverse=True):
                print(f"{occ_str}: {coef:10.5f}")

    # ── miniccpy correlated calculation ──────────────────────────────────────
    if args.cc_method:
        if args.verbose:
            print(f"\nRunning {args.cc_method.upper()} via miniccpy...")
        try:
            result = solve_miniccpy(
                data,
                method=args.cc_method,
                maxit=args.cc_maxit,
                convergence=args.cc_convergence,
                level_shift=args.cc_level_shift,
                diis_size=args.cc_diis_size,
                miniccpy_path=args.miniccpy_path,
            )
        except ImportError as e:
            print(f"Error: {e}")
            print("  Hint: install miniccpy or pass --miniccpy-path <path>")
            sys.exit(1)
        except NotImplementedError:
            print(f"Error: method '{args.cc_method}' is not implemented in miniccpy.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during {args.cc_method.upper()} calculation: {e}")
            sys.exit(1)

    # ── PySCF CCSD ───────────────────────────────────────────────────────────
    if args.pyscf_ccsd:
        if args.verbose:
            print("\nRunning CCSD via PySCF...")
        try:
            data.compute_fock_matrix()
            result = solve_ccsd(data, threshold=args.pyscf_ccsd_threshold)
            print(f"\nCCSD Correlation Energy = {result['e_corr']:.12f}")
            print(f"CCSD Total Energy       = {result['energy']:.12f}")
        except ImportError as e:
            print(f"Error: {e}")
            print("  Hint: install pyscf with 'pip install pyscf'")
            sys.exit(1)
        except Exception as e:
            print(f"Error during PySCF CCSD calculation: {e}")
            sys.exit(1)

    # Determine which formats to write
    formats_to_write = args.format
    if 'all' in formats_to_write:
        formats_to_write = ['fcidump', 'yaml', 'xacc']
    
    # Get available writers
    writers = get_format_writers(output_prefix)
    
    # Write requested formats
    for format_name in formats_to_write:
        if format_name in writers:
            writer = writers[format_name]
            try:
                if args.verbose:
                    print(f"Writing {format_name} format ({writer.get_description()})...")
                writer.write(data)
                if args.verbose:
                    print(f"  -> {output_prefix}{writer.get_file_extension()}")
            except Exception as e:
                print(f"Error writing {format_name} format: {e}")
                if format_name == 'yaml' and 'yaml' in str(e).lower():
                    print("  Hint: Install PyYAML or ruamel.yaml: pip install PyYAML")
                sys.exit(1)
        else:
            print(f"Warning: Unknown format '{format_name}' ignored")
    
    if args.verbose:
        print("Hamiltonian extraction completed successfully!")


if __name__ == '__main__':
    main()
