import numpy as np
import re

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def dict_sort_and_select(my_dict:dict, num_select:int = 10):
    """
    ## sort the dict value from large to small and return the first num_select items
    Args:
        my_dict (dict): The dictionary to be sorted.
        num_select (int): The number of items to select from the sorted dictionary.
    Returns:    
        dict: A dictionary containing the first num_select items sorted by value.
    """
    sorted_items = sorted(my_dict.items(), key=lambda item: np.abs(np.array(item[1])).sum(), reverse=True)
    selected_items = sorted_items[:num_select]
    return dict(selected_items)

def dict_sort_and_select_partial(my_dict:dict, start:int, end:int):
    """
    ## sort the dict value from large to small and return the first num_select items
    Args:
        my_dict (dict): The dictionary to be sorted.
        num_select (int): The number of items to select from the sorted dictionary.
    Returns:    
        dict: A dictionary containing the first num_select items sorted by value.
    """
    sorted_items = sorted(my_dict.items(), key=lambda item: np.abs(np.array(item[1])).sum(), reverse=True)
    selected_items = sorted_items[start:end]
    if len(selected_items) == 0:
        return {}
    return dict(selected_items)


def pauli_coeff_separation(qis_qubit_op:qiskit.quantum_info.SparsePauliOp):
    pauli_arr = []
    coeff_arr = []
    for pp, cc in zip(qis_qubit_op.paulis, qis_qubit_op.coeffs):
        pauli_arr.append(str(pp))
        if np.imag(cc) == 0:
            cc = np.real(cc)
        coeff_arr.append(cc)
    return pauli_arr, coeff_arr





def qiskit_normal_order_switch_vec(qiskit_vector):
    ## qiskit matrix to normal matrix or verse versa
    num_qubits = int(np.log2(len(qiskit_vector)))
    bin_str = '{0:0'+str(num_qubits)+'b}'
    new_vector = np.zeros(qiskit_vector.shape, dtype=complex)
    for i in range(len(qiskit_vector)):
        normal_i = int(bin_str.format(i)[::-1],2)
        new_vector[normal_i] = qiskit_vector[i]
    if np.linalg.norm(new_vector.imag) < 1e-12:
        new_vector = new_vector.real
    return new_vector


def parse_operator_string(op_str):
    """Parse operator string from ADAPT-VQE output format to Pauli string format."""
    # Extract the coefficient and Pauli string
    match = re.match(r'\(([-+]?[\d\.e]+ \+ [-+]?[\d\.e]+i)\)(.*)', op_str)
    if match:
        coeff_str, pauli_str = match.groups()
        real_str, imag_str = coeff_str.split(' + ')
        real_val = float(real_str.replace(' ', ''))
        imag_val = float(imag_str.replace('i', ''))
        # Convert coefficient string to complex number
        coeff = complex(real_val, imag_val)
        return coeff, pauli_str
    return None, None

def extract_operators_from_file(results_file, inverse_pauli = False):
    """Extract operators and initial parameters from ADAPT-VQE results."""
    # Read the results file
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract the final parameters section
    final_params_match = re.search(r'Final parameters:(.*?)$', content, re.DOTALL)
    if not final_params_match:
        raise ValueError("Could not find final parameters in the results file")
    
    final_params_section = final_params_match.group(1).strip()
    
    # Parse operators and parameters
    operators = []
    coeffs = []
    final_params = []
    
    for line in final_params_section.split('\n'):
        if not line.strip():
            continue
        # Extract operator and parameter using a more flexible regex
        match = re.match(r'\s*(\([^)]+\)[A-Z]+)\s*::\s*([-+]?[\d\.e]+)', line)
        if match:
            op_str, param_val = match.groups()
            coeff, pauli_str = parse_operator_string(op_str)
            if inverse_pauli:
                pauli_str = pauli_str[::-1]
            pauli_op = SparsePauliOp(pauli_str, coeff)
            if coeff is not None:
                operators.append(pauli_op)
                final_params.append(float(param_val))
    return operators, final_params

def extract_iteration_energies(results_file):
    """Extract objective values (energies) from each iteration in the ADAPT-VQE results."""
    # Read the results file
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract the iteration summary section
    iter_summary_match = re.search(r'----------- Iteration Summary -----------\n(.*?)--------- Result Summary ---------', content, re.DOTALL)
    if not iter_summary_match:
        raise ValueError("Could not find iteration summary in the results file")
    
    iter_summary = iter_summary_match.group(1).strip()
    # Parse iteration data
    iterations = []
    energies = []
    
    # Skip the header line
    for line in iter_summary.split('\n')[2:]:
        if not line.strip():
            continue
        
        # Extract iteration number and objective value
        parts = line.split()
        if len(parts) >= 3:
            iter_num = int(parts[0])
            energy = float(parts[1])
            iterations.append(iter_num)
            energies.append(energy)
    
    return iterations, energies

def extract_nwqsim_output(results_file, inverse_pauli = False):
    """Extract the output from NWQ-Sim."""
    res_dict = {"iterations": [], "energies": [], "operators": [], "parameters": []}
    operators, final_params = extract_operators_from_file(results_file, inverse_pauli)
    res_dict["operators"] = operators
    res_dict["parameters"] = final_params
    iterations, energies = extract_iteration_energies(results_file)
    res_dict["iterations"] = iterations
    res_dict["energies"] = energies
    return res_dict



from qiskit.circuit import ParameterVector
def create_vqe_circ(operators,  n_qubits, n_electrons, interleaved=False, no_HF=False, half_barriar=False):
    """Create a parameterized circuit for VQE using PauliEvolutionGate."""

    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    # Add Hartree-Fock initial state (|10101010>)
    if not no_HF: ## for GCIM left and right state
        if not interleaved: ## Qiskit mapping
            for i in range(n_electrons//2):
                qc.x(i)
                qc.x(i+n_qubits//2)
        else:
            for i in range(n_electrons):
                qc.x(i)

    if len(operators) == 0:
        return qc, []

    # Create parameters
    params = ParameterVector('θ', len(operators))
    
    # Apply operators in reverse order (since they were added in order in ADAPT-VQE)
    for i in range(len(operators)):
        if half_barriar and i == int(len(operators)//2):
            qc.barrier()
        evolution_gate = PauliEvolutionGate(operators[i], time=params[i])
        qc.append(evolution_gate, range(n_qubits))
    
    return qc, params


from qiskit.primitives import StatevectorEstimator
def cost_function_custom(prep_circ, ham_op, parameters):
    """Compute the energy for given parameters."""
    estimator = StatevectorEstimator()
    job = estimator.run([(prep_circ, ham_op, parameters)])
    return job.result()

def cost_function_custom_assign(prep_circ, ham_op, parameters):
    """Compute the energy for given parameters."""
    estimator = StatevectorEstimator()
    job = estimator.run([(prep_circ.assign_parameters(parameters), ham_op)])
    return job.result()







def decide_grouping_pauli(group):
    """
    Given a list of Pauli strings (e.g., "XIZ"), decide the effective grouping Pauli.
    For each qubit (column), if any string applies a non-identity operator, we require that all
    non-identity operators are the same. Otherwise, the effective operator is 'I'.
    """
    n_qubits = len(group[0])
    grouping = ""
    
    for i in range(n_qubits):
        # Collect the non-identity Pauli operators at qubit i
        non_identity = {p[i] for p in group if p[i] != 'I'}
        if len(non_identity) > 1:
            raise ValueError(f"Inconsistent non-identity operators at qubit {i}: {non_identity}")
        elif len(non_identity) == 1:
            # The effective operator is the unique non-identity element
            grouping += list(non_identity)[0]
        else:
            # All operators are identity at this qubit
            grouping += 'I'
    return grouping



def grouped_pauli_in_dict(pauli_op, use_qwc=False):
    """Group the Pauli operators into commuting groups.
    
    Args:
        pauli_op (Qiskit.SparsePauliOp): The linear combination of Pauli operators.
        use_qwc (bool): Whether to use the QWC for the grouping method.
    Returns:
        dict: A dictionary of commuting groups with the first Pauli operator in each group as the key.
    """
    qwc_group = pauli_op.group_commuting(qubit_wise=use_qwc) ## Return a list of commuting groups in the form of [SparsePauliOp(), SparsePauliOp(), ...]
    # qwc_group = [p for p in pauli_op]
    ## Select the first Pauli operator in the group as the representative
    qwc_group_dict_paulis = {}
    qwc_group_dict_coeffs = {}
    for sp in qwc_group:
        pauli_str_arr = [str(p) for p in sp.paulis]
        grouping_pauli = decide_grouping_pauli(pauli_str_arr)
        qwc_group_dict_paulis[grouping_pauli] = pauli_str_arr
        qwc_group_dict_coeffs[grouping_pauli] = list(sp.coeffs)
    return qwc_group_dict_paulis, qwc_group_dict_coeffs



def expectation_from_probabilities(probabilities, observable):
    """
    Compute the expectation value of a Pauli operator given measurement outcomes.
    
    Parameters:
      probabilities: array
          values are the corresponding probabilities.
      pauli_str: str
          A string representing the Pauli operator (e.g., "XZZ").
          For qubits where the operator is the identity ('I'), we ignore the measurement outcome.
    Returns:
      expectation: float
          The expectation value computed as the weighted sum of ±1 contributions.
    """
    expectation = 0.0
    num_qubits = len(observable)
    # Loop over every measurement outcome
    for index,prob in enumerate(probabilities): #probabilities:
        outcome = format(index, '0'+str(num_qubits)+'b')
        # Initialize contribution for this outcome.
        # For each qubit where the Pauli is nontrivial, multiply by the eigenvalue:
        # '0' -> +1, '1' -> -1.
        value = 1
        for i, op in enumerate(observable):
            if op != 'I':
                # Adjust: outcome[i]=='0' gives eigenvalue +1, '1' gives -1.
                # cf = conversion_factor(grouping, observable)
                value *= -1 if outcome[i] == '0' else 1
        expectation += value * prob
    return expectation




def zne_folding(circuit, noise_factor):
    """
    Add controlled noise to a quantum circuit for zero noise extrapolation.
    
    Args:
        circuit (QuantumCircuit): The original quantum circuit
        noise_factor (int): The factor by which to increase the noise (1, 3, 5, etc.)
    Returns:
        QuantumCircuit: The circuit with added noise
    """
    if noise_factor <= 0:
        raise ValueError("Noise factor must be positive")
    
    if noise_factor == 1: 
        # No additional noise
        return circuit.copy()
    
    # For two-qubit gate folding, we need to:
    # 1. Extract the two-qubit gates
    # 2. Apply folding to just those gates
    # 3. Reconstruct the circuit
    
    # Create a new circuit with the same qubits and classical bits
    noisy_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    
    # Number of folds to apply
    num_folds = (noise_factor - 1) // 2
    
    # Process each gate in the original circuit
    for inst, qargs, cargs in circuit.data:
        # Check if this is a two-qubit gate (has 2 quantum arguments)
        if len(qargs) == 2:
            # This is a two-qubit gate, apply folding
            # First, add the original gate
            noisy_circuit.append(inst, qargs, cargs)
            
            # Then add the folded gates (inverse followed by original, repeated num_folds times)
            for _ in range(num_folds):
                if inst.name == 'ms':
                    ## from qiskit_ionq, which does not implement inverse()
                    ## So for ms(phi1,phi2,theta), the inverse is ms(phi1+0.5,phi2,theta)
                    from qiskit_ionq.ionq_gates import MSGate
                    phi1_ms,phi2_ms,theta_ms = inst.params
                    noisy_circuit.append(MSGate(phi1_ms+0.5,phi2_ms,-theta_ms), qargs, cargs)
                elif inst.name == 'zz':
                    from qiskit_ionq.ionq_gates import ZZGate
                    theta_zz = inst.params[0]
                    noisy_circuit.append(ZZGate(-theta_zz), qargs, cargs)
                else:
                    noisy_circuit.append(inst.inverse(), qargs, cargs)
                # Add original gate again
                noisy_circuit.append(inst, qargs, cargs)
        else:
            # This is a one-qubit gate or a gate with different structure
            # Just add it without folding
            noisy_circuit.append(inst, qargs, cargs)
    return noisy_circuit

    

def zne_half(circuit, noise_factor):
    """
    Add controlled noise to a quantum circuit for zero noise extrapolation. Only after barrier
    
    Args:
        circuit (QuantumCircuit): The original quantum circuit
        noise_factor (int): The factor by which to increase the noise (1, 3, 5, etc.)
    Returns:
        QuantumCircuit: The circuit with added noise
    """
    if noise_factor <= 0:
        raise ValueError("Noise factor must be positive")
    
    if noise_factor == 1: 
        # No additional noise
        return circuit.copy()
    
    # For two-qubit gate folding, we need to:
    # 1. Extract the two-qubit gates
    # 2. Apply folding to just those gates
    # 3. Reconstruct the circuit
    
    # Create a new circuit with the same qubits and classical bits
    noisy_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    
    # Number of folds to apply
    num_folds = (noise_factor - 1) // 2
    start_to_fold = False
    
    # Process each gate in the original circuit
    for inst, qargs, cargs in circuit.data:
        # Check if this is a two-qubit gate (has 2 quantum arguments)
        if inst.name == 'barrier':
            # If we hit a barrier, we start folding from the next gate
            start_to_fold = True
        elif len(qargs) == 2 and start_to_fold:
            # This is a two-qubit gate, apply folding
            # First, add the original gate
            noisy_circuit.append(inst, qargs, cargs)
            # Then add the folded gates (inverse followed by original, repeated num_folds times)
            for _ in range(num_folds):
                # Add inverse gate
                if inst.name == 'ms':
                    ## from qiskit_ionq, which does not implement inverse()
                    ## So for ms(phi1,phi2,theta), the inverse is ms(phi1+0.5,phi2,theta)
                    from qiskit_ionq.ionq_gates import MSGate
                    phi1_ms,phi2_ms,theta_ms = inst.params
                    noisy_circuit.append(MSGate(phi1_ms+0.5,phi2_ms,theta_ms), qargs, cargs)
                elif inst.name == 'zz':
                    from qiskit_ionq.ionq_gates import ZZGate
                    theta_zz = inst.params[0]
                    noisy_circuit.append(ZZGate(-theta_zz), qargs, cargs)
                else:
                    noisy_circuit.append(inst.inverse(), qargs, cargs)
                # Add original gate again
                noisy_circuit.append(inst, qargs, cargs)
        else:
            # This is a one-qubit gate or a gate with different structure
            # Just add it without folding
            noisy_circuit.append(inst, qargs, cargs)
    return noisy_circuit




def add_noise_to_circuit(circuit, noise_factor=1, noise_type='folding', seed=7):
    """
    Wrapper for adding controlled noise to a quantum circuit for zero noise extrapolation.
    
    Args:
        circuit (QuantumCircuit): The original quantum circuit
        noise_factor (int): The factor by which to increase the noise (1, 3, 5, etc.)
        noise_type (str): The type of noise to add ('folding', 'half', etc.)
        
    Returns:
        QuantumCircuit: The circuit with added noise
    """
    if noise_type == 'folding':
        return zne_folding(circuit, noise_factor)
    elif noise_type == 'half':
        return zne_half(circuit, noise_factor)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Supported types are 'folding' and 'half'.")






def commuting_circs(pauli_obs, state_prep_circ, add_measure=True, inverse_pauli=False, noise_factors=[1], noise_type='folding'):
    """Create commuting circuits from a Pauli operator and a state preparation circuit.
    
    Args:
        pauli_obs (list[str]): a list of Pauli strings to be measured. Example: ['IIIIIIIXYXYI','IXZYIIIIXZYI',...]
        state_prep_circ (QuantumCircuit): The state preparation circuit. Example:
        add_measure (bool): Whether to add measurement gates to the circuit.
        inverse_pauli (bool): Whether to invert the Pauli string.
        noise_factor (int): The factor by which to increase the noise for ZNE (1, 3, 5, etc.)
        noise_type (str): The type of noise to add ('folding', 'cnot-pair', etc.)
    Returns:
        res_circs (dict): A dictionary of commuting circuits with the Pauli string as the key. Example:
            {
            'IIIIIIIXYXYI':QuantumCircuit,
            'IXZYIIIIXZYI':QuantumCircuit,
            ...
            }
    """
    res_circs = {}
    for pobs in pauli_obs:
        res_circs[pobs] = []
        for nf in noise_factors:
            temp_circ =  add_noise_to_circuit(state_prep_circ, nf, noise_type)
            for pii in range(len(pobs)):
                if inverse_pauli:
                    pii = len(pobs) - 1 - pii
                if pobs[pii] == 'X':
                    temp_circ.h(pii)
                elif pobs[pii] == 'Y':
                    temp_circ.sdg(pii)
                    temp_circ.h(pii)
                elif pobs[pii] == 'Z' or pobs[pii] == 'I':
                    pass
            if add_measure:
                temp_circ.measure_all()
            res_circs[pobs].append(temp_circ)
    return res_circs
    

def str_array_to_prob_array(str_array):
    """Convert a list of bitstrings to a list of probabilities.
    
    Args:
        str_array (list[str]): A list of bitstrings.
    Returns:
        list[float]: A list of probabilities.
    """
    n_qubits = len(str_array[0])
    n_samples = len(str_array)
    prob_array = [0]*(2**n_qubits)
    ## Count the number of times each bitstring appears
    for i in range(2**n_qubits):
        pos = int(str_array[i], 2)
        prob_array[pos] += 1
    ## Normalize the probabilities
    for i in range(2**n_qubits):
        prob_array[i] = prob_array[i]/n_samples
    return prob_array


def freq_dict_to_prob_array(freq_dict, n_qubits, endien_switch = False, first_n=None):
    """Convert a frequency dictionary to a probability array.
    
    Args:
        freq_dict (dict): A dictionary of frequencies. key is the base 10 integer, value is the frequency.
    Returns:
        list[float]: A list of probabilities.
    """
    import re
    n_samples = sum(list(freq_dict.values()))
    arr_size = 2**n_qubits
    prob_array = [0]*arr_size
    for key, value in freq_dict.items():
        if type(key) == tuple: ## Quantinuum returns like Counter{(0,0):300, (1,0):500}
            # Convert tuple to binary string
            binary_str = ''.join(str(bit) for bit in key)
            # if first_n is not None:
            #     binary_str = binary_str[:first_n] ## Seems like Quantinuum repeat measurements for 3 times
            if endien_switch:
                prob_array[int(binary_str[::-1], 2)] += value/n_samples
            else:
                prob_array[int(binary_str, 2)] += value/n_samples
        elif re.fullmatch('[01]+', key):
            if endien_switch:
                prob_array[int(key[::-1], 2)] = value/n_samples
            else:
                prob_array[int(key, 2)] = value/n_samples
        else:
            if endien_switch:
                prob_array[arr_size - 1 - int(key)] = value/n_samples
            else:
                prob_array[int(key)] = value/n_samples
    return prob_array


def find_commuting_key(measurement_dict, ham_pauli_str):
    """Find the key in the measurement dictionary that commutes with the given Pauli string.
    
    Args:
        measurement_dict (dict): A dictionary of measurement results. Example:
            {
            'IIIIIIIXYXYI': [0, 0, 0.1, 0.2, 0,0 ,... ],
            'IXZYIIIIXZYI': [0, 0.1, 0.1, 0.2, 0,0 ,... ],
            ...
            }
        ham_pauli_str (str): The Pauli string to check for commutation.
    Returns:
        str: The key in the measurement dictionary that commutes with the given Pauli string.
    """
    qis_pauli_key = qiskit.quantum_info.Pauli(ham_pauli_str)
    for key in measurement_dict.keys():
        if qiskit.quantum_info.Pauli(key).commutes(qis_pauli_key):
            return key
    return None

def est_exp_sampler(grouping_pauli_dict, grouping_coeff_dict, prob_dict, return_var=False, num_shots=0,debug_state = None):
    """Estimate the expectation using commuting group, using bitstring measurement results.
    
    Args:
        grouping_pauli_dict (dict): A dictionary of commuting groups with the first Pauli operator in each group as the key.
        grouping_coeff_dict (dict): A dictionary of coefficients for the commuting groups.
        prob_dict (dict): A dictionary of measurement results.

        grouping_pauli_dict example:
        {
        'IIIIIIIXYXYI': ['IIIIIIIXYXYI','IIIIIIIYYYYI','IIIIIIIXXYYI'],
        'IXZYIIIIXZYI': ['IXZYIIIIXZYI','IYZYIIIIYZYI','IXZXIIIIYZYI','IYZXIIIIXZYI'],
        ...
        }
        grouping_coeff_dict example:
        {
        'IIIIIIIXYXYI': [1.0, 1.0, 1.0],
        'IXZYIIIIXZYI': [1.0, 1.0, 1.0, 1.0],
        ...
        }
        prob_dict example:
        {
        'IIIIIIIXYXYI': [0, 0, 0.1, 0.2, 0,0 ,... ]  ## sum to 1
        'IXZYIIIIXZYI': [0, 0.1, 0.1, 0.2, 0,0 ,... ]  ## sum to 1
        ...
        }

    Returns:
        res_exp (float): The estimated expectation of the Pauli operator.
    """
    if return_var and num_shots == 0:
        raise Exception("Please provide the number of shots.")
    res_exp = 0
    res_var = 0
    for pkey, parr in grouping_pauli_dict.items():
        # key_in_meas = find_commuting_key(prob_dict, pkey)
        # key_in_meas = pkey
        if pkey not in prob_dict.keys():
            print(f"Skipping Pauli key {pkey} as it is not in the measurement results.")
            continue
        prob_arr = prob_dict[pkey]
        coeff_arr = grouping_coeff_dict[pkey]
        for pi in range(len(parr)):
            pstr = parr[pi]
            single_exp = expectation_from_probabilities(prob_arr, pstr)
            single_var = 1-single_exp**2
            # Debug output if debug_state is provided
            if debug_state is not None:
                single_exp_debug = debug_state.expectation_value(SparsePauliOp(pstr[::-1])).real
                if pstr.count('Z') == 1 and pstr.count('X') == 0 and pstr.count('Y') == 0:
                    single_exp_debug = -single_exp_debug
                if abs(single_exp_debug - single_exp) > 1e-12:
                    print(pkey, pstr, Pauli(pkey).commutes(Pauli(pstr)), single_exp, single_exp_debug)
            # Add to total expectation value
            res_exp += coeff_arr[pi] * single_exp
            if return_var:
                res_var += (np.real(coeff_arr[pi]**2) * single_var)/num_shots
    if abs(res_exp.imag) < 1e-12:
        res_exp = res_exp.real

    if return_var:
        return res_exp,res_var
    else:
        return res_exp


# def exp_from_sampler(ham_op:qiskit.quantum_info.SparsePauliOp, prob_dict):
#     """Estimate the expectation value of a Hamiltonian operator using measurement results.
    
#     Args:
#         ham_op (qiskit.quantum_info.SparsePauliOp): The Hamiltonian operator to be measured.
#         prob_dict (dict): A dictionary of measurement results. Example:
#             {
#             'IIIIIIIXYXYI': [0, 0, 0.1, 0.2, 0,0 ,... ],
#             'IXZYIIIIXZYI': [0, 0.1, 0.1, 0.2, 0,0 ,... ],
#             ...
#             }
#     Returns:
#         float: The estimated expectation value of the Hamiltonian operator.
#     """
#     res_exp = 0
#     coeff_abs_sum = 0
#     missed_coeff_abs_sum = 0
#     used_count = {k:0 for k in prob_dict.keys()}
#     for pp,cc in zip(ham_op.paulis, ham_op.coeffs):
#         pp = str(pp)
#         coeff_abs_sum += abs(cc)
#         key_in_meas = find_commuting_key(prob_dict, pp)
#         used_count[key_in_meas] += 1
#         print(f"  - Pauli: {pp}, Key in measurement: {key_in_meas}")
#         if key_in_meas is not None:
#             prob_arr = prob_dict[key_in_meas]
#             single_exp = expectation_from_probabilities(prob_arr, pp)
#         else:
#             missed_coeff_abs_sum += abs(cc)
#         res_exp += cc * single_exp
#     if abs(res_exp.imag) < 1e-12:
#         res_exp = res_exp.real
#     print(used_count)
#     print(f"  - Coefficients sum: {coeff_abs_sum}, Missed coefficients sum: {missed_coeff_abs_sum}, Ratio: {missed_coeff_abs_sum/(coeff_abs_sum+1e-12):.3f}")
#     return res_exp



def extrapolate_to_zero_noise(noise_factors, expectation_values, exp_vars = [], num_shots=1024, plot_alpha=0.05, 
                              extrapolation_method='linear', 
                              plot=False, plot_title=None, 
                              plot_horizontal_line=None,
                              plot_horizontal_line_legend=None,
                              intercept_error_bar=False,
                              return_plot=False):
    """
    Extrapolate expectation values to the zero-noise limit.
    
    Args:
        noise_factors (list): List of noise factors used (e.g., [1, 3, 5, 7])
        expectation_values (list): Corresponding expectation values measured
        extrapolation_method (str): Method for extrapolation ('linear', 'quadratic', 'exponential')
        plot (bool): Whether to generate and show a plot of the extrapolation
        
    Returns:
        float: The extrapolated zero-noise expectation value
    """
    if len(exp_vars) > 0 and len(exp_vars) != len(expectation_values):
        raise ValueError("Variance of expectation values and expectation values must have the same length")
    
    if len(noise_factors) != len(expectation_values):
        raise ValueError("Noise factors and expectation values must have the same length")
    
    if len(noise_factors) < 2:
        raise ValueError("Need at least two data points for extrapolation")

    if plot_horizontal_line is not None and plot_horizontal_line_legend is None:
        raise ValueError("If plot_horizontal_line is provided, plot_horizontal_line_legend must also be provided")
    
    # Define fitting functions
    def linear_func(x, a, b):
        return a * x + b
    
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    # Convert to numpy arrays
    x_data = np.array(noise_factors)
    y_data = np.array(expectation_values)
    
    # Perform the fit based on the selected method
    if extrapolation_method == 'linear':
        popt, pcov = curve_fit(linear_func, x_data, y_data)
        zero_noise_value = linear_func(0, *popt)
        fit_func = linear_func
    # elif extrapolation_method == 'quadratic':
    #     if len(noise_factors) < 3:
    #         raise ValueError("Need at least three data points for quadratic extrapolation")
    #     popt, pcov = curve_fit(quadratic_func, x_data, y_data)
    #     zero_noise_value = quadratic_func(0, *popt)
    #     fit_func = quadratic_func
    # elif extrapolation_method == 'exponential':
    #     if len(noise_factors) < 3:
    #         raise ValueError("Need at least three data points for exponential extrapolation")
    #     popt, pcov = curve_fit(exponential_func, x_data, y_data)
    #     zero_noise_value = exponential_func(0, *popt)
    #     fit_func = exponential_func
    else:
        raise ValueError(f"Unknown extrapolation method: {extrapolation_method}")
    
    # residuals = y_data- fit_func(x_data, *popt)
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((y_data-np.mean(y_data))**2)
    # r_squared = 1 - (ss_res / ss_tot)
    # print(f"RSS: {ss_res}")
    # print(f"R^2 value: {r_squared}")

    import statsmodels.api as sm
    linear_model = sm.WLS(y_data, sm.add_constant(x_data), weights=1/np.array(exp_vars))
    res_sp = linear_model.fit()
    slope = res_sp.params[1]
    intercept = res_sp.params[0]
    zero_noise_value = intercept
    ##
    r_squared = res_sp.rsquared
    intercept_se = res_sp.bse[0]
    slope_se = res_sp.bse[1]
    popt = res_sp.params[::-1]
    residuals = y_data- (slope * x_data + intercept)
    sse = np.sum(residuals**2)
    sst = np.sum((y_data-np.mean(y_data))**2)

    rmse = np.sqrt(sse/(len(x_data)-2))
    print(f"R^2 value: {r_squared}")
    # print(f"Slope: {res_sp.slope} +- {slope_se}")
    # print(f"Intercept: {res_sp.intercept} +- {intercept_se}")
    print(f"Slope: {slope} +- {slope_se}")
    print(f"Intercept: {intercept} +- {intercept_se}")
    print(f"Residuals: {residuals}")
    print(f"Sum of the squared redisuals: {sse}")
    print(f"Sum of the squared total: {sst}")
    print(f"RMSE: {rmse}")
    
    # Generate plot if requested
    if plot:
        plt.figure(figsize=(8, 6))
        if len(exp_vars) > 0:
            # from scipy import stats
            from scipy.stats import t
            t_crit = t.ppf(1 - plot_alpha/2, df=num_shots-2)
            CI_bound = t_crit * np.sqrt(np.array(exp_vars))
            print("95% CI:", CI_bound)
            plt.errorbar(x_data, y_data, yerr=CI_bound, fmt='o', color='blue', label='Measured values', capsize=3)
        else:
            plt.scatter(x_data, y_data, color='blue', label='Measured values')
        # Plot the fit line
        x_fit = np.linspace(0, max(x_data), 100)
        y_fit = fit_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label=f'{extrapolation_method.capitalize()} fit(r²={r_squared:.5f})')

        if plot_horizontal_line is not None and plot_horizontal_line_legend is not None:
            if type(plot_horizontal_line) is list:
                for line,leg in zip(plot_horizontal_line, plot_horizontal_line_legend):
                    plt.axhline(y=line, color='orange', linestyle='--', 
                                label=f"{leg}: {line:.6f}")
            else:
                plt.axhline(y=plot_horizontal_line, color='orange', linestyle='--', 
                            label=f"{plot_horizontal_line_legend}: {plot_horizontal_line:.6f}")
            

        # Mark the extrapolated zero-noise value
        intercept_CI_bound = 0
        if intercept_error_bar:
            from scipy.stats import t
            SE_manual = intercept_se
            t_crit = t.ppf(1 - plot_alpha/2, df=len(x_data)-2)
            intercept_CI_bound = t_crit * SE_manual
            print("95% CI for intercept:", intercept_CI_bound)
            plt.errorbar([0], [zero_noise_value], yerr=intercept_CI_bound, color='green', fmt='o', capsize=3, markersize=8,
                    label=f'Zero-noise estimate: {zero_noise_value:.6f}')
        else:
            plt.scatter([0], [zero_noise_value], color='green', s=100,
                    label=f'Zero-noise estimate: {zero_noise_value:.6f}')
        
        plt.xlabel('Noise Factor')
        plt.ylabel('Expectation Value')
        if plot_title is not None:
            plt.title(plot_title)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.xticks([0]+noise_factors)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=zero_noise_value-intercept_CI_bound-0.005 if plot_horizontal_line is None else min(plot_horizontal_line-0.005, zero_noise_value-intercept_CI_bound-0.005))
        plt.legend()
        plt.tight_layout()
        return_fig = plt.gcf()
    if return_plot:
        return zero_noise_value, popt, return_fig
    return zero_noise_value, popt





def intercept_with_uncertainty(x, y, alpha=0.05):
    """
    Calculate the intercept and its uncertainty for y = a*x + b.

    Parameters:
    x (list or array): Independent variable data.
    y (list or array): Dependent variable data.
    alpha (float): Significance level for confidence interval (default 0.05 for 95%).

    Returns:
    dict: Contains intercept estimate, SciPy SE of intercept, manual SE, and 95% CI.
    """
    import numpy as np
    from scipy import stats
    from scipy.stats import t

    x = np.array(x)
    y = np.array(y)
    n = len(x)

    # Use SciPy's linregress to get slope, intercept, and SciPy intercept SE
    result = stats.linregress(x, y)
    intercept = result.intercept
    intercept_se_scipy = result.intercept_stderr

    # 95% confidence interval (using manual SE)
    t_crit = t.ppf(1 - alpha/2, df=n-2)
    CI_lower = intercept - t_crit * intercept_se_scipy
    CI_upper = intercept + t_crit * intercept_se_scipy

    return {
        'intercept': intercept,
        'intercept_se': intercept_se_scipy,
        'uncertainty':t_crit * intercept_se_scipy,
        'CI': (CI_lower, CI_upper)
    }





if __name__ == '__main__':
    ##
    import qiskit
    import qiskit.quantum_info as qi
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from _gcim_utilis import parse_hamiltonian
    import scipy.sparse.linalg as ssl
    import scipy.linalg as sl
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit import transpile
    from qiskit.primitives import StatevectorEstimator
    ##

    n_orb = 6
    n_a = 3
    n_b = 3
    ducc_lvl = 3
    results_file = 'printout_bz66ducc3.txt'

    mol_name =  f'Benzene{n_a+n_b}{n_orb}DUCC{ducc_lvl}'
    file_path = f'Input_Data/bz{n_a+n_b}{n_orb}ducc{ducc_lvl}.hamil'
    print("  - Number of electrons:", n_a + n_b)
    print("  - Number of orbitals:", n_orb)
    print("  - DUCC:", ducc_lvl)
    print("  - Target molecule:", mol_name)
    print("  - Input file:", file_path)

    fermi_ham = parse_hamiltonian(file_path, n_orb, interleaved=False, return_type = 'qiskit')
    nwqsim_dict = extract_nwqsim_output(results_file, inverse_pauli=False)

    # mapper = InterleavedQubitMapper(JordanWignerMapper())
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermi_ham)
    # for pi in range(len(qubit_op.paulis)):
    #     qubit_op.paulis[pi] = qiskit.quantum_info.Pauli( qubit_op.paulis[pi][::-1] )

    selected_iter = 1
    vqe_circ, symbol_params = create_vqe_circ(nwqsim_dict['operators'][:selected_iter+1],
                                                n_orb*2, n_a+n_b, interleaved=False)

    ## Optimize parameters
    def cost_func(parameters):
        """Compute the energy for given parameters."""
        estimator = StatevectorEstimator()
        job = estimator.run([(vqe_circ, qubit_op, parameters)])
        return float( job.result()[0].data.evs )

    num_vars = len(nwqsim_dict['parameters'][:selected_iter+1])
    optimizer = COBYLA(maxiter=5000, disp=True)
    result = optimizer.minimize(cost_func, nwqsim_dict['parameters'][:selected_iter+1])

    basis_gates = ['u', 'cx']
    trans_vqe_opt = transpile(vqe_circ.assign_parameters(result.x), basis_gates=basis_gates, optimization_level=2)
    print(trans_vqe_opt.count_ops(), cost_func(result.x), nwqsim_dict['energies'][selected_iter])

    ##
    pauli_dict, coeff_dict = grouped_pauli_in_dict(qubit_op, use_qwc=False) ## grouping paulis in the Hamiltonian and separate Pauli strings and Coefficients
    circ_dict = commuting_circs(list(pauli_dict.keys()), trans_vqe_opt, inverse_pauli = False) ## Form circuits for each commuting group

    ##
    # num_shots = 10000000
    # from qiskit.primitives import StatevectorSampler
    # sampler = StatevectorSampler()
    # meas_prob_dict = {}
    # for key, circ in circ_dict.items():
    #     temp_circ = circ.copy()
    #     temp_circ.measure_all()
    #     meas_quasi_res = sampler.run(temp_circ, shots=num_shots).result().quasi_dists[0]
    #     meas_prob_dict[key] = freq_dict_to_prob_array(meas_quasi_res, len(key))
    meas_prob_dict = {}
    for key, circ in circ_dict.items():
        meas_prob_dict[key] = qiskit_normal_order_switch_vec( qi.Statevector.from_instruction(circ).probabilities(decimals=16)  )
    ##

    final_exp = est_exp_sampler(pauli_dict, coeff_dict, meas_prob_dict) ## compute the expectation value and sum them up
    print(final_exp, cost_func(result.x), qi.Statevector.from_instruction(vqe_circ.assign_parameters(result.x)).expectation_value(qubit_op))