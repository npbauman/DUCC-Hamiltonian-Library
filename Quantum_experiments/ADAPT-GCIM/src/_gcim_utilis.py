import re
import os
import openfermion
import numpy
# import qiskit
from qiskit_nature.second_q.operators import FermionicOp

## https://github.com/pnnl/NWQ-Sim/blob/main/vqe/src/hamiltonian.cpp
HAM_XACC_PATTERN =  r"\(\s*([\d\.e\+-]+),\s*([\d\.]+)\)([\d^\s]+){0,1}" ## https://github.com/pnnl/NWQ-Sim/blob/5aae11a189f546bf9ec7e74e03416bf752fb221d/vqe/src/hamiltonian.cpp#L15C30-L15C91

## Orbital mapping
## The recevied order is a0, a1, ..., aN, b0, b1, ..., bN
## But we only handle a0, b0, a1, b1, ..., aN, bN
## 0 - > 0
## 1 -> 2
## 2 -> 4
## ...
## N/2 -> 1
## M/2+1 -> 3
def a1a2_to_a1b1_mapping(n_qubits):
    ## Qubit i in a1a2 order is mapped to perm_arr[i] in a1b1 order
    perm_arr = list(range(0, n_qubits, 2)) + list(range(1, n_qubits, 2))
    # mapping_dict = {i:perm_arr[i] for i in range(n_qubits)}
    return perm_arr 



def parse_hamiltonian(ham_path, n_orbs, use_interleaved=True, return_type = 'of'):
    Ham = None
    if not os.path.isfile(ham_path):
        raise ValueError(f"The {ham_path} is not a valid file")
    
    permu_arr = a1a2_to_a1b1_mapping(n_orbs*2)
    with open(ham_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

        # Process each line in the file
        for line in lines:
            match = re.search(HAM_XACC_PATTERN, line)

            coeff = float(match.group(1)) if float(match.group(2)) == 0 else float(match.group(1)) + float(match.group(2))*1j
            opstr = match.group(3) if match.group(3) else ()
            ## Openfermion
            if return_type.lower() == 'of' or return_type.lower() == 'openfermion':
                if match.group(3) and use_interleaved:
                    ##
                    orb_nums = re.findall(r'\d+', opstr) ## 6^ 7^ 11 6 -> ['6', '7', '11', '6']
                    orb_nums = [str(permu_arr[int(i)]) for i in orb_nums] ## permute the orbitals
                    ## form back the formatted string
                    perm_oplist = []
                    for i in range(len(orb_nums)):
                        perm_oplist.append(orb_nums[i])
                        if i < len(orb_nums)*0.5:
                            perm_oplist.append('^')
                        if i < len(orb_nums)-1:
                            perm_oplist.append(' ')
                    opstr = ''.join(perm_oplist)

                if Ham is None:
                    Ham = openfermion.FermionOperator(opstr, coeff)
                else:
                    Ham += openfermion.FermionOperator(opstr, coeff)
            ## Qiskit
            elif return_type.lower() == 'qiskit':
                if match.group(3) and use_interleaved:
                    ##
                    orb_nums = re.findall(r'\d+', opstr) ## 6^ 7^ 11 6 -> ['6', '7', '11', '6']
                    orb_nums = [str(permu_arr[int(i)]) for i in orb_nums] ## permute the orbitals
                    ## form back the formatted string
                    perm_oplist = []
                    for i in range(len(orb_nums)):
                        if i < len(orb_nums)*0.5:
                            perm_oplist.append('+_')
                        else:
                            perm_oplist.append('-_')
                        perm_oplist.append(orb_nums[i])
                        if i < len(orb_nums)-1:
                            perm_oplist.append(' ')
                    opstr = ''.join(perm_oplist)

                if Ham is None:
                    Ham = FermionicOp({opstr:coeff}, num_spin_orbitals=n_orbs*2)
                else:
                    Ham += FermionicOp({opstr:coeff}, num_spin_orbitals=n_orbs*2)
            else:
                raise ValueError(f"Unknown return type {return_type}")
    return Ham

##############


import scipy
import numpy as np
import scipy.sparse as sp
# import scipy.sparse.linalg as ssl
import scipy.linalg as sl
import pandas as pd

def HFstate(nele, nq):
    hf_str = '1'*nele + '0'*(nq - nele)
    hf = np.zeros(2**(nq))
    hf[int(hf_str,2)] = 1
    return hf

def save_eigens(eigenvals, eigenvecs, data_folder, exp_name):
    df1 = pd.DataFrame(eigenvals.reshape(-1, len(eigenvals)))
    df2 = pd.DataFrame(eigenvecs)
    df = pd.concat([df1, df2])
    df.to_csv(data_folder+exp_name+'.csv', header=None, index=False)

def gcm_res_validation(input_dict, gcm_file_address):
    mole_name = input_dict['mole_name']
    iter_num = input_dict['iter_num']
    nele = input_dict['nele']
    purt = input_dict['purt']
    print("\n---------------------------------------")
    print(mole_name, "validation begins")

    ## Extract the basis vectors |phi_j> and the eigenvector corresponds to the lowest energy vector v from GCM
    # gcm_folder = 'Output_Data/ADAPT-GCM/GCM_'
    # fix = '_tp4/'
    # gcm_file_address = gcm_folder + mole_name + fix

    mole_ham = sp.load_npz(gcm_file_address+f'MoleHam.npz')
    nq = int( np.log2(mole_ham.shape[1]) )
    hf_vec = HFstate(nele, nq)

    ##
    try:
        basis_mat = np.matrix( np.load(gcm_file_address+f'IterMats/IterAG{iter_num}_basis_mat.npy') )
        oldH = np.load(gcm_file_address+f'IterMats/IterAG{iter_num}_H.npy')
        oldS = np.load(gcm_file_address+f'IterMats/IterAG{iter_num}_S.npy')
    except:
        compressed = np.load(gcm_file_address+f'IterMats/IterAG{iter_num}.npz')
        basis_mat = np.matrix(compressed['iter_basis_mat'])
        oldH = compressed['gcmH']
        oldS = compressed['gcmS']
    ##
    basis_mat_sp = sp.csc_matrix(basis_mat)
    newH = basis_mat_sp.conjugate().transpose() @ mole_ham @ basis_mat_sp
    # newH_orth = basis_mat_orth_sp.conjugate().transpose() @ mole_ham @ basis_mat_orth_sp

    # mole_eval, mole_evec = sl.eigh(mole_ham)
    true_ev, low_evec = scipy.sparse.linalg.eigsh(mole_ham,1,which='SA',v0=hf_vec)
    true_ev = true_ev[0]
    print("   True lowest eigenvalue:", true_ev)

    ## verification
    print("   Verification of the right basis matrix:", np.linalg.norm(oldH - newH))

    ## Eigenvalues and eigenvectors
    eval, evec = sl.eigh(oldH, oldS+np.eye(oldS.shape[0])*purt)
    lowest_ind = np.argmin(eval)
    gcm_lowest_eval = eval[lowest_ind]
    gcm_lowest_evec = evec[:,lowest_ind]
    print("   Verification of eigenvectors:", np.linalg.norm(oldH.dot(gcm_lowest_evec) - gcm_lowest_eval * oldS.dot(gcm_lowest_evec) ))
    # print("   Verification of eigenvectors:", np.linalg.norm(newH_orth.dot(gcm_lowest_evec) - gcm_lowest_eval * gcm_lowest_evec ))
    print("   Lowest eigenvalue error:", np.abs(gcm_lowest_eval - true_ev))
    # print("   GCM Eigenvector norm:", np.linalg.norm(gcm_lowest_evec, ord=2))
    save_eigens(eval, evec, gcm_file_address, mole_name)

    ## Compute normalization factor
    nf = 0
    for ii in range(basis_mat.shape[1]):
        for jj in range(basis_mat.shape[1]):
            nf += gcm_lowest_evec[ii]*gcm_lowest_evec[jj]*( np.matrix(basis_mat[:,ii]).H.dot(basis_mat[:,jj]) )
    nf = nf[0,0]
    print('   nf=', nf)
    ##


    ## Compute GCM state $\ket{\psi_{GCM}} = \sum_j v_j \ket{\phi_{j}}$ up to a normalization
    gcm_vec = basis_mat[:,0] * gcm_lowest_evec[0]
    for ii in range(1, basis_mat.shape[1]):
        gcm_vec += basis_mat[:,ii] * gcm_lowest_evec[ii]
    # gcm_vec = basis_mat_orth[:,0] * gcm_lowest_evec[0]
    # for ii in range(1, basis_mat_orth.shape[1]):
    #     gcm_vec += basis_mat_orth[:,ii] * gcm_lowest_evec[ii]
    norm_gcm_vec = np.linalg.norm(gcm_vec, ord=2)
    print("   GCM vector norm:", norm_gcm_vec)
    gcm_state = gcm_vec / norm_gcm_vec
    # gcm_state = gcm_vec / nf
    if np.linalg.norm(gcm_state - low_evec, ord=2) > 1:
        gcm_state = -gcm_state

    # print("First element", gcm_vec[1:5], low_evec[1:5])

    ## Check with exact eigenvectors
    overlap = np.abs(gcm_state.conjugate().T.dot(low_evec).flatten()[0,0])**2
    # overlap = np.abs(gcm_state.conjugate().T.dot(low_evec).flatten()[0])**2
    print('\n   Eigenvalue = {:8.7f}  '.format(true_ev), 
        "1 - |<GCM|Exact>|^2=", 1-overlap, "||GCM - Exact||_2=", np.linalg.norm(gcm_state - low_evec, ord=2)) 
    return np.array(gcm_vec).flatten(), np.array(low_evec).flatten()

# ## First good iteration
# H620Bare_dict = {'mole_name': 'H620','iter_num': 80,'nele': 6,'purt': 1e-12}
# H620O6DUCC3_dict = {'mole_name': 'H620O6DUCC3','iter_num': 74,'nele': 6,'purt': 1e-12}
# H630O6DUCC3_dict = {'mole_name': 'H630O6DUCC3','iter_num': 71,'nele': 6,'purt': 1e-12}


# _,_ = gcm_res_validation(H620Bare_dict, 'Output_Data/ADAPT-GCM/GCM_'+H620Bare_dict['mole_name']+ '_tp4/')
# _,_ = gcm_res_validation(H620O6DUCC3_dict, 'Output_Data/ADAPT-GCM/GCM_'+H620O6DUCC3_dict['mole_name']+ '_tp4/')
# _,_ = gcm_res_validation(H630O6DUCC3_dict, 'Output_Data/ADAPT-GCM/GCM_'+H630O6DUCC3_dict['mole_name']+ '_tp4/')



############




from qiskit.quantum_info import Pauli, SparsePauliOp
# from qiskit.quantum_info import Statevector
# from qiskit_algorithms import TimeEvolutionProblem
# from qiskit.quantum_info import Operator
# from qiskit_algorithms.time_evolvers.trotterization import TrotterQRTE

def format_qubit_operator(qubit_op, n_qubits, real_coeff = False):
    # qubit_op: openfermion.ops.operators.qubit_operator.QubitOperator
    # n_qubits: int
    # return e.g.,  {'YZXIIIII': 0.5j, 'XZYIIIII': -0.5j}
    # No need to inverse, since if you convert back to matrix, it follows what is printed off
    imaginary_flag = False
    coefs = []
    paulis = []
    for term, term_coef in qubit_op.terms.items():
        # Create a list of 'I' for all qubits
        formatted_term = ['I'] * n_qubits
        for qubit, pauli in term:
            formatted_term[qubit] = pauli
        if real_coeff:
            if term_coef != numpy.imag(term_coef):
                imaginary_flag = True
            term_coef = numpy.imag(term_coef)
        coefs.append(term_coef)
        paulis.append(''.join(formatted_term))
    return coefs, paulis, imaginary_flag


def generate_qiskit_ansatzs(fermi_op_list, nq):
    ## fermi_op_list: a list of openfermion.ops.operators.qubit_operator.QubitOperator
    ## nq: number of qubits

    of_jw_ops = numpy.array([openfermion.transforms.jordan_wigner(i) for i in fermi_op_list])
    qiskit_op_array = []
    for jwp in of_jw_ops:
        op_coefs, op_paulis, if_imag = format_qubit_operator(jwp, nq, real_coeff = False)
        qiskit_op = SparsePauliOp(op_paulis, coeffs=1*numpy.array(op_coefs) )
        qiskit_op_array.append(qiskit_op)
    return qiskit_op_array


## Trotter functions
def opI(n): # identity operator
    return SparsePauliOp(Pauli('I'*n))


def expn_convert(coef, power_op):
    # for exp(itP) and P = IIXZ, t is coef, P is a power_op and must be a single Pauli term
    # P is qiskit.opflow.PauliOp
    # t is int or float
    # expand to cos(coef)Id + isin(coef)P
    id_op = opI(power_op.num_qubits)
    res_op = (numpy.cos(coef)*id_op) + (1j*numpy.sin(coef)*SparsePauliOp(power_op))
    return res_op


def expn_trotter(power_opm:SparsePauliOp, steps=1, atol = 1e-12, order = 1):
    # Find trottered exp(iA) where A = \sum_{i = 1}^m s_i P_i
    # A is qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp
    # steps is number of Trotter steps
    # Default Suzuki-trotter order is 1
    
    ## Separate coefficients and Pauli operators
    A_coefs, A_paulis = -1j*power_opm.coeffs, power_opm.paulis ## expn_convert accept t not it
    # print(A_coefs)
    # print(A_paulis)
    ## Convert each term to cos(s)I + isin(s)P form
    converted_terms = []
    const = order*steps
    for i in range(len(A_coefs)):
        single_term = expn_convert(A_coefs[i]/const, A_paulis[i])
        converted_terms.append(single_term)
    ## Trotterize
    LHSop = converted_terms[0].copy()
    for j in range(1, len(converted_terms)):
        LHSop = (LHSop @ converted_terms[j]).simplify(atol=atol)
    if order == 1:
        one_term = LHSop.copy()
    elif order == 2:
        # raise Warning("Order 2 is not tested")
        ## Do 2nd-order trotter steps
        RHSop = converted_terms[-1].copy()
        for k in range(len(converted_terms)-2 ,-1, -1):
            RHSop = (RHSop @ converted_terms[k]).simplify(atol=atol)
        ## Multiply the resultant op by "steps" times
        one_term = (LHSop @ RHSop).simplify(atol=atol)
    else:
        raise ValueError("Order must be 1 or 2")
    sol = one_term.copy()
    for _ in range(steps-1):
        sol = (sol @ one_term).simplify(atol=atol)
    return sol

