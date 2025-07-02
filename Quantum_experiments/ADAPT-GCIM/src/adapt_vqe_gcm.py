import numpy as np # Muqing
import scipy
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import sys

import openfermion as of
from openfermion import *
from tVQE import *

import random

from timeit import default_timer as timer


# print("WARNING: the product basis computation is changed")







## ------------------- Classical computation for GCM matrices Starts -----------------------
def gcm_mats(hamiltonian, GCM_SINGLE_MATs, reference_ket, iter_coeffs, 
             prev_basis = None,
             make_orth = False):
    """Create H and S matrices from GCM methods

    Args:
        hamiltonian (_type_): _description_
        GCM_SINGLE_MATs (_type_): _description_
        reference_ket (_type_): _description_
        iter_coeffs (_type_): _description_
        make_orth (boolean): If orthogonalize the GCM basis so overlap matrix S is just an identity matrix. Default is False
        prev_basis : to pass necessary basis vectors from last iteration
    """
    # iter_coeffs is from previous VQE iteration, note that the 1st elemenet is the newly inserted 0 from
    # a few lines above this function called in adapt_vqe_gcm()
    ### Construct Basis
    num_basis = len(GCM_SINGLE_MATs)
    GCM_BASIS = {'R0': reference_ket.copy()}
    constant_t = np.pi/4
    print("Orthogonalization: ", make_orth)

    for k in range(len(GCM_SINGLE_MATs)):
        t = iter_coeffs[k] if iter_coeffs[k] != 0 else constant_t
        opmat = GCM_SINGLE_MATs[k]
        GCM_BASIS['+R'+str(k)] = ssl.expm_multiply(   (t*opmat), reference_ket  )
        GCM_BASIS['-R'+str(k)] = ssl.expm_multiply(  (-t*opmat), reference_ket  )

    # Final Product basis, only +/- in the newly-added element
    if len(GCM_SINGLE_MATs) > 1:
        prod_basis = reference_ket.copy()
        for i in range(num_basis-1, 0, -1):
            op = GCM_SINGLE_MATs[i]
            op_coef = iter_coeffs[i]
            prod_basis = ssl.expm_multiply(  (op_coef*op), prod_basis  )
        ## newly-added element
        last_op =  GCM_SINGLE_MATs[0]
        last_op_coef = constant_t
        GCM_BASIS['+R'+str(num_basis)] = ssl.expm_multiply(  (last_op_coef*last_op), prod_basis  )
        GCM_BASIS['-R'+str(num_basis)] = ssl.expm_multiply( (-last_op_coef*last_op), prod_basis  )
        
        if not make_orth:
            print("Passing out extra product bases and no orthgonalization")
            # Add product bases from previous iterations
            ri = 1
            bi = 0
            while bi < len(prev_basis):
                GCM_BASIS['+R'+str(num_basis+ri)] = prev_basis[bi]
                GCM_BASIS['-R'+str(num_basis+ri)] = prev_basis[bi+1]
                ri += 1
                bi += 2
            # Only the new product bases are carrid out to the next iteration
            prev_basis.append(GCM_BASIS['+R'+str(num_basis)])
            prev_basis.append(GCM_BASIS['-R'+str(num_basis)])
    
    ### Compute Matrices
    names = list(GCM_BASIS.keys())
    if not make_orth: # Usual GCM method
        H = np.zeros((len(names),len(names)), dtype=complex)
        S = np.zeros((len(names),len(names)), dtype=complex)
        basis_mat = np.zeros(   (GCM_BASIS[names[0]].shape[0], len(names))   , dtype=complex)
        for i in range(len(names)):
            basis_mat[:,i] = GCM_BASIS[ names[i] ].toarray().flatten()
            for j in range(i,len(names)):
                left_op  = GCM_BASIS[ names[i] ].transpose().conj()
                right_op = GCM_BASIS[ names[j] ]
                H[i,j] = left_op.dot(  hamiltonian.dot(right_op)  )[0,0]
                S[i,j] = left_op.dot(right_op)[0,0]
                if i != j:
                    H[j,i] = H[i,j].conj()
                    S[j,i] = S[i,j].conj()
    else: # Use orthogonalization
        # Create a matrix whose column is each GCM basis
        if len(GCM_SINGLE_MATs) > 1:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+4
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-4] = prev_basis
            basis_mat[:,num_orthbasis-4] = GCM_BASIS['+R'+str(0)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-3] = GCM_BASIS['-R'+str(0)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-2] = GCM_BASIS['+R'+str(num_basis)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['-R'+str(num_basis)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
        else:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+2
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-2] = prev_basis
            basis_mat[:,num_orthbasis-2] = GCM_BASIS['+R'+str(0)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['-R'+str(0)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix

        orth_basis_mat = sl.orth(basis_mat)
        # S = np.identity(orth_basis_mat.shape[1]) # trivial
        H = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex)
        S = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex) # We can check if S is identity
        # Create H from orthogonalized matrix
        for p in range(orth_basis_mat.shape[1]):
            for q in range(p, orth_basis_mat.shape[1]):
                left_op  = orth_basis_mat[:,p].transpose().conj()
                right_op = orth_basis_mat[:,q]
                H[p,q] = left_op.dot(  hamiltonian.dot(right_op)  )
                S[p,q] = left_op.dot(right_op)
                if p != q:
                    H[q,p] = H[p,q].conj()
                    S[q,p] = S[p,q].conj()
        print("S is an identity matrix:", np.allclose(S, np.identity(S.shape[0]))) # for verfication

    if np.linalg.norm(H.imag) < 1e-14:
        H = H.real
    if np.linalg.norm(S.imag) < 1e-14:
        S = S.real

    if not make_orth:
        return H, S, prev_basis
    else:
        return H, S, orth_basis_mat
    


    
## ------------------- Classical computation for GCM matrices Ends -----------------------







## ------------------- Eigenvalue computation while solving the numerical issues Starts -----------------------

def gcm_eig_helper(Hmat, Smat):
    try:
        evals_def = sl.eigh(Hmat, Smat)[0]
        return evals_def, True
    except:
        return None, False

def gcm_eig(H_indef, S_indef, ev_thresh=1e-14, printout=True):
    evals, evecs = sl.eig(S_indef)
    # sort from largest to smallest
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]

    solvable_flag = False
    curr_thresh = ev_thresh
    while not solvable_flag:
        num_pos_eigs = np.sum(np.abs(evals) > curr_thresh)
        evecs_trunc = evecs[:,:num_pos_eigs]
        evecs_trunc = np.matrix(evecs_trunc)
        S_def = evecs_trunc.H.dot(S_indef).dot(evecs_trunc)
        H_def = evecs_trunc.H.dot(H_indef).dot(evecs_trunc)
        #
        evals_def, solvable_flag = gcm_eig_helper(H_def,S_def)
        if not solvable_flag:
            curr_thresh += ev_thresh
    if printout:
        print("   Eigensolver: dimension reduced from {:d} to {:d}, with eigval > {:.3e}".format(len(evals), num_pos_eigs, curr_thresh))
    return evals_def

## ------------------- Eigenvalue computation while solving the numerical issues Ends -----------------------







## ---------------------------------------------------------

def gcm_grad(op_expmat,bas_old,hf,Hmat):
    #
    # _,op = ans_pool(indx_tuple,rot)
    bas = bas_old.copy()
    bas.insert(0, sps.csc_matrix( np.eye(op_expmat.shape[0]) ) )
    bas.append(op_expmat)
    #
    L = len(bas)
    Heff = np.zeros((L,L),dtype=complex)
    ovlp = np.zeros((L,L),dtype=complex)
    d_Heff = np.zeros((L,L),dtype=complex)
    d_ovlp = np.zeros((L,L),dtype=complex)
    for i in range(L):
        for j in range(L):
            _ = sps.csc_matrix.conj(bas[i]@hf)
            _ = sps.csc_matrix.transpose(_)
            Heff[i,j] = _.dot(Hmat.dot(bas[j]@hf))[0,0].real
            ovlp[i,j] = _.dot(bas[j]@hf)[0,0].real
            if i == L-1 and j != L-1:
                d_Heff[i,j] = -1*_.dot((bas[i]@Hmat).dot(bas[j]@hf))[0,0].real
                d_ovlp[i,j] = _.dot(bas[i].dot(bas[j]@hf))[0,0].real
            elif i != L-1 and j == L-1:
                d_Heff[i,j] = _.dot((Hmat@bas[i]).dot(bas[j]@hf))[0,0].real
                d_ovlp[i,j] = _.dot(bas[i].dot(bas[j]@hf))[0,0].real
            elif i != L-1 and j != L-1:
                d_Heff[i,j] = _.dot((Hmat@bas[i]-bas[i]@Hmat).dot(bas[j]@hf))[0,0].real
    e,f = sl.eig(Heff,ovlp)
    idx = np.argsort(e)
    f = f[:,idx]
    H_k = np.conj(f[0]).T@Heff@f[0]
    S_k = np.conj(f[0]).T@ovlp@f[0]
    d_H_k = np.conj(f[0]).T@d_Heff@f[0]
    d_S_k = np.conj(f[0]).T@d_ovlp@f[0]
    g = (d_H_k*S_k - H_k*d_S_k)/(S_k*S_k)
    return g

# No +- for basis
def gcm_mats_forag(hamiltonian, GCM_SINGLE_MATs, reference_ket, iter_coeffs, 
                   constant_t = np.pi/4,
                    prev_basis = None,
                    make_orth = False):
    """Create H and S matrices from GCM methods
        The difference between gcm_mats() is the GCM_SINGLE_MATs in this function has the lastest 
        basis matrix at the END of the list

    Args:
        hamiltonian (_type_): _description_
        GCM_SINGLE_MATs (_type_): _description_
        reference_ket (_type_): _description_
        iter_coeffs (_type_): _description_
        make_orth (boolean): If orthogonalize the GCM basis so overlap matrix S is just an identity matrix. Default is False
        prev_basis : to pass necessary basis vectors from last iteration
    """
    # iter_coeffs is from previous VQE iteration, note that the 1st elemenet is the newly inserted 0 from
    # For ADAPT-GCM-OPT, iter_coeffs is optimized right after basis selection
    # a few lines above this function called in adapt_vqe_gcm()
    ### Construct Basis
    num_basis = len(GCM_SINGLE_MATs)
    latest_basis_index = len(GCM_SINGLE_MATs) - 1
    GCM_BASIS = {'R0': reference_ket.copy()}
    # constant_t = np.pi/4
    print("Orthogonalization: ", make_orth)

    for k in range(len(GCM_SINGLE_MATs)):
        t = iter_coeffs[k] if iter_coeffs[k] != 0 else constant_t
        # print("DEBUG: used t is", t)
        opmat = GCM_SINGLE_MATs[k]
        GCM_BASIS['+R'+str(k)] = ssl.expm_multiply(   (t*opmat), reference_ket  )
        # GCM_BASIS['-R'+str(k)] = ssl.expm_multiply(  (-t*opmat), reference_ket  )

    # Final Product basis, only + in the newly-added element
    if len(GCM_SINGLE_MATs) > 1:
        prod_basis = reference_ket.copy()
        # product basis from all previous operators and the newly-added one
        for i in range( len(GCM_SINGLE_MATs) ):
            op = GCM_SINGLE_MATs[i]
            op_coef = iter_coeffs[i]
            prod_basis = ssl.expm_multiply(  (op_coef*op), prod_basis  )
        # ## product basis from the most recent two bases WARNING: new product basis computatin
        # if len(GCM_SINGLE_MATs) > 2:
        #     new_list = [-3, -2, -1]
        # else:
        #     new_list = [-2, -1]
        # for i in new_list:
        #     op = GCM_SINGLE_MATs[i]
        #     op_coef = iter_coeffs[i]
        #     prod_basis = ssl.expm_multiply(  (op_coef*op), prod_basis  )
        GCM_BASIS['+R'+str( len(GCM_SINGLE_MATs) )] = prod_basis
        ## newly-added element
        # last_op =  GCM_SINGLE_MATs[-1]
        # last_op_coef = constant_t
        # GCM_BASIS['+R'+str(num_basis)] = ssl.expm_multiply(  (last_op_coef*last_op), prod_basis  )
        # GCM_BASIS['-R'+str(num_basis)] = ssl.expm_multiply( (-last_op_coef*last_op), prod_basis  )
        
        if not make_orth:
            print("Passing out extra product bases and no orthgonalization")
            # Add product bases from previous iterations
            ri = 1
            bi = 0
            while bi < len(prev_basis):
                GCM_BASIS['+R'+str(num_basis+ri)] = prev_basis[bi]
                # GCM_BASIS['-R'+str(num_basis+ri)] = prev_basis[bi+1]
                ri += 1
                # bi += 2
                bi += 1
            # Only the new product bases are carrid out to the next iteration
            prev_basis.append(GCM_BASIS['+R'+str(num_basis)])
            # prev_basis.append(GCM_BASIS['-R'+str(num_basis)])
    
    ### Compute Matrices
    names = list(GCM_BASIS.keys())
    if not make_orth: # Usual GCM method
        H = np.zeros((len(names),len(names)), dtype=complex)
        S = np.zeros((len(names),len(names)), dtype=complex)
        basis_mat = np.zeros(   (GCM_BASIS[names[0]].shape[0], len(names))   , dtype=complex)
        for i in range(len(names)):
            basis_mat[:,i] = GCM_BASIS[ names[i] ].toarray().flatten()
            for j in range(i,len(names)):
                left_op  = GCM_BASIS[ names[i] ].transpose().conj()
                right_op = GCM_BASIS[ names[j] ]
                H[i,j] = left_op.dot(  hamiltonian.dot(right_op)  )[0,0]
                S[i,j] = left_op.dot(right_op)[0,0]
                if i != j:
                    H[j,i] = H[i,j].conj()
                    S[j,i] = S[i,j].conj()
    else: # Use orthogonalization
        # Create a matrix whose column is each GCM basis
        if len(GCM_SINGLE_MATs) > 1:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+2
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-2] = prev_basis
            basis_mat[:,num_orthbasis-2] = GCM_BASIS['+R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            # basis_mat[:,num_orthbasis-3] = GCM_BASIS['-R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['+R'+str(num_basis)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            # basis_mat[:,num_orthbasis-1] = GCM_BASIS['-R'+str(num_basis)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
        else:
            mat_height = prev_basis.shape[0]
            num_orthbasis = prev_basis.shape[1]+1
            basis_mat = np.zeros((mat_height, num_orthbasis), dtype=complex)
            basis_mat[:,:num_orthbasis-1] = prev_basis
            basis_mat[:,num_orthbasis-1] = GCM_BASIS['+R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix
            # basis_mat[:,num_orthbasis-1] = GCM_BASIS['-R'+str(latest_basis_index)].toarray().flatten() # NOTE: GCM_BASIS[ names[i] ] is a CSR matrix

        orth_basis_mat = sl.orth(basis_mat)
        # S = np.identity(orth_basis_mat.shape[1]) # trivial
        H = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex)
        S = np.zeros((orth_basis_mat.shape[1],orth_basis_mat.shape[1]), dtype=complex) # We can check if S is identity
        # Create H from orthogonalized matrix
        for p in range(orth_basis_mat.shape[1]):
            for q in range(p, orth_basis_mat.shape[1]):
                left_op  = orth_basis_mat[:,p].transpose().conj()
                right_op = orth_basis_mat[:,q]
                H[p,q] = left_op.dot(  hamiltonian.dot(right_op)  )
                S[p,q] = left_op.dot(right_op)
                if p != q:
                    H[q,p] = H[p,q].conj()
                    S[q,p] = S[p,q].conj()
        print("S is an identity matrix:", np.allclose(S, np.identity(S.shape[0]))) # for verfication

    if np.linalg.norm(H.imag) < 1e-14:
        H = H.real
    if np.linalg.norm(S.imag) < 1e-14:
        S = S.real

    if not make_orth:
        return H, S, prev_basis, basis_mat
    else:
        return H, S, orth_basis_mat


## ADAPT-GCM major function
def adapt_gcm(hamiltonian_op, pool, reference_ket, theta, 
                use_gcmgrad  = False,
                use_qugcm       = False,
                file_prefix = None,
                stop_type      = 'long_flat',
                stop_thresh    = None,
                adapt_maxiter   = 200,
                no_printout_level = 0,
                make_orth = False,
                ev_thresh = 1e-14,
                ):
    """Running ADAPT VQE and GCM at the same time

    long_flat_jump has stop_thresh (flat_iter, change_tol, relative_jump): if the lowest eigenvalue has not changed greater than change_tol in
    flat_iter number of iterations, or there is a sudden jump that greater than relative_jump, then set to converge

    Args:
        file_prefix (str): file path for saving GCM matrices in each iteration END with '/'. None for not saving matrices. If path does not exists, it does NOT create the path. Defaults to None. 
        no_printout_level (int, optional): 0 means printout everything, 1 means no VQE parameter optimization info, 2 means ALSO no ansatz info in each iteration. Defaults to 0.
        show_callback (bool, optional): If print information for iterations of parameter optimization. Defaults to True. 
        make_orth (bool, optional): If make basis orthgonal when construct matrices for GCM. Defaults to False. 
    """
    long_flat_counter = 0
    absolute_counter = 0
    times_grad = [] # Muqing: Time for gradient calculation
    times_gcm = [] # Muqing: Time for GCM algorithm
    times_gcmeig = [] # Muqing: Time for GCM eigenvalue solving

    jumps = []
    GCM_DIFFS = [] # Muqing: Error from FCI energy for GCM algorihtm
    adapt_maxiter = min(adapt_maxiter, len(pool.fermi_ops))
    hamiltonian = of.linalg.get_sparse_operator(hamiltonian_op)
    scipy.sparse.save_npz(file_prefix+'MoleHam.npz', hamiltonian, compressed=True)
    # raise Exception("Temp Stop")
    ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print(" Reference Energy: %12.8f" %ref_energy)

    #Thetas
    parameters = []
    curr_state = 1.0*reference_ket

    pool.generate_SparseMatrix()
    if no_printout_level < 1:
        pool.gradient_print_thresh = 1e-3
    elif no_printout_level == 1:
        pool.gradient_print_thresh = 1e-1
    else:
        pool.gradient_print_thresh = 1
    # pool.gradient_print_thresh = 0

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_expmat = [] 
    ansatz_indices = []

    sys.stdout.flush() 

    print(" Start ADAPT-GCM algorithm")
    
    GCM_RESULT = [ref_energy]
    ev_records = [ref_energy]
    GCM_BASIS_SIZE = []
    
    prodBasis_set = []
    flat_ref_ket = reference_ket.toarray().flatten()
    last_basis_mat = flat_ref_ket.reshape(  (len(flat_ref_ket), 1)  )
    ####
    true_lowest_ev = scipy.sparse.linalg.eigsh(hamiltonian,1,which='SA',v0=reference_ket.todense())[0][0].real # Muqing: Find the actual FCI energy   ## Cannot converge for H65A
    ####
    sig = hamiltonian.dot(reference_ket)

    print(" Now start to grow the ansatz")
    sys.stdout.flush() 
    for n_iter in range(0,adapt_maxiter+1):
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-GCM iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        next_expop = None
        curr_norm = 0

        # print(" Check each new operator for coupling")
        sig = hamiltonian.dot(curr_state)
        tstart_grad = timer()
        for oi in range(pool.n_ops):
            if oi not in ansatz_indices:
                if use_gcmgrad:
                    op_expmat = ssl.expm( theta*pool.spmat_ops[oi] )
                    gi = gcm_grad(op_expmat,ansatz_expmat,curr_state,hamiltonian)
                    curr_norm += gi*gi
                else:
                    # op_expmat = ssl.expm( theta*pool.spmat_ops[oi] )
                    gi = pool.compute_gradient_i(oi, curr_state, sig)
                    curr_norm += gi*gi
                if abs(gi) > abs(next_deriv):
                    next_deriv = gi
                    next_index = oi
                    # next_expop = op_expmat
        tend_grad = timer()
        times_grad.append(tend_grad-tstart_grad) # Muqing: Time for gradient calculation

        curr_norm = np.sqrt(curr_norm)

        # min_options = {'gtol': 1e-3, 'disp':False}

        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)
        print(" Max  of gradient = %12.8f" %next_deriv)

        converged = False
        if n_iter >= (adapt_maxiter):
            print(" Converged due to maximum iterations")
            converged = True
        elif pool.n_ops <= len(ansatz_indices):
            print(" Converged due to running out the operators")
            converged = True
        elif curr_norm <= 1e-8 or next_index is None:
            print(" Converged due to all gradients are 0")
            converged = True
        elif stop_type == 'long_flat':
            try:
                flat_iter, change_tol = stop_thresh
                if flat_iter is None:
                    flat_iter = min(stop_thresh[0], int(0.2*np.abs(pool.n_ops - len(ansatz_indices))) )
            except:
                change_tol= 1e-6
                flat_iter = min(stop_thresh[0], int(0.2*np.abs(pool.n_ops - len(ansatz_indices))) )
            if len(ev_records) > min(4, len(pool.spmat_ops)): # no check in first 3 iterations to avoid "quick" converges
                if (np.abs(ev_records[-1] - ev_records[-2])) < change_tol: ## if absolute change is too smal
                    print(" Small change in {:d}/{:d} iterations".format(long_flat_counter, flat_iter))
                    long_flat_counter += 1
                    if long_flat_counter >= flat_iter:
                        print(" Converged due to tiny changes {:.3e} in eigenvalues in {:d} iterations".format( np.abs(ev_records[-1] - ev_records[-2]), flat_iter))
                        converged = True
                else:
                    long_flat_counter = 0 # reset the counter
        elif stop_type == 'relative':
            if len(ev_records) > 1:
                iter_diffs = np.abs(ev_records[-1] - ev_records[-2])
                if iter_diffs < stop_thresh:
                    print(" Ready to converge due to small changes in eigenvalues: |{:.3e} - {:.3e}| = {:.3e}".format(ev_records[-1], ev_records[-2], iter_diffs))
                    converged = True
        elif stop_type == 'absolute':
            if len(GCM_DIFFS) >= 2 and np.abs(GCM_DIFFS[-1]) < (stop_thresh[1]*100):
                if np.abs(GCM_DIFFS[-1] - GCM_DIFFS[-2]) < stop_thresh[1]:
                    print("Convergence check: absolute counter = {:d}".format(absolute_counter))
                    absolute_counter += 1
                if absolute_counter > stop_thresh[0]:
                    print(f"Converge due to small changes ({stop_thresh}) in GCM errors and errors < {stop_thresh[1]*100}")
                    converged = True
            else:
                pass
        else:
            raise Exception("Not Implement.")

        if converged:
            # printout
            sys.stdout.flush() 
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" Number of basis used in final iteration of GCM:", GCM_BASIS_SIZE[-1])
            print(" *True Energy : %20.12f" % true_lowest_ev)
            print(" *GCM Finished: %20.12f" % GCM_RESULT[0])

            gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
            print(" *GCM error {:.6e}".format( gcm_diff ))
                  
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = pool.get_string_for_term(ansatz_ops[si], final=True) #Muqing, add final=True
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            print(" ---------------------- ")
            break

        print(" Add operator %4i" %next_index)
        parameters.append(theta)
        ansatz_ops.append(pool.fermi_ops[next_index])
        ansatz_mat.append(pool.spmat_ops[next_index])
        ansatz_indices.append(next_index)
        if use_gcmgrad:
            ansatz_expmat.append(next_expop)
        
        sys.stdout.flush() 
        ################ GCM Computation #######################
        print("\n--------- GCM Computation Start ---------")
        if not make_orth:
            if not use_qugcm:
                tstart_gcm = timer()
                gcmH, gcmS, prodBasis_set, iter_basis_mat = gcm_mats_forag(hamiltonian, ansatz_mat, reference_ket, parameters,
                                                           constant_t=theta,
                                                    prev_basis = prodBasis_set, make_orth = make_orth)
                tend_gcm = timer()
                times_gcm.append(tend_gcm-tstart_gcm) # Muqing: Time for GCM algorithm
            else:
                raise Exception("Not implement.")
            print("Number of saved prod bases", len(prodBasis_set), "H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        else:
            if not use_qugcm:
                tstart_gcm = timer()
                gcmH, gcmS, last_basis_mat = gcm_mats_forag(hamiltonian, ansatz_mat, reference_ket, parameters, 
                                                            constant_t=theta,
                                                        prev_basis=last_basis_mat, make_orth = make_orth)
                tend_gcm = timer()
                times_gcm.append(tend_gcm-tstart_gcm) # Muqing: Time for GCM algorithm
            else:
            #     print("NOT IMPLEMENT. Quantum orthgonalization is not implemented yet")
            #     gcmH, gcmS, last_basis_mat = gcm_mats(hamiltonian, ansatz_mat, reference_ket, parameters, 
            #                                             prev_basis=last_basis_mat, make_orth = make_orth)
                raise Exception("Not implement.")
            print("basis matrix size", last_basis_mat.shape, "H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        if file_prefix is not None:
            import os
            if not os.path.exists(file_prefix+'IterMats'):
                os.makedirs(file_prefix+'IterMats')
            # np.save(file_prefix+'IterMats/IterAG{:d}_S.npy'.format(n_iter), gcmS)
            # np.save(file_prefix+'IterMats/IterAG{:d}_H.npy'.format(n_iter), gcmH)
            # np.save(file_prefix+'IterMats/IterAG{:d}_basis_mat.npy'.format(n_iter), iter_basis_mat)
            # # np.save(file_prefix+'IterMats/IterAG{:d}_bases.npy'.format(n_iter), np.array(ansatz_mat))
            # # np.save(file_prefix+'IterMats/IterAG{:d}_preop_coeffs.npy'.format(n_iter), np.array(parameters))
            np.savez_compressed(file_prefix+'IterMats/IterAG{:d}.npz'.format(n_iter),
                                gcmH=gcmH, gcmS=gcmS, iter_basis_mat=iter_basis_mat)
        tstart_gcmeig = timer()
        if not make_orth:
            GCM_RESULT = gcm_eig(gcmH, gcmS, ev_thresh=ev_thresh, 
                                printout=True)
        else:
            GCM_RESULT = sl.eigh(gcmH)[0]
        tend_gcmeig = timer()
        times_gcmeig.append(tend_gcmeig-tstart_gcmeig) # Muqing: Time for GCM eigenvalue solving

        ev_records.append(GCM_RESULT[0])
        print("--------- GCM Computation Over  ---------\n")
        ###################################################
        sys.stdout.flush()

        
        curr_state = ssl.expm_multiply(theta*pool.spmat_ops[next_index], curr_state)
        print(" GCM Finished: %20.12f" % GCM_RESULT[0])
        gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
        GCM_DIFFS.append(gcm_diff)
        print(" GCM error {:.6e}".format( gcm_diff ))
        
        if no_printout_level < 2:
            print(" -----------New ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                sys.stdout.flush() 
                opstring = pool.get_string_for_term(ansatz_ops[si])
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
        sys.stdout.flush() 

        ## iteration time
        iter_time_arr = np.array([tend_grad-tstart_grad, 
                                  tend_gcm-tstart_gcm,
                                  tend_gcmeig-tstart_gcmeig])
        if file_prefix is not None:
            try:
                np.save(file_prefix+'IterMats/Iter{:d}_time.npy'.format(n_iter), iter_time_arr)
            except:
                np.save(file_prefix+'Iter{:d}_time.npy'.format(n_iter), iter_time_arr)

    ## save time
    if file_prefix is not None:
        np.save(file_prefix+'times_grad.npy', np.array(times_grad))
        np.save(file_prefix+'times_gcm.npy', np.array(times_gcm))
        np.save(file_prefix+'times_gcmeig.npy', np.array(times_gcmeig))
    return ev_records[1:],  ansatz_indices, GCM_DIFFS,GCM_BASIS_SIZE


###-------------------------------------------------------------------------------------------------------












###-------------------------------------------------------------------------------------------------------





## ADAPT-GCM major function
def adapt_gcm_opt(hamiltonian_op, pool, reference_ket, theta, 
                adapt_opt_intvl = 5,
                opt_depth = 3,
                use_gcmgrad  = False,
                use_qugcm       = False,
                file_prefix = None,
                stop_type      = 'long_flat',
                stop_thresh    = None,
                adapt_maxiter   = 200,
                no_printout_level = 0,
                make_orth = False,
                ev_thresh = 1e-14,
                ):
    """Running ADAPT VQE and GCM at the same time

    long_flat_jump has stop_thresh (flat_iter, change_tol, relative_jump): if the lowest eigenvalue has not changed greater than change_tol in
    flat_iter number of iterations, or there is a sudden jump that greater than relative_jump, then set to converge

    Args:
        adapt_opt_intvl: ADAPT iteration interval for optimization
        opt_depth: VQE optimization iterations when needed
        file_prefix (str): file path for saving GCM matrices in each iteration END with '/'. None for not saving matrices. If path does not exists, it does NOT create the path. Defaults to None. 
        no_printout_level (int, optional): 0 means printout everything, 1 means no VQE parameter optimization info, 2 means ALSO no ansatz info in each iteration. Defaults to 0.
        show_callback (bool, optional): If print information for iterations of parameter optimization. Defaults to True. 
        make_orth (bool, optional): If make basis orthgonal when construct matrices for GCM. Defaults to False. 
    """
    print(f" OPTIMIZE every {adapt_opt_intvl} iterations for max {opt_depth} times")
    long_flat_counter = 0
    absolute_counter = 0
    VQE_ITERS = [] # number of optimization iterations
    times_vqe = [] # Muqing: Time for VQE optimization
    times_grad = [] # Muqing: Time for gradient calculation
    times_gcm = [] # Muqing: Time for GCM algorithm
    times_gcmeig = [] # Muqing: Time for GCM eigenvalue solving

    GCM_DIFFS = [] # Muqing: Error from FCI energy for GCM algorihtm
    adapt_maxiter = min(adapt_maxiter, len(pool.fermi_ops))
    hamiltonian = of.linalg.get_sparse_operator(hamiltonian_op)
    scipy.sparse.save_npz(file_prefix+'MoleHam.npz', hamiltonian, compressed=True)
    ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print(" Reference Energy: %12.8f" %ref_energy)

    #Thetas
    parameters = []
    curr_state = 1.0*reference_ket

    pool.generate_SparseMatrix()
    if no_printout_level < 1:
        pool.gradient_print_thresh = 1e-3
    elif no_printout_level == 1:
        pool.gradient_print_thresh = 1e-1
    else:
        pool.gradient_print_thresh = 1

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_expmat = [] 
    ansatz_indices = []

    sys.stdout.flush() 

    print(" Start ADAPT-GCM algorithm")
    
    GCM_RESULT = [ref_energy]
    ev_records = [ref_energy]
    GCM_BASIS_SIZE = []
    
    prodBasis_set = []
    flat_ref_ket = reference_ket.toarray().flatten()
    last_basis_mat = flat_ref_ket.reshape(  (len(flat_ref_ket), 1)  )
    ####
    true_lowest_ev = scipy.sparse.linalg.eigsh(hamiltonian,1,which='SA',v0=reference_ket.todense())[0][0].real # Muqing: Find the actual FCI energy   ## Cannot converge for H65A
    ####
    sig = hamiltonian.dot(reference_ket)

    print(" Now start to grow the ansatz")
    sys.stdout.flush() 
    for n_iter in range(0,adapt_maxiter+1):
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-GCM iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        next_expop = None
        curr_norm = 0

        # print(" Check each new operator for coupling")
        sig = hamiltonian.dot(curr_state)
        tstart_grad = timer()
        for oi in range(pool.n_ops):
            if oi not in ansatz_indices:
                if use_gcmgrad:
                    op_expmat = ssl.expm( theta*pool.spmat_ops[oi] )
                    gi = gcm_grad(op_expmat,ansatz_expmat,curr_state,hamiltonian)
                    curr_norm += gi*gi
                else:
                    # op_expmat = ssl.expm( theta*pool.spmat_ops[oi] )
                    gi = pool.compute_gradient_i(oi, curr_state, sig)
                    curr_norm += gi*gi
                if abs(gi) > abs(next_deriv):
                    next_deriv = gi
                    next_index = oi
                    # next_expop = op_expmat
        tend_grad = timer()
        times_grad.append(tend_grad-tstart_grad) # Muqing: Time for gradient calculation

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': 1e-3, 'disp':False, 'maxiter': opt_depth}

        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)
        print(" Max  of gradient = %12.8f" %next_deriv)

        converged = False
        if n_iter >= (adapt_maxiter):
            print(" Converged due to maximum iterations")
            converged = True
        if pool.n_ops <= len(ansatz_indices):
            print(" Converged due to running out the operators")
            converged = True
        if curr_norm <= 1e-8 or next_index is None:
            print(" Converged due to all gradients are 0")
            converged = True
        if stop_type == 'long_flat':
            try:
                flat_iter, change_tol = stop_thresh
                if flat_iter is None:
                    flat_iter = min(stop_thresh[0], int(0.2*np.abs(pool.n_ops - len(ansatz_indices))) )
            except:
                change_tol= 1e-6
                flat_iter = min(stop_thresh[0], int(0.2*np.abs(pool.n_ops - len(ansatz_indices))) )
            if len(ev_records) > min(4, len(pool.spmat_ops)): # no check in first 3 iterations to avoid "quick" converges
                if (np.abs(ev_records[-1] - ev_records[-2])) < change_tol: ## if absolute change is too smal
                    print(" Small change in {:d}/{:d} iterations".format(long_flat_counter, flat_iter))
                    long_flat_counter += 1
                    if long_flat_counter >= flat_iter:
                        print(" Converged due to tiny changes {:.3e} in eigenvalues in {:d} iterations".format( np.abs(ev_records[-1] - ev_records[-2]), flat_iter))
                        converged = True
                else:
                    long_flat_counter = 0 # reset the counter
        elif stop_type == 'relative':
            if len(ev_records) > 1:
                iter_diffs = np.abs(ev_records[-1] - ev_records[-2])
                if iter_diffs < stop_thresh:
                    print(" Ready to converge due to small changes in eigenvalues: |{:.3e} - {:.3e}| = {:.3e}".format(ev_records[-1], ev_records[-2], iter_diffs))
                    converged = True
        elif stop_type == 'absolute':
            if len(GCM_DIFFS) >= 2 and np.abs(GCM_DIFFS[-1]) < (stop_thresh[1]*100):
                if np.abs(GCM_DIFFS[-1] - GCM_DIFFS[-2]) < stop_thresh[1]:
                    print("Convergence check: absolute counter = {:d}".format(absolute_counter))
                    absolute_counter += 1
                if absolute_counter > stop_thresh[0]:
                    print(f"Converge due to small changes ({stop_thresh}) in GCM errors and errors < {stop_thresh[1]*100}")
                    converged = True
            else:
                pass
        else:
            raise Exception("Not Implement.")

        if converged:
            # printout
            sys.stdout.flush() 
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" Number of basis used in final iteration of GCM:", GCM_BASIS_SIZE[-1])
            print(" *True Energy : %20.12f" % true_lowest_ev)
            print(" *GCM Finished: %20.12f" % GCM_RESULT[0])
            gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
            print(" *GCM error {:.6e}".format( gcm_diff ))
                  
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = pool.get_string_for_term(ansatz_ops[si], final=True) #Muqing, add final=True
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            print(" ---------------------- ")
            break

        print(" Add operator %4i" %next_index)
        parameters.append(theta)
        ansatz_ops.append(pool.fermi_ops[next_index])
        ansatz_mat.append(pool.spmat_ops[next_index])
        ansatz_indices.append(next_index)
        if use_gcmgrad:
            ansatz_expmat.append(next_expop)
        
        sys.stdout.flush() 

        # ###### Selective VQE optimization #######
        if n_iter % adapt_opt_intvl == 0:
            tstart_vqe = timer() # Muqing: timer
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
            if no_printout_level == 0:
                opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                        options = min_options, method = 'BFGS', callback=trial_model.callback)
            else:
                opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                        options = min_options, method = 'BFGS')
            tend_vqe = timer() # Muqing: timer
            times_vqe.append(tend_vqe-tstart_vqe) # Muqing: time in seconds

            sys.stdout.flush() 
            print(" Number of optimization iteration: ", opt_result.nit)
            VQE_ITERS.append(opt_result.nit)
            parameters = list(opt_result['x'])
            curr_state = trial_model.prepare_state(parameters)
        else:
            curr_state = ssl.expm_multiply(theta*pool.spmat_ops[next_index], curr_state)

        # ## WARNING: new curr_state computation corresponds to the new product basis computation
        # ###### Selective VQE optimization #######
        # if n_iter % adapt_opt_intvl == 0:
        #     tstart_vqe = timer() # Muqing: timer
        #     if len(ansatz_mat) > 2:
        #         new_ansatz_mat = ansatz_mat[-3:]
        #         new_parameters = parameters[-3:]
        #     else:
        #         new_ansatz_mat = ansatz_mat
        #         new_parameters = parameters
        #     trial_model = tUCCSD(hamiltonian, new_ansatz_mat, reference_ket, new_parameters)
        #     if no_printout_level == 0:
        #         opt_result = scipy.optimize.minimize(trial_model.energy, new_parameters, jac=trial_model.gradient,
        #                 options = min_options, method = 'BFGS', callback=trial_model.callback)
        #     else:
        #         opt_result = scipy.optimize.minimize(trial_model.energy, new_parameters, jac=trial_model.gradient,
        #                 options = min_options, method = 'BFGS')
        #     tend_vqe = timer() # Muqing: timer
        #     times_vqe.append(tend_vqe-tstart_vqe) # Muqing: time in seconds

        #     sys.stdout.flush() 
        #     print(" Number of optimization iteration: ", opt_result.nit)
        #     VQE_ITERS.append(opt_result.nit)
        #     if len(parameters) > 2:
        #         parameters[-3:] = list(opt_result['x'])
        #     else:
        #         parameters = list(opt_result['x'])
        #     curr_state = trial_model.prepare_state(list(opt_result['x']))
        # else:
        #     curr_state = reference_ket.copy()
        #     if len(ansatz_mat) > 2:
        #         new_list = [-3,-2,-1]
        #     else:
        #         new_list = [-2,-1]
        #     for i in new_list:
        #         op = ansatz_mat[i]
        #         op_coef = theta
        #         curr_state = ssl.expm_multiply(  (op_coef*op), curr_state  )
        # ##


        ################ GCM Computation #######################
        print("\n--------- GCM Computation Start ---------")
        if not make_orth:
            if not use_qugcm:
                tstart_gcm = timer()
                gcmH, gcmS, prodBasis_set, iter_basis_mat = gcm_mats_forag(hamiltonian, ansatz_mat, reference_ket, parameters,
                                                           constant_t=theta,
                                                    prev_basis = prodBasis_set, make_orth = make_orth)
                tend_gcm = timer()
                times_gcm.append(tend_gcm-tstart_gcm) # Muqing: Time for GCM algorithm
            else:
                raise Exception("Not implement.")
            print("Number of saved prod bases", len(prodBasis_set), "H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        else:
            if not use_qugcm:
                tstart_gcm = timer()
                gcmH, gcmS, last_basis_mat = gcm_mats_forag(hamiltonian, ansatz_mat, reference_ket, parameters, 
                                                            constant_t=theta,
                                                        prev_basis=last_basis_mat, make_orth = make_orth)
                tend_gcm = timer()
                times_gcm.append(tend_gcm-tstart_gcm) # Muqing: Time for GCM algorithm
            else:
                raise Exception("Not implement.")
            print("basis matrix size", last_basis_mat.shape, "H matrix size", gcmH.shape)
            GCM_BASIS_SIZE.append(gcmH.shape[0])
        if file_prefix is not None:
            import os
            if not os.path.exists(file_prefix+'IterMats'):
                os.makedirs(file_prefix+'IterMats')
            # np.save(file_prefix+'IterMats/IterAG{:d}_S.npy'.format(n_iter), gcmS)
            # np.save(file_prefix+'IterMats/IterAG{:d}_H.npy'.format(n_iter), gcmH)
            # np.save(file_prefix+'IterMats/IterAG{:d}_basis_mat.npz'.format(n_iter), iter_basis_mat)
            # # np.save(file_prefix+'IterMats/IterAG{:d}_bases.npy'.format(n_iter), np.array(ansatz_mat))
            # np.save(file_prefix+'IterMats/IterAG{:d}_preop_coeffs.npy'.format(n_iter), np.array(parameters))
            np.savez_compressed(file_prefix+'IterMats/IterAG{:d}.npz'.format(n_iter),
                                gcmH=gcmH, gcmS=gcmS, iter_basis_mat=iter_basis_mat, coeffs=np.array(parameters))
        tstart_gcmeig = timer()
        if not make_orth:
            GCM_RESULT = gcm_eig(gcmH, gcmS, ev_thresh=ev_thresh, 
                                printout=True)
        else:
            GCM_RESULT = sl.eigh(gcmH)[0]
        tend_gcmeig = timer()
        times_gcmeig.append(tend_gcmeig-tstart_gcmeig) # Muqing: Time for GCM eigenvalue solving

        ev_records.append(GCM_RESULT[0])
        print("--------- GCM Computation Over  ---------\n")
        ###################################################
        sys.stdout.flush()

        
        # curr_state = ssl.expm_multiply(theta*pool.spmat_ops[next_index], curr_state) ## move to other places
        print(" GCM Finished: %20.12f" % GCM_RESULT[0])
        gcm_diff = np.abs(true_lowest_ev-GCM_RESULT[0])
        GCM_DIFFS.append(gcm_diff)
        print(" GCM error {:.6e}".format( gcm_diff ))
        
        if no_printout_level < 2:
            print(" -----------New ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                sys.stdout.flush() 
                opstring = pool.get_string_for_term(ansatz_ops[si])
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
        sys.stdout.flush() 

        ## iteration time
        iter_time_arr = np.array([tend_grad-tstart_grad, 
                                  tend_gcm-tstart_gcm,
                                  tend_gcmeig-tstart_gcmeig])
        if file_prefix is not None:
            try:
                np.save(file_prefix+'IterMats/Iter{:d}_time.npy'.format(n_iter), iter_time_arr)
            except:
                np.save(file_prefix+'Iter{:d}_time.npy'.format(n_iter), iter_time_arr)

    ## save time
    if file_prefix is not None:
        np.save(file_prefix+'times_grad.npy', np.array(times_grad))
        np.save(file_prefix+'times_gcm.npy', np.array(times_gcm))
        np.save(file_prefix+'times_gcmeig.npy', np.array(times_gcmeig))
        np.save(file_prefix+'times_vqe.npy', np.array(times_vqe))
    return ev_records[1:], ansatz_indices, GCM_DIFFS, GCM_BASIS_SIZE, VQE_ITERS