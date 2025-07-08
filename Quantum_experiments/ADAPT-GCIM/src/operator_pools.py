import openfermion
import numpy as np
import copy as cp
import scipy.sparse.linalg as ssl # Muqing: for Pauli Excitation

from openfermion import *

print("This is a modified operator_pools.py by Muqing Zheng")

class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0
        self.gradient_print_thresh = 0

    def init(self,n_orb,
            n_occ_a=None,
            n_occ_b=None,
            n_vir_a=None,
            n_vir_b=None):
        self.n_orb = n_orb
        self.n_spin_orb = 2*self.n_orb

        if n_occ_a!=None and n_occ_b!=None:
            assert(n_occ_a == n_occ_b)
            self.n_occ = n_occ_a
            self.n_occ_a = n_occ_a
            self.n_occ_b = n_occ_b
            self.n_vir = n_vir_a
            self.n_vir_a = n_vir_a
            self.n_vir_b = n_vir_b
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            temp_sp = linalg.get_sparse_operator(op, n_qubits = self.n_spin_orb) # Muqing
            if np.allclose(ssl.norm(temp_sp.imag), 0): # Muqing
                temp_sp = temp_sp.real # Muqing
            self.spmat_ops.append(temp_sp) # Muqing
        assert(len(self.spmat_ops) == self.n_ops)
        return


    def get_string_for_term(self,op, final=False): # Muqing: add parameter to print operator as a whole
        if final:
            return str(op).replace("\n", "")
        if type(op) == QubitOperator: # Muqing
            return str(op) # Muqing
        opstring = ""
        spins = ""
        for t in op.terms:
            opstring = "("
            for ti in t:
                opstring += str(int(ti[0])) # Muqing: show the original orbital index
                # opstring += str(int(ti[0]/2))
                if ti[1] == 0:
                    opstring += "  "
                elif ti[1] == 1:
                    opstring += "' "
                else:
                    print("wrong")
                    exit()
                spins += str(ti[0]%2)
            opstring += ")"
            spins += " "
        opstring = " %18s : %s" %(opstring, spins)
        return opstring



    def compute_gradient_i(self,i,v,sig):
        """
        For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
        g(k) = 2Real<HA(k)>

        Note - this assumes A(k) is an antihermitian operator. If this is not the case, the derived class should
        reimplement this function. Of course, also assumes H is hermitian

        v   = current_state
        sig = H*v

        """
        opA = self.spmat_ops[i]
        gi = 2*(sig.transpose().conj().dot(opA.dot(v)))
        assert(gi.shape == (1,1))
        gi = gi[0,0]
        assert(np.isclose(gi.imag,0))
        gi = gi.real

        opstring = self.get_string_for_term(self.fermi_ops[i])

        if abs(gi) >= self.gradient_print_thresh:
            print(" %4i %12.8f %s" %(i, gi, opstring) )
        return gi

    
class UCCSD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        Generate UCCSD operators which include:
        1. Singles: excitations from occupied to virtual orbitals
        2. Doubles: paired excitations from occupied to virtual orbitals
        
        Assumes spin orbitals are labeled as 0a,0b,1a,1b,2a,2b,...
        where a is alpha spin and b is beta spin.
        """
        print(" Form UCCSD excitation operators")
        self.fermi_ops = []
        
        # Ensure equal number of alpha and beta occupied orbitals
        assert(self.n_occ_a == self.n_occ_b)
        n_occ = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_vir_b = self.n_vir_b
        
        # Singles (alpha)
        num_singles = 0
        for i in range(n_occ):
            ia = 2*i  # alpha spin orbital index
            for a in range(n_vir_a):
                aa = 2*n_occ + 2*a  # alpha virtual orbital index
                
                # Create excitation operator
                term = FermionOperator(((aa, 1), (ia, 0)))
                term -= hermitian_conjugated(term)
                term = normal_ordered(term)
                
                # Normalize
                coeff = 0
                for t in term.terms:
                    coeff_t = term.terms[t]
                    coeff += coeff_t * coeff_t
                
                if term.many_body_order() > 0:
                    term = term / np.sqrt(coeff)
                    self.fermi_ops.append(term)
                    num_singles += 1
        
        # Singles (beta)
        for i in range(n_occ):
            ib = 2*i + 1  # beta spin orbital index
            for a in range(n_vir_b):
                ab = 2*n_occ + 2*a + 1  # beta virtual orbital index
                
                # Create excitation operator
                term = FermionOperator(((ab, 1), (ib, 0)))
                term -= hermitian_conjugated(term)
                term = normal_ordered(term)
                
                # Normalize
                coeff = 0
                for t in term.terms:
                    coeff_t = term.terms[t]
                    coeff += coeff_t * coeff_t
                
                if term.many_body_order() > 0:
                    term = term / np.sqrt(coeff)
                    self.fermi_ops.append(term)
                    num_singles += 1
        
        # Doubles (alpha-alpha)
        num_doubles = 0
        for i in range(n_occ):
            ia = 2*i  # alpha
            for j in range(i, n_occ):
                ja = 2*j  # alpha
                for a in range(n_vir_a):
                    aa = 2*n_occ + 2*a  # alpha
                    for b in range(a, n_vir_a):
                        ba = 2*n_occ + 2*b  # alpha
                        
                        # Skip diagonal terms for same indices
                        if i == j and a == b:
                            continue
                            
                        # Create excitation operator
                        term = FermionOperator(((aa, 1), (ba, 1), (ja, 0), (ia, 0)))
                        term -= hermitian_conjugated(term)
                        term = normal_ordered(term)
                        
                        # Normalize
                        coeff = 0
                        for t in term.terms:
                            coeff_t = term.terms[t]
                            coeff += coeff_t * coeff_t
                        
                        if term.many_body_order() > 0:
                            term = term / np.sqrt(coeff)
                            self.fermi_ops.append(term)
                            num_doubles += 1
                        
        
        # Doubles (beta-beta)
        for i in range(n_occ):
            ib = 2*i + 1  # beta
            for j in range(i, n_occ):
                jb = 2*j + 1  # beta
                for a in range(n_vir_b):
                    ab = 2*n_occ + 2*a + 1  # beta
                    for b in range(a, n_vir_b):
                        bb = 2*n_occ + 2*b + 1  # beta
                        
                        # Skip diagonal terms for same indices
                        if i == j and a == b:
                            continue
                            
                        # Create excitation operator
                        term = FermionOperator(((ab, 1), (bb, 1), (jb, 0), (ib, 0)))
                        term -= hermitian_conjugated(term)
                        term = normal_ordered(term)
                        
                        # Normalize
                        coeff = 0
                        for t in term.terms:
                            coeff_t = term.terms[t]
                            coeff += coeff_t * coeff_t
                        
                        if term.many_body_order() > 0:
                            term = term / np.sqrt(coeff)
                            self.fermi_ops.append(term)
                            num_doubles += 1
        
        # Doubles (alpha-beta)
        for i in range(n_occ):
            ia = 2*i  # alpha
            for j in range(n_occ):
                jb = 2*j + 1  # beta
                for a in range(n_vir_a):
                    aa = 2*n_occ + 2*a  # alpha
                    for b in range(n_vir_b):
                        bb = 2*n_occ + 2*b + 1  # beta
                        
                        # Create excitation operator
                        term = FermionOperator(((aa, 1), (bb, 1), (jb, 0), (ia, 0)))
                        term -= hermitian_conjugated(term)
                        term = normal_ordered(term)
                        
                        # Normalize
                        coeff = 0
                        for t in term.terms:
                            coeff_t = term.terms[t]
                            coeff += coeff_t * coeff_t
                        
                        if term.many_body_order() > 0:
                            term = term / np.sqrt(coeff)
                            self.fermi_ops.append(term)
                            num_doubles += 1
        
        self.n_ops = len(self.fermi_ops)
        print(f" Number of operators: {self.n_ops}, Number of Singles: {num_singles}, Number of Doubles: {num_doubles}")
        return

    