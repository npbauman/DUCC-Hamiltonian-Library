from moles import *
## ADAPT-GCM
from adapt_vqe_gcm import *
from _gcim_utilis import *
from timeit import default_timer as timer

import argparse



#########
ham_bz66="../../Benzene/cc-pVDZ/FrozenCoreCCSD_6Elec_6Orbs/DUCC3/benzene-FrozenCoreCCSD_6Elec_6Orbs.out-xacc"
ham_bztz66="../../Benzene/cc-pVTZ/FrozenCoreCCSD_6Elec_6Orbs/DUCC3/benzene-FC-CCSD-6Electrons-6Orbitals.out-xacc"

ham_fbp="../../FBP/FreeBasedPorphyrin.out-xacc"

ham_n210d3="../../N2/cc-pVTZ/1.0_Eq-2.0680au/DUCC3/1.0_Eq-2.0680au-DUCC3.out-xacc"
ham_n215d3="../../N2/cc-pVTZ/1.5_Eq-3.1020au/DUCC3/1.5_Eq-3.1020au-DUCC3.out-xacc"
ham_n220d3="../../N2/cc-pVTZ/2.0_Eq-4.1360au/DUCC3/2.0_Eq-4.1360au-DUCC3.out-xacc"
ham_n225d3="../../N2/cc-pVTZ/2.5_Eq-5.1700au/DUCC3/2.5_Eq-5.1700au-DUCC3.out-xacc"
ham_n230d3="../../N2/cc-pVTZ/3.0_Eq-6.2040au/DUCC3/3.0_Eq-6.2040au-DUCC3.out-xacc"

ham_n210d2="../../N2/cc-pVTZ/1.0_Eq-2.0680au/DUCC2/1.0_Eq-2.0680au-DUCC2.out-xacc"
ham_n215d2="../../N2/cc-pVTZ/1.5_Eq-3.1020au/DUCC2/1.5_Eq-3.1020au-DUCC2.out-xacc"
ham_n220d2="../../N2/cc-pVTZ/2.0_Eq-4.1360au/DUCC2/2.0_Eq-4.1360au-DUCC2.out-xacc"
ham_n225d2="../../N2/cc-pVTZ/2.5_Eq-5.1700au/DUCC2/2.5_Eq-5.1700au-DUCC2.out-xacc"
ham_n230d2="../../N2/cc-pVTZ/3.0_Eq-6.2040au/DUCC2/3.0_Eq-6.2040au-DUCC2.out-xacc"


ham_n210d0="../../N2/cc-pVTZ/1.0_Eq-2.0680au/Bare/1.0_Eq-2.0680au-Bare.out-xacc"
ham_n215d0="../../N2/cc-pVTZ/1.5_Eq-3.1020au/Bare/1.5_Eq-3.1020au-Bare.out-xacc"
ham_n220d0="../../N2/cc-pVTZ/2.0_Eq-4.1360au/Bare/2.0_Eq-4.1360au-Bare.out-xacc"
ham_n225d0="../../N2/cc-pVTZ/2.5_Eq-5.1700au/Bare/2.5_Eq-5.1700au-Bare.out-xacc"
ham_n230d0="../../N2/cc-pVTZ/3.0_Eq-6.2040au/Bare/3.0_Eq-6.2040au-Bare.out-xacc"


if __name__ == '__main__':
    ####
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='GCIM description')
    parser.add_argument('mole_code', type=str,
                        help='(Requiered) Mole code name from {bz,bztz,n210,n215,n220,n225,n230}')
    parser.add_argument('--stop_iter', type=int, default=25,
                        help='(Optional) Stop Iteration')
    parser.add_argument('--stop_thresh', type=float, default=1e-6,
                        help='(Optional) Stop Threshold')
    parser.add_argument('--ducc', type=int, default=3,
                        help='(Optional) DUCC')
    parser.add_argument('--num_electrons', type=int, default=6,
                        help='(Optional) Number of electrons')
    parser.add_argument('--num_orbitals', type=int, default=6,
                        help='(Optional) Number of orbitals')
    parser.add_argument('--adapt_opt_intvl', type=int, default=0,
                        help='(Optional) ADAPT optimization interval')
    parser.add_argument('--opt_depth', type=int, default=0,
                        help='(Optional) VQE optimization depth')
    parser.add_argument('--adapt_maxiter', type=int, default=150,
                        help='(Optional) ADAPT maximum iterations')
    args = parser.parse_args()

    mole_codename = args.mole_code
    n_orb = args.num_orbitals
    n_a = args.num_electrons//2
    n_b = args.num_electrons//2
    ducc_lvl = args.ducc
    adapt_opt_intvl = args.adapt_opt_intvl
    opt_depth = args.opt_depth
    adapt_maxiter = args.adapt_maxiter

    stop_iter = args.stop_iter
    stop_thresh = args.stop_thresh

    mol_name =  f'{mole_codename}{args.num_electrons}{args.num_orbitals}DUCC{ducc_lvl}'
    if mole_codename == 'bz':
        file_path = ham_bz66
    elif mole_codename == 'bztz':
        file_path = ham_bztz66
    elif mole_codename == 'fbp':
        file_path = ham_fbp
    elif mole_codename == 'n210':
        if ducc_lvl == 0:
            file_path = ham_n210d0
        elif ducc_lvl == 2:
            file_path = ham_n210d2
        elif ducc_lvl == 3:
            file_path = ham_n210d3
        else:
            raise Exception(f"N2 has no DUCC {ducc_lvl}")
    elif mole_codename == 'n215':
        if ducc_lvl == 0:
            file_path = ham_n215d0
        elif ducc_lvl == 2:
            file_path = ham_n215d2
        elif ducc_lvl == 3:
            file_path = ham_n215d3
        else:
            raise Exception(f"N2 has no DUCC {ducc_lvl}")
    elif mole_codename == 'n220':
        if ducc_lvl == 0:
            file_path = ham_n220d0
        elif ducc_lvl == 2:
            file_path = ham_n220d2
        elif ducc_lvl == 3:
            file_path = ham_n220d3
        else:
            raise Exception(f"N2 has no DUCC {ducc_lvl}")
    elif mole_codename == 'n225':
        if ducc_lvl == 0:
            file_path = ham_n225d0
        elif ducc_lvl == 2:
            file_path = ham_n225d2
        elif ducc_lvl == 3:
            file_path = ham_n225d3
        else:
            raise Exception(f"N2 has no DUCC {ducc_lvl}")
    elif mole_codename == 'n230':
        if ducc_lvl == 0:
            file_path = ham_n230d0
        elif ducc_lvl == 2:
            file_path = ham_n230d2
        elif ducc_lvl == 3:
            file_path = ham_n230d3
        else:
            raise Exception(f"N2 has no DUCC {ducc_lvl}")
    else:
        raise Exception('Mole code name from {bz,bztz,n210,n215,n220,n225,n230}')

    print("Received Arguments")
    print("  - Mole code:", mole_codename)
    print("  - Input file:", file_path)
    print(f"  - Stopping Criteria: change < {stop_thresh} in {stop_iter} iterations")
    print("  - Number of electrons:", args.num_electrons)
    print("  - Number of orbitals:", args.num_orbitals)
    print("  - DUCC:", args.ducc)
    print("  - ADAPT maximum iterations:", args.adapt_maxiter)
    print("  - ADAPT optimization interval:", args.adapt_opt_intvl)
    print("  - VQE optimization depth:", args.opt_depth)
    print("  - Target molecule:", mol_name)

    ###
    tstart_total = timer()

    fermi_ham = parse_hamiltonian(file_path, n_orb, use_interleaved=True, return_type = 'of')
    spin_complete_pool = False
    pauli_pool = False
    make_orth = False
    theta = np.pi/4
    use_gcmgrad = False
    #
    occupied_list = []
    for i in range(n_a):
        occupied_list.append(i*2)
    for i in range(n_b):
        occupied_list.append(i*2+1)
    print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
    reference_ket = scipy.sparse.csc_matrix(of.jw_configuration_state(occupied_list, 2*n_orb)).transpose()
    #
    custom_str = '_tp4'
    plot_str = 'π/4'

    file_suffix = ''
    file_prefix = 'Output_Data/ADAPT-GCM/GCM_{:s}{:s}{:s}/'.format(mol_name,custom_str,file_suffix)
    if adapt_opt_intvl > 0:
        file_prefix = 'Output_Data/ADAPT-GCM-OPT/GCMOPT_{:s}{:s}{:s}_{:d}_{:d}/'.format(mol_name,custom_str,file_suffix,
                                                                                        adapt_opt_intvl, opt_depth)
        
    import os
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)
    # else:
    #     raise Warning("Folder already exists, please change the file_prefix")

    # file_prefix = 'Output_Data/ADAPT-GCM/GCM_Test/'

    print(" Use Spin-complete Pool: ", spin_complete_pool)
    print(" Saved in folder: ", file_prefix)

    pool = operator_pools.UCCSD()
    pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)
    sys.stdout.flush()
    print("Finish Pool Construction.\n\n")

    #----------------------- Modification ------------------#
    """
    A few parameters added by Muqing:
    file_prefix (str, optional): file path for saving GCM matrices in each iteration END with '/'. 
                                None for not saving matrices. 
                                If path does not exists, it does NOT create the path. 
                                Defaults to None. 
    no_printout_level (int, optional): 0 means print out everything, 
                                        1 means no VQE parameter optimization info, grad print thresh*100, 
                                        2 means ALSO no ansatz info in each VQE iteration, no grad info. 
                                        Defaults to 0.
    make_orth (bool, optional): If make basis orthgonal when construct matrices for GCM. 
                                Defaults to False. 
    theta_thresh is for VQE convergence, origianlly in adapt_vqe()
    """
    # file_prefix = None
    # [e,v,params,GCM_DIFFS,VQE_DIFFS,GCM_BASIS_SIZE] = adapt_vqe_gcm(fermi_ham, pool, reference_ket,
    #                                                                         file_prefix=file_prefix,
    #                                                                         theta_thresh=1e-9,
    #                                                                         no_printout_level=2, 
    #                                                                         make_orth=make_orth)
    # temp_ham = of.linalg.get_sparse_operator(fermi_ham)
    # true_lowest_ev = scipy.sparse.linalg.eigsh(temp_ham,1,which='SA')[0][0].real
    # np.floor(true_lowest_ev)-1
    # print(f"!!! Convergence criteria: GCM errors do not change(<1e-{10}) for {20} iterations!!!")
    if adapt_opt_intvl == 0:
        VQE_ITERS = None
        [GCM_EVS,GCM_Indices, GCM_DIFFS, GCM_BASIS_SIZE] = adapt_gcm(fermi_ham, pool, reference_ket,
                                        theta = theta,
                                        use_gcmgrad  = use_gcmgrad,
                                        use_qugcm    = False,
                                        file_prefix  = file_prefix,
                                        stop_type    = 'long_flat',
                                        stop_thresh  = (stop_iter,stop_thresh),
                                        adapt_maxiter= adapt_maxiter,
                                        no_printout_level=2, 
                                        make_orth = make_orth,
                                        ev_thresh = 1e-12)
    else:
        [GCM_EVS,GCM_Indices, GCM_DIFFS, GCM_BASIS_SIZE, VQE_ITERS] = adapt_gcm_opt(fermi_ham, pool, reference_ket,
                                        theta = theta,
                                        adapt_opt_intvl = adapt_opt_intvl,
                                        opt_depth = opt_depth,
                                        use_gcmgrad  = use_gcmgrad,
                                        use_qugcm    = False,
                                        file_prefix  = file_prefix,
                                        stop_type    = 'long_flat',
                                        stop_thresh  = (stop_iter,stop_thresh),
                                        adapt_maxiter= adapt_maxiter,
                                        no_printout_level=2, 
                                        make_orth = make_orth,
                                        ev_thresh = 1e-12)
    tend_total = timer()
    print("\n\n\n\n\n>>> Total GCIM time (sec): ", tend_total-tstart_total)
    #----------------------- Modification ------------------#
    print("Selected indices:", GCM_Indices)
    print("Eigenvalues:",GCM_EVS)

    ## H6_2au_DUCC3_6-electrons_6-Orbitals
    ## - Spin adapted: (5,2) (10,1e-6)
    ## - Complete: (3,3) (25,1e-6)

    ## H6_3au_DUCC3_6-electrons_6-Orbitals
    ## - Spin adapted: (5,2) (10,1e-6)
    ## - Complete: (5,2) (10,1e-6)


    np.save(file_prefix+'GCM_DIFF.npy', np.array(GCM_DIFFS))
    np.save(file_prefix+'GCM_SIZE.npy', np.array(GCM_BASIS_SIZE))
    np.save(file_prefix+'GCM_EVS.npy', np.array(GCM_EVS))
    np.save(file_prefix+'GCM_Indices.npy', np.array(GCM_Indices))
    if VQE_ITERS:
        np.save(file_prefix+'VQE_NITERS.npy', np.array(VQE_ITERS))



    ## First good iteration
    print("\nVerfiy eigenvector correctness")
    mole_dict = {'mole_name': mol_name,'iter_num': len(GCM_Indices)-1,'nele': n_a+n_b,'purt': 1e-12}
    _,_ = gcm_res_validation(mole_dict , file_prefix)


    index_start = 0
    index_end = len(GCM_DIFFS) # not included
    iter_range = list(range( index_start, index_end ))
    # plt.plot(iter_range, np.array(VQE_DIFFS)[iter_range], linewidth=4, label='ADAPT VQE')
    plt.plot(iter_range, np.array(GCM_DIFFS)[iter_range], color = 'b', linewidth=4, label='ADAPT GCM, θ={:s}'.format(plot_str))
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error from ground-state FCI energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_prefix+mol_name+'.png', dpi=300,facecolor='white', edgecolor='white')
    plt.clf()
