# Code and Data for Experiments in the Paper "Coupled Cluster Downfolding Theory in Simulations of Chemical Systems on Quantum Hardware"

## UCCGSD

**Author:** Nathan M. Myers

To reproduce our results in the paper, see `./Quantum_experiments/UCCGSD/PL_UCCGSD_H_conversion.ipynb`. 

The package versions used here are:
```text
Python 3.13.3
Pennylane==0.41.0
```


## ADAPT-GCIM

**Author**: Muqing Zheng

All code resides in 
```
./Quantum_experiments/ADAPT-GCIM/src/
```

Iterative printouts for each molecule are saved in files named
```
{method_name}_{molecule_name}{# of orbitals}{# of electrons}ducc3.txt
```
 where
 - `method_name` is `gcim` for ADAPT-CGIM or `gcimopt22` for ADAPT-GCIM(2,2)
 - `molecule_name` is one of
   - `bz` for Benzene (ccp-VDZ)
   - `bztz` for Benzene (ccp-VTZ basis)
   - `n2{#}` for N2 with different bond lengths
   - `fbp` for free-base porphyrin
 - `# of orbitals' and '# of electrons` are both `6`

**The 1st line of each text file contains the correspinding command.**


To reproduce the results
1. please first download three python files `pyscf_helper.py`, `tVQE.py`, `vqe_methods.py` from [our repository](https://github.com/Firepanda415/adapt-vqe-for-gcim/tree/master) forked from [original ADAPT-VQE repo](https://github.com/mayhallgroup/adapt-vqe). We update all three files for package version changes.

2. Run command such as: `$ python adgcm_main.py bz --stop_iter 30 --stop_thresh 1e-8`



The package versions used here are:
```text
python=3.12.4
openfermion==1.6.1
openfermionpsi4==0.5
pyscf==2.6.2
```
Note that, if your computer uses Apple M series chip, please install PySCF as
```text
ARCHFLAGS="-arch arm64e" pip install pyscf==2.6.2
```



## ADAPT-VQE and Qubit-ADAPT-VQE

**Author**: Muqing Zheng

For questions about NWQSim, please contact the NWQSim developers directly.


Iterative printouts for each molecule appear in
```
./Quantum_experiments/{method_name}/printout_{optional_phase}{molecule_name}ducc3.txt
```
where
 - `method_name` is either `ADAPT-VQE` or `Qubit-ADAPT-VQE`
 - `molecule_name` is one of
  -  bz66` for Benzene (ccp-VDZ)
  - `bztz66` for Benzene (ccp-VTZ)
  - `n2{#}` for N2 with different bond lengths
  - `fbp` for free-base porphyrin

 **The 1st line of each text file contains the correspinding command.**

To reproduce our results
1. please install [NWQSim](https://github.com/pnnl/NWQ-Sim) and use the built-in ADAPT-VQE and Qubit-ADAPT-VQE implementaitons. See the corresponding [guide](https://github.com/pnnl/NWQ-Sim/blob/main/vqe/README.md).
2. Run commands such as `$ ./vqe/nwq_vqe -f ./Input_Data/Benzene/cc-pVTZ/FrozenCoreCCSD_6Elec_6Orbs/DUCC3/benzene-FC-CCSD-6Electrons-6Orbitals.out-xacc -p 6 -v --abstol 1e-8 -lb -3.1 -ub 3.1 --maxeval 1000 -o LN_BOBYQA --adapt -ag 1e-4
`





## Experiments on Quantinuum and IBM devices

**Author**: Muqing Zheng

All data are collected in
```
./Quantum_experiments/Real_hardware/Result Summary.xlsx
```

**To generate IBM data**
The `Benzene_IBM_QESEM.ipynb` and `FBP_IBM_QESEM.ipynb` are two notebooks we used with the QESEM error mitigation software from Qedma.

**To generate Quantinuum data**
The `Benzene_quantinuum_emulator.ipynb` and `FBP_quantinuum_emulator` are two notebooks showing how we chop Hamiltonians, construct circuits, optimize parameters, transpile circuits, submit jobs to QNEXUS, and post-process the returned data. We used the exactly the same code for submitting jobs to H1-1 quantum computer.


The package versions used are:
```
python=3.12.4
openfermion==1.6.1
pytket==1.41.0
pytket-qiskit==0.60.0
qnexus==0.11.0
quantinuum-schemas==2.0.0
qiskit==1.4.0
qiskit-aer==0.16.4
qiskit-algorithms==0.3.0
qiskit-ibm-catalog==0.4.0
qiskit-ibm-provider==0.11.0
qiskit-ibm-runtime==0.30.0
qiskit-nature==0.7.2
qiskit-qasm3-import==0.5.1
qiskit-serverless==0.20.0
```