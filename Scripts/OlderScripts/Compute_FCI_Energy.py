from pyscf import fci
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import cc
import sys
import yaml
from yaml import SafeLoader
import numpy as np
from numpy import linalg as LA
from pprint import pprint


def get_spatial_integrals(one_electron, two_electron, n_orb):
    one_electron_spatial_integrals = np.zeros((n_orb, n_orb))
    two_electron_spatial_integrals = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for ind, val in enumerate(one_electron):
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        one_electron_spatial_integrals[i, j] = val[2]
        if i != j:
            one_electron_spatial_integrals[j, i] = val[2]

    for ind, val in enumerate(two_electron):
        i = int(val[0]-1)
        j = int(val[1]-1)
        k = int(val[2]-1)
        l = int(val[3]-1)
        two_electron_spatial_integrals[i, j, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, i, j] == 0:
            two_electron_spatial_integrals[k, l, i, j] = val[4]
        if two_electron_spatial_integrals[i, j, l, k] == 0:
            two_electron_spatial_integrals[i, j, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, i, j] == 0:
            two_electron_spatial_integrals[l, k, i, j] = val[4]
        if two_electron_spatial_integrals[j, i, k, l] == 0:
            two_electron_spatial_integrals[j, i, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, j, i] == 0:
            two_electron_spatial_integrals[k, l, j, i] = val[4]
        if two_electron_spatial_integrals[j, i, l, k] == 0:
            two_electron_spatial_integrals[j, i, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, j, i] == 0:
            two_electron_spatial_integrals[l, k, j, i] = val[4]

    return one_electron_spatial_integrals, two_electron_spatial_integrals


def parse_fci_vector(ci_vecmat, orbs, electrons):
    from pyscf.fci import cistring
    conf_bin = cistring.gen_strings4orblist(list(range(orbs)), electrons/2)

    OCCMAP = {('0', '0'): '0',
              ('1', '0'): 'a',
              ('0', '1'): 'b',
              ('1', '1'): '2'}

    civecs = {}
    for i, ca in enumerate(conf_bin):
        for j, cb in enumerate(conf_bin):
            astring = bin(ca)[2:].zfill(orbs)
            bstring = bin(cb)[2:].zfill(orbs)
            s = ''.join(reversed([OCCMAP[a, b]
                        for a, b in zip(astring, bstring)]))
            civecs[s] = ci_vecmat[i, j]
    return civecs

def parse_fci_vector_single(ci_vecmat, orbs, electrons):
    from pyscf.fci import cistring
    conf_bin = cistring.gen_strings4orblist(list(range(orbs)), electrons/2)

    OCCMAP = {('0', '0'): '0',
              ('1', '0'): 'a',
              ('0', '1'): 'b',
              ('1', '1'): '2'}

    civecs = {}
    for i, ca in enumerate(conf_bin):
        astring = bin(ca)[2:].zfill(orbs)
        bstring = bin(ca)[2:].zfill(orbs)
        s = ''.join(reversed([OCCMAP[a, b]
                    for a, b in zip(astring, bstring)]))
        civecs[s] = ci_vecmat[i]
    return civecs


if len(sys.argv) != 4:
    print("Command Line Error : python3 FCI-PYSCF [yaml file] [#Frzn Core] [# Roots]")
    exit()

np.set_printoptions(formatter={'float_kind': '{:f}'.format})
data = yaml.load(open(sys.argv[1], "r"), SafeLoader)
frzncore =  int(sys.argv[2])
roots = int(sys.argv[3])
threshold = 0.001

if (data['format']['version'] == '0.3'):
    n_electrons = data['problem_description'][0]['n_electrons']
    n_spatial_orbitals = data['problem_description'][0]['n_orbitals']
    nuclear_repulsion_energy = data['problem_description'][0]['coulomb_repulsion']['value']
    one_electron_import_ = data['problem_description'][0]['hamiltonian']['one_electron_integrals']['values']
    two_electron_import_ = data['problem_description'][0]['hamiltonian']['two_electron_integrals']['values']
    one_electron_import = []
    two_electron_import = []
    for kv in one_electron_import_:
        item = kv['key']
        item.append(kv['value'])
        one_electron_import.append(item)
    for kv in two_electron_import_:
        item = kv['key']
        item.append(kv['value'])
        two_electron_import.append(item)
else:
    n_electrons = data['integral_sets'][0]['n_electrons']
    n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
    nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
    one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
    two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

one_electron_spatial_integrals, two_electron_spatial_integrals = get_spatial_integrals(
    one_electron_import, two_electron_import, n_spatial_orbitals)

#######################################
# Hartree-Fock Energy and Core Freezing
#######################################

print('File = ', sys.argv[1])
print('Orbitals = ', n_spatial_orbitals)
print('Frozen Orbitals = ', frzncore)
print('Correlated Orbitals = ', n_spatial_orbitals - frzncore)
print('Electrons = ', n_electrons)
print('Correlated Electrons = ', n_electrons - 2*frzncore)
print('\nRepulsion Energy = %.12f' % (nuclear_repulsion_energy))

# Compute the Hartree--Fock energy
Energy = nuclear_repulsion_energy
for x in range(int(n_electrons/2)):
    Energy += 2*one_electron_spatial_integrals[x,x]
    for y in range(int(n_electrons/2)):
        Energy += 2*two_electron_spatial_integrals[x,x,y,y] - two_electron_spatial_integrals[x,y,y,x]
print('SCF Energy = %.12f' % (Energy))

 # Compute the core/offset energy
FrznCore_Energy = 0.0
for x in range(frzncore):
    FrznCore_Energy += 2*one_electron_spatial_integrals[x,x]
    for y in range(frzncore):
        FrznCore_Energy += 2*two_electron_spatial_integrals[x,x,y,y] - two_electron_spatial_integrals[x,y,y,x]
print('Core Energy = %.12f' % (FrznCore_Energy))
print('New Repulsion Energy = %.12f' % (FrznCore_Energy+nuclear_repulsion_energy))

# Form Fock operator
Fock = np.zeros((n_spatial_orbitals,n_spatial_orbitals))
for x in range(n_spatial_orbitals):
    for y in range(n_spatial_orbitals):
        Fock[x,y] += one_electron_spatial_integrals[x,y]
        for z in range(int(n_electrons/2)):
            Fock[x,y] += 2*two_electron_spatial_integrals[z,z,x,y] - two_electron_spatial_integrals[z,y,x,z]
print("\nNew orbital energies after downfolding")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
Orb_E,Orb_Eig = LA.eig(Fock)
for x in sorted(Orb_E):
    if sorted(Orb_E).index(x) < frzncore:
        print('%.6f    FROZEN' % (x))
    else:
        print('%.6f' % (x))
print('')

# Zero out frozen core part
for x in range(n_spatial_orbitals):
    for y in range(n_spatial_orbitals):
        if x < frzncore:
            Fock[x,y] = 0.0
        if y < frzncore:
            Fock[x,y] = 0.0

# Remove two-electron contribution
for x in range(frzncore,n_spatial_orbitals):
    for y in range(frzncore,n_spatial_orbitals):
        for z in range(frzncore,int(n_electrons/2)):
            Fock[x,y] -= 2*two_electron_spatial_integrals[z,z,x,y] - two_electron_spatial_integrals[z,y,x,z]

one_electron_spatial_integrals = Fock[frzncore:,frzncore:]
two_electron_spatial_integrals = two_electron_spatial_integrals[frzncore:,frzncore:,frzncore:,frzncore:]
n_spatial_orbitals = n_spatial_orbitals - frzncore
n_electrons = n_electrons - 2*frzncore

############
mol = gto.M()
fake_hf = scf.RHF(mol)
fake_hf._eri = two_electron_spatial_integrals
fake_hf.get_hcore = lambda *args: one_electron_spatial_integrals

mo_coeff = np.eye(n_spatial_orbitals)
mo_occ = np.zeros(n_spatial_orbitals)
mo_occ[:int(n_electrons/2)] = 2

#mycc = cc.ccsd.CCSD(fake_hf, mo_coeff=mo_coeff, mo_occ=mo_occ)
mycc = cc.CCSD(fake_hf, mo_coeff=mo_coeff, mo_occ=mo_occ)
mycc.diis_start_cycle = 0
mycc.diis_start_energy_diff = 1e2
mycc.verbose = 4

ecc = mycc.kernel()
print('CCSD correlation energy = %.15g' % mycc.e_corr)

#         # Pseudo code how eris are implemented:
# nocc = int(n_electrons/2)
# eris_oooo = fake_hf._eri[:nocc,:nocc,:nocc,:nocc].copy()
# eris_ovoo = fake_hf._eri[:nocc,nocc:,:nocc,:nocc].copy()
# eris_ovvo = fake_hf._eri[nocc:,:nocc,nocc:,:nocc].copy()
# eris_ovov = fake_hf._eri[nocc:,:nocc,:nocc,nocc:].copy()


        # nmo = self.nmo
        # nvir = nmo - nocc
        # eris = _ChemistsERIs()
        # eri = ao2mo.incore.full(self._scf._eri, mo_coeff)
        # eri = ao2mo.restore(1, eri, nmo)
        # eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
        # eris.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
        # eris.ovvo = eri[nocc:,:nocc,nocc:,:nocc].copy()
        # eris.ovov = eri[nocc:,:nocc,:nocc,nocc:].copy()
        # eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
        # ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
        # eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir))
        # eris.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:], nvir)
        # eris.fock = numpy.diag(self._scf.mo_energy)
        # return eris
# eris = mycc.ao2mo(mo_coeff=mo_coeff)



############
# FCI Solver
############

cis = fci.direct_nosym.FCISolver
p = cis()
e, fcivec = p.kernel(one_electron_spatial_integrals, two_electron_spatial_integrals,
                     n_spatial_orbitals, n_electrons, max_space=450, nroots=roots, verbose=5)


if roots == 1:
    print('\nE = %.12f  2S+1 = %.7f' %
          (e+nuclear_repulsion_energy+FrznCore_Energy, p.spin_square(fcivec, n_spatial_orbitals, n_electrons)[1]))
    print(e+nuclear_repulsion_energy+FrznCore_Energy-Energy)

    fcivecorb = parse_fci_vector_single(fcivec, n_spatial_orbitals, n_electrons)
    fcivec_short = {}
    for key in fcivecorb:
        if (abs(fcivecorb[key]) > threshold):
            fcivec_short[key] = round(fcivecorb[key], 5)
    pprint(sorted(fcivec_short.items(), key=lambda kv: (
        abs(kv[1]), kv[0]), reverse=True))

else:
    for i in range(roots):
        print('\nE = %.12f  2S+1 = %.7f' %
              (e[i]+nuclear_repulsion_energy+FrznCore_Energy, p.spin_square(fcivec[i], n_spatial_orbitals, n_electrons)[1]))

        fcivecorb = parse_fci_vector(
            fcivec[i], n_spatial_orbitals, n_electrons)
        fcivec_short = {}
        for key in fcivecorb:
            if (abs(fcivecorb[key]) > threshold):
                fcivec_short[key] = round(fcivecorb[key], 5)
        pprint(sorted(fcivec_short.items(), key=lambda kv: (
            abs(kv[1]), kv[0]), reverse=True))
