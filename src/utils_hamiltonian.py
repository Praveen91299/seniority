
###visualization and saving utils
import py3Dmol
import pyscf.tools
from math import sin, cos, radians
import numpy as np
import pyscf as ps
from pyscf import mcscf
import pickle



def get_h2o_geom(R=1.0, theta=None):
    if theta is None:
        theta = radians(107.6 / 2)
    
    xH2O = sin(theta)*R
    zH2O = cos(theta)*R

    geo = [ ['O', [0,0,0]], ['H', [-xH2O,0, zH2O]], ['H', [xH2O,0, zH2O]] ]
    return geo

def get_n2_geom(R=1.0):
    zN = 0.5*R
    geo = [ ['N', [0,0, zN]], ['N', [0,0, -zN]] ]

    return geo

def generate_hamiltonian_integrals(moltag, basis_set, geo, ncas = 6, nelecas = 8, file_prefix='./', geo_desc='', run_fci=False, fci_ncas=7, fci_nelec=10):

    mol = ps.gto.Mole(atom = geo, basis = basis_set, verbose = 4)
    mol.build()
    mf = ps.scf.RHF(mol)
    mf.kernel()
    print(dir(mf))
    fock_matrix = mf.get_fock()
    print(f'\nFock Matrix in Spatial AO basis:\n')
    print_matrix(fock_matrix)
    cmo_coef_in_ao_basis = mf.mo_coeff
    print(f'\ncmo coefficients in ao basis\n')
    print_eigen_solution(mf.mo_energy,cmo_coef_in_ao_basis)
    orbene_rhf = mf.mo_energy

    # filnam = f'{file_prefix}{moltag}_{basis_set}_{geo_desc}_rhf'
    # save_molecular_orbitals(mol, mf, filnam,[i for i in range(ncas)])

    mcscf_calc = mcscf.CASSCF(mf, ncas, nelecas)
    #Targetting singlet state
    mcscf_calc.fix_spin_(ss=0)
    print(dir(mcscf_calc))
    mcscf_calc.kernel()
    fock_matrix_mcscf = mcscf_calc.get_fock()
    print(f'\nMCSCF Fock Matrix')
    print_matrix(fock_matrix_mcscf)
    print(f'\ncas cmo coefficients in ao basis\n')
    idx = mcscf_calc.mo_energy.argsort()
    orbene_mcscf_sorted = mcscf_calc.mo_energy[idx]
    mocoef_mcscf_sorted = mcscf_calc.mo_coeff[:,idx]
    print_eigen_solution(orbene_mcscf_sorted,mocoef_mcscf_sorted)
    mcscf_calc.verbose = 4
    mcscf_calc.analyze()
    mcscf_calc_2nd = mcscf.CASSCF(mf, ncas, nelecas)
    #Targetting singlet state
    mcscf_calc_2nd.fix_spin_(ss=0)
    mcscf_calc_2nd.kernel(mocoef_mcscf_sorted)
    print_eigen_solution(mcscf_calc_2nd.mo_energy,mcscf_calc_2nd.mo_coeff)
    mcscf_calc_2nd.verbose = 4
    mcscf_calc_2nd.analyze()
    filnam = f'{file_prefix}{moltag}_{basis_set}_{ncas}o{nelecas}e_{geo_desc}'
    #save_molecular_orbitals(mol, mcscf_calc_2nd, filnam,[0,1,2,3,4,5,6]) #time consuming. Comment off only when needed.
    Ct_F_C = mcscf_calc_2nd.mo_coeff.transpose()@fock_matrix_mcscf@mcscf_calc_2nd.mo_coeff
    print_eigen_solution(mcscf_calc_2nd.mo_energy,mcscf_calc_2nd.mo_coeff)
    orbene_mcscf = mcscf_calc_2nd.mo_energy
    print(f'\nCt@F@C matrix:')
    print_matrix(Ct_F_C)

    h1e_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    s_ao = mol.intor('int1e_ovlp')
    g_ao = mol.intor('int2e')
    E_nuc_repulsion = mf.energy_nuc()
    print(f'\nNuclear repulsion. Constant term in Hamiltonian: {E_nuc_repulsion}')

    print('h1e_ao:')
    print_matrix(h1e_ao)
    print('s_ao:')
    print_matrix(s_ao)
    #print(g_ao)

    h1e_mcscf_cmo = mocoef_mcscf_sorted.transpose()@h1e_ao@mocoef_mcscf_sorted
    h1e_rhf_cmo = cmo_coef_in_ao_basis.transpose()@h1e_ao@cmo_coef_in_ao_basis
    s_mcscf_cmo = mocoef_mcscf_sorted.transpose()@s_ao@mocoef_mcscf_sorted
    O = mocoef_mcscf_sorted
    #The psqr,pa,qb,rc,sd->abcd below is to change the g_ao in chemist notation to g_mcscf_cmo in physicist notation.
    #The 0.5 multiplication corresponds to the 1/2 multiplication in the most symmetric 
    #2nd quantizaiton form of 2el operator.
    #g_mcscf_cmo also needs to be expanded from spatial orbitals to spin orbitals.
    g_mcscf_cmo = 0.5*np.einsum('psqr,pa,qb,rc,sd->abcd',g_ao,O,O,O,O)
    O = cmo_coef_in_ao_basis
    g_rhf_cmo = 0.5*np.einsum('psqr,pa,qb,rc,sd->abcd',g_ao,O,O,O,O)

    nmo = s_mcscf_cmo.shape[0]
    n_spinmo = 2*nmo
    n_elec = mol.nelectron

    phys_filename = f'{file_prefix}{moltag}_{basis_set}_rhf_phys_spatial_{geo_desc}'
    with open(phys_filename, 'wb') as f:
        pickle.dump([E_nuc_repulsion,h1e_rhf_cmo,g_rhf_cmo,orbene_rhf,n_elec],f)

    phys_filename = f'{file_prefix}{moltag}_{basis_set}_{ncas}o{nelecas}e_phys_spatial_{geo_desc}'
    with open(phys_filename, 'wb') as f:
        pickle.dump([E_nuc_repulsion,h1e_mcscf_cmo,g_mcscf_cmo,orbene_mcscf,n_elec,ncas, nelecas],f)

    if run_fci:
        print('\nRunning full CI calculation with CAS: {}, Electrons: {}'.format(fci_ncas, fci_nelec))
        mcscf_calc_3rd = mcscf.CASSCF(mf, fci_ncas, fci_nelec)
        #Targetting singlet state
        mcscf_calc_3rd.fix_spin_(ss=0)
        mcscf_calc_3rd.kernel(mocoef_mcscf_sorted)
        mcscf_calc_3rd.verbose = 4
        mcscf_calc_3rd.analyze()

    return

def save_molecular_orbitals(mol, mf, filename_base,list_save=[]):
    if len(list_save) == 0:
        for i in range(mol.nao):
            pyscf.tools.cubegen.orbital(mol, filename_base + f'_{i}.cube', mf.mo_coeff[:,i])
    else:
        for i in list_save:
            pyscf.tools.cubegen.orbital(mol, filename_base + f'_{i}.cube', mf.mo_coeff[:,i])
    
    return None

def start_vis(geo=None):
    view = py3Dmol.view()
    
    if geo is None:
        return view
    
    view.addModel(geo, 'xyz')
    return view

def stylize_view(view):
    view.addStyle({
        'stick'  : {'radius' : 0.05},
        'sphere' : {'scale' : 0.1}
    })


def draw_orbital(view, filename, isoval=0.04):
    with open(filename, 'r') as f:
        cube_data = f.read()

    view.addVolumetricData(
        cube_data,
        'cube',
        {
            'isoval'  : isoval, 
            'color'   : 'blue', 
            'opacity' : 0.75
        }
    )

    view.addVolumetricData(
        cube_data,
        'cube',
        {
            'isoval'  : -isoval,
            'color'   : 'red',
            'opacity' : 0.75
        }
    )

    return None

def print_matrix(matrix,n_per_group=6):

    nrow = matrix.shape[0]
    ncolumn = matrix.shape[1]

    ngroup = ncolumn // n_per_group
    nleft = ncolumn - ngroup*n_per_group

    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))
    
        print("")
   
    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = ncolumn
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

def print_eigen_solution(eigen_values,eigen_vectors):

    nsolut = len(eigen_values)
    n_per_group = 5
    ngroup = nsolut // n_per_group
    nleft = nsolut - ngroup*n_per_group
   #print(nsolut,ngroup,nleft)
    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = nsolut
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")
