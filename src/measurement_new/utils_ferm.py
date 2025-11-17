import numpy as np
import scipy
from openfermion import (
    FermionOperator,
    hermitian_conjugated
)

def obt_phys_spatial_to_spin(obt_phys_spatial):
    """
    Convert the obt of spatial orbitals to obt of spin orbitals
    """

    n_spatial = obt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    obt_phys_spin = np.zeros([n_spin,n_spin])
    obt_phys_spin[0:n_spin:2,0:n_spin:2] = obt_phys_spatial
    obt_phys_spin[1:n_spin:2,1:n_spin:2] = obt_phys_spatial

    return obt_phys_spin

def tbt_phys_spatial_to_spin(tbt_phys_spatial):
    """
    Convert the tbt in physicist notation of spatial orbitals 
    to tbt of spin orbitals in physicist notation
    """
 
    n_spatial = tbt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    tbt_phys_spin = np.zeros([n_spin,n_spin,n_spin,n_spin])
    tbt_phys_spin[0:n_spin:2,0:n_spin:2,0:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,1:n_spin:2,1:n_spin:2,1:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[0:n_spin:2,1:n_spin:2,1:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,0:n_spin:2,0:n_spin:2,1:n_spin:2] = tbt_phys_spatial

    return tbt_phys_spin


def orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_phys_spatial):
    """
    Given the orbital rotational angles and orbital pairs, transform the obt and tbt of spatial orbitals
    """

    assert len(x_orbrot) == len(list_orb_rot)

    n_spatialmo = obt_spatial.shape[0]
    kappa_mat = np.zeros([n_spatialmo,n_spatialmo])

    for ix, pair in enumerate(list_orb_rot):
        [iorb,jorb] = pair
        kappa_mat[iorb,jorb] = x_orbrot[ix]
        kappa_mat[jorb,iorb] = -x_orbrot[ix]

    Omat = scipy.linalg.expm(kappa_mat)

    obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_spatial,Omat,Omat,optimize=True)
    tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

    obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
    tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return obt_phys_spin_trans, tbt_phys_spin_trans

def make_short_H_ferm_op(const,obt_phys,tbt_phys):
    """
    Read in a hermitian fermionic operator and change it from sum p q r s
    to sum p>q, r>s for the 2-body terms
    """

    N = obt_phys.shape[0]

    H1 = FermionOperator()
    H2 = FermionOperator()
    for p in range(N):
        for q in range(p,N):
            if not np.isclose(obt_phys[p,q],0.0):
                coef = obt_phys[p,q]
                term = ((p,1), (q,0))
                H1 += FermionOperator(term,coef)
                if p != q:
                    H1 += hermitian_conjugated(FermionOperator(term,coef))

    for p in range(N):
        for q in range(p):
            for r in range(N):
                for s in range(r+1,N):
                    term = ((p,1), (q,1), (r,0), (s,0))
                    coef_coul = tbt_phys[p,q,r,s]
                    coef_exch = tbt_phys[p,q,s,r]
                    if np.isclose(coef_coul,0.0) and np.isclose(coef_exch,0.0):
                        continue
                    else:
                        H2 += FermionOperator(term,2.0*(coef_coul - coef_exch))
                    
    H_short = FermionOperator((),const)
    H_short += H1 + H2

    return(H_short)
