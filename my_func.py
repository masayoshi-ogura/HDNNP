# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

# calculate the symmetry functions and derivatives of it
### input
# comm,rank: MPI object
# atoms_objs: list of Atoms Object
# Rcs,Rss,etas: list of float
# lams,zetas: list of int
### output
# Gs: numpy array (nsample x natom x ninput)
# dGs: numpy array (nsample x natom x 3*natom * ninput)
def symmetric_func(comm, rank, atoms_objs, natom, nsample, ninput, Rcs, Rss, etas, lams, zetas):
    Gs = np.empty((nsample, natom, ninput))
    dGs = np.empty((nsample, natom, 3*natom, ninput))
    for m in range(nsample):
        # prepare R and cosine
        atoms = atoms_objs[m]
        atoms.set_cutoff(Rc)
        atoms.calc_connect()
        n_neighb = atoms.n_neighbours(rank+1) # 1-indexed
        R = distance_ij(rank, atoms)
        cosine = cosine_ijk(rank, atoms, n_neighb)
        # prepare just slightly deviated R and cosine for numerical derivatives
        # assume that neighbour atoms are not changed after displacing since dr is too small
        R_array,cosine_array = np.empty((1+2*3*natom,n_neighb)),np.empty((1+2*3*natom,n_neighb,n_neighb))
        R_array[0] = R; cosine_array[0] = cosine
        dr = 0.001
        for r in range(3*natom):
            k=r/3; alpha=r%3
            # prepare displaced R and cosine
            atoms_plus = atoms.copy(); atoms_plus.pos[k][alpha] += dr
            atoms_plus.calc_connect()
            R_array[2*r+1] = (distance_ij(rank, atoms_plus))
            cosine_array[2*r+1] = (cosine_ijk(rank, atoms_plus, n_neighb))
            atoms_minus = atoms.copy(); atoms_minus.pos[k][alpha] -= dr
            atoms_minus.calc_connect()
            R_array[2*r+2] = (distance_ij(rank, atoms_minus))
            cosine_array[2*r+2] = (cosine_ijk(rank, atoms_minus, n_neighb))
        
        G = np.empty((ninput, natom))
        dG = np.empty((ninput, 3*natom, natom))
        n = 0
        for Rc in Rcs:
            # prepare cutoff functions with Rc
            fc_array = cutoff_func(R_array, Rc)
            
            # G1
            G1_array = G1(comm, rank, natom, fc_array)
            G[n] = G1_array[0]
            for r in range(3*natom):
                dG[n][r] = (G1_array[2*r+1] - G1_array[2*r+2]) / (2 * dr)
            n += 1
            
            for eta in etas:
                # G2
                for Rs in Rss:
                    G2_array = G2(comm, rank, R_array, natom, fc_array, Rs, eta)
                    G[n] = G2_array[0]
                    for r in range(3*natom):
                        dG[n][r] = (G2_array[2*r+1] - G2_array[2*r+2]) / (2 * dr)
                    n += 1
                
                # G4
                for lam in lams:
                    for zeta in zetas:
                        G4_array = G4(comm, rank, R_array, cosine_array, natom, n_neighb, fc_array, eta, lam, zeta)
                        G[n] = G4_array[0]
                        for r in range(3*natom):
                            dG[n][r] = (G4_array[2*r+1] - G4_array[2*r+2]) / (2 * dr)
                        n += 1
        Gs[m] = G.T
        dGs[m] = dG.T
    return Gs, dGs

# calculate symmetric function type-1
# 2-bodies
def G1(comm, i, natom, fc):
    G,Gi = np.empty((natom,1+2*3*natom)),np.zeros((natom,1+2*3*natom))
    Gi[i] = np.sum(fc, axis=1)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate symmetric function type-2
# 2-bodies
def G2(comm, i, R, natom, fc, Rs, eta):
    G,Gi = np.empty((natom,1+2*3*natom)),np.zeros((natom,1+2*3*natom))
    gi = np.exp(- eta * (R - Rs) ** 2) * fc
    Gi[i] = np.sum(gi, axis=1)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate symmetric function type-4
# 3-bodies
def G4(comm, i, R, cosine, natom, n_neighb, fc, eta, lam, zeta):
    G,Gi = np.empty((natom,1+2*3*natom)),np.zeros((natom,1+2*3*natom))
    R_ij = np.exp(- eta * (R**2))[:,:,None].repeat(n_neighb,axis=2)
    R_ik = np.exp(- eta * (R**2))[:,None,:].repeat(n_neighb,axis=1)
    fc_ij = fc[:,:,None].repeat(n_neighb,axis=2)
    fc_ik = fc[:,None,:].repeat(n_neighb,axis=1)
    gi = ((1+lam*cosine)**zeta) * R_ij * R_ik * fc_ij * fc_ik
    filter = np.array([k == j for j in range(n_neighb) for k in range(n_neighb)]).repeat(1+2*3*natom).reshape((1+2*3*natom,n_neighb,n_neighb))
    gi[filter] = 0.0
    Gi[i] = (2 ** (1-zeta)) * np.sum(gi, axis=(1,2))
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate interatomic distances with neighbours
def distance_ij(i, atoms):
    R = np.array([con.distance for con in atoms.connect[i+1]])
    return R

# calculate anlges with neighbours
def cosine_ijk(i, atoms, n_neighb):
    cosine = np.zeros((n_neighb,n_neighb))
    for j in range(n_neighb):
        for k in range(n_neighb):
            if k == j:
                pass
            else:
                cosine[j][k] = atoms.cosine_neighbour(i+1,j+1,k+1)
    return cosine

# calculate cutoff function with neighbours
def cutoff_func(R, Rc):
    fc = np.tanh(1-R/Rc)**3
    return fc
