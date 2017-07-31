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
        G = np.empty((ninput, natom))
        dG = np.empty((ninput, 3*natom, natom))
        n = 0
        for Rc in Rcs:
            # prepare R and cosine
            atoms = atoms_objs[m]
            atoms.set_cutoff(Rc)
            atoms.calc_connect()
            R = distance_ij(rank, atoms)
            fc = cutoff_func(R, Rc)
            cosine = cosine_ijk(rank, atoms)
            
            # prepare just slightly deviated R and cosine for numerical derivatives
            # assume that neighbour atoms are not changed after displacing since dr is too small
            R_array,fc_array,cosine_array = [],[],[]
            R_array[0] = R; fc_array[0] = fc; cosine_array[0] = cosine
            dr = 0.0001
            for r in range(3*natom):
                k=r/3; alpha=r%3
                # prepare displaced R and cosine
                atoms_plus = atoms.copy(); atoms_plus.pos[k+1][alpha+1] += dr
                atoms_plus.calc_connect()
                R_array[2*r+1] = distance_ij(rank, atoms_plus)
                fc_array[2*r+1] = cutoff_func(R_array[2*r+1], Rc)
                cosine_array[2*r+1] = cosine_ijk(rank, atoms_plus)
                atoms_minus = atoms.copy(); atoms_minus.pos[k+1][alpha+1] -= dr
                atoms_minus.calc_connect()
                R_array[2*r+2] = distance_ij(rank, atoms_minus)
                fc_array[2*r+2] = cutoff_func(R_array[2*r+2], Rc)
                cosine_array[2*r+2] = cosine_ijk(rank, atoms_minus)
            
            # G1
            G[n] = G1(comm, rank, natom, fc_array[0])
            for r in range(3*natom):
                G1_plus = G1(comm, rank, natom, fc_array[2*r+1])
                G1_minus = G1(comm, rank, natom, fc_array[2*r+2])
                dG[n][r] = (G1_plus - G1_minus) / (2 * dr)
            n += 1
            
            for eta in etas:
                # G2
                for Rs in Rss:
                    G[n] = G2(comm, rank, R_array[0], natom, fc_array[0], Rs, eta)
                    for r in range(3*natom):
                        G2_plus = G2(comm, rank, R_array[2*r+1], natom, fc_array[2*r+1], Rs, eta)
                        G2_minus = G2(comm, rank, R_array[2*r+2], natom, fc_array[2*r+2], Rs, eta)
                        dG[n][r] = (G2_plus - G2_minus) / (2 * dr)
                    n += 1
                
                # G4
                for lam in lams:
                    for zeta in zetas:
                        G[n] = G4(comm, rank, R_array[0], cosine_array[0], natom, fc_array[0], eta, lam, zeta)
                        for r in range(3*natom):
                            G4_plus = G4(comm, rank, R_array[2*r+1], cosine_array[2*r+1], natom, fc_array[2*r+1], eta, lam, zeta)
                            G4_minus = G4(comm, rank, R_array[2*r+2], cosine_array[2*r+2], natom, fc_array[2*r+2], eta, lam, zeta)
                            dG[n][r] = (G4_plus - G4_minus) / (2 * dr)
                        n += 1
        Gs[m] = G.T
        dGs[m] = dG.T
    return Gs, dGs

# calculate symmetric function type-1
# 2-bodies
def G1(comm, i, natom, fc):
    G,Gi = np.empty(natom),np.zeros(natom)
    Gi[i] = np.sum(fc)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate symmetric function type-2
# 2-bodies
def G2(comm, i, R, natom, fc, Rs, eta):
    G,Gi = np.empty(natom),np.zeros(natom)
    gi = np.exp(- eta * (R - Rs) ** 2) * fc
    Gi[i] = np.sum(gi)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate symmetric function type-4
# 3-bodies
def G4(comm, i, R, cosine, natom, fc, eta, lam, zeta):
    G,Gi = np.empty(natom),np.zeros(natom)
    gauss = np.exp(- eta * (R**2)).reshape((-1,1))
    cutoff = fc.reshape((-1,1))
    gi = ((1+lam*cosine)**zeta) * np.dot(gauss, gauss.T) * np.dot(cutoff, cutoff.T)
    filter = np.identity(len(R), dtype=bool)
    gi[filter] = 0.0
    Gi[i] = (2 ** (1-zeta)) * np.sum
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate interatomic distances with neighbours
def distance_ij(i, atoms):
    R = np.array([con.distance for con in atoms.connect[i+1]])
    return R

# calculate anlges with neighbours
def cosine_ijk(i, atoms):
    n_neighb = atoms.n_neighbours(i+1)
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
