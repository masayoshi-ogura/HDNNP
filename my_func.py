# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import time

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
    if rank == 0:
        st = time.time()
    Gs = np.empty((nsample, natom, ninput))
    dGs = np.empty((nsample, natom, 3*natom, ninput))
    for m in range(nsample):
        if rank == 0:
            print 'start m:'+str(m)
            print time.time()-st
        # prepare R and cosine
        extend = atoms_objs[m].repeat(2)
        R = distance_ij(rank, extend, natom)
        cosine = cosine_ijk(rank, extend, natom)
        if rank == 0:
            print 'checkpoint 1:'
            print time.time()-st
        # prepare just slightly deviated R and cosine for numerical derivatives
        R_array,cosine_array = np.empty((1+2*3*natom,8*natom)),np.empty((1+2*3*natom,8*natom,8*natom))
        R_array[0] = R; cosine_array[0] = cosine
        dr = 0.001
        for r in range(3*natom):
            disp_plus = np.array([dr if i==r else 0 for i in range(8*3*natom)]).reshape((8*natom,3))
            disp_minus = np.array([-dr if i==r else 0 for i in range(8*3*natom)]).reshape((8*natom,3))
            # prepare displaced R and cosine
            extend_plus = extend.copy(); extend_plus.translate(disp_plus)
            R_array[2*r+1] = (distance_ij(rank, extend_plus, natom))
            cosine_array[2*r+1] = (cosine_ijk(rank, extend_plus, natom))
            extend_minus = extend.copy(); extend_minus.translate(disp_minus)
            R_array[2*r+2] = (distance_ij(rank, extend_minus, natom))
            cosine_array[2*r+2] = (cosine_ijk(rank, extend_minus, natom))
            if rank == 0:
                print 'checkpoint 2, r:'+str(r)
                print time.time()-st
        
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
                        G4_array = G4(comm, rank, R_array, cosine_array, natom, fc_array, eta, lam, zeta)
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
    filter = np.array([[j == i for j in range(8*natom)]]).repeat(1+2*3*natom, axis=0)
    fc[filter] = 0.0
    Gi[i] = np.sum(fc, axis=1)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate symmetric function type-2
# 2-bodies
def G2(comm, i, R, natom, fc, Rs, eta):
    G,Gi = np.empty((natom,1+2*3*natom)),np.zeros((natom,1+2*3*natom))
    gi = np.exp(- eta * (R - Rs) ** 2) * fc
    filter = np.array([[j == i for j in range(8*natom)]]).repeat(1+2*3*natom, axis=0)
    gi[filter] = 0.0
    Gi[i] = np.sum(gi, axis=1)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate symmetric function type-4
# 3-bodies
def G4(comm, i, R, cosine, natom, fc, eta, lam, zeta):
    G,Gi = np.empty((natom,1+2*3*natom)),np.zeros((natom,1+2*3*natom))
    R_ij = np.exp(- eta * (R**2))[:,:,None].repeat(8*natom,axis=2)
    R_ik = np.exp(- eta * (R**2))[:,None,:].repeat(8*natom,axis=1)
    fc_ij = fc[:,:,None].repeat(8*natom,axis=2)
    fc_ik = fc[:,None,:].repeat(8*natom,axis=1)
    gi = ((1+lam*cosine)**zeta) * R_ij * R_ik * fc_ij * fc_ik
    filter = np.array([j == i or k == i or k == j for j in range(8*natom) for k in range(8*natom)]).repeat(1+2*3*natom).reshape((1+2*3*natom,8*natom,8*natom))
    gi[filter] = 0.0
    Gi[i] = (2 ** (1-zeta)) * np.sum(gi, axis=(1,2))
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G.T

# calculate interatomic distances
def distance_ij(i, atoms_obj, natom):
    # R = atoms_obj.get_all_distances(mic=True)[i]
    R = atoms_obj.get_distances(i, range(8*natom), mic=True)
    return R

# calculate anlges
def cosine_ijk(i, atoms_obj, natom):
    cosine = np.zeros((8*natom,8*natom))
    for j in range(8*natom):
        for k in range(8*natom):
            if j == i or k == i or k == j:
                pass
            else:
                cosine[j][k] = atoms_obj.cosine(j,i,k)
    return cosine

# calculate cutoff function
def cutoff_func(R, Rc):
    filter = R > Rc
    fc = np.tanh(1-R/Rc)**3
    fc[filter] = 0.0
    return fc
