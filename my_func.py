# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import math

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
        # prepare R and angle
        extend = atoms_objs[m].repeat(2)
        pos = extend.get_positions()
        R = distance_ij(rank, extend, natom)
        angle = angle_ijk(rank, extend, natom)
        # prepare just slightly deviated R and angle for numerical derivatives
        R_plus,R_minus,angle_plus,angle_minus = [],[],[],[]
        dr = 0.001
        for r in range(3*natom):
            k,alpha = r/3,r%3
            # copy extend and pos to prevent them from destroying
            extend_plus = extend.copy(); newpos = pos.copy()
            newpos[k][alpha] += dr
            extend_plus.set_positions(newpos) # displase k-th atom along alpha-axis
            R_plus.append(distance_ij(rank, extend_plus, natom))
            angle_plus.append(angle_ijk(rank, extend_plus, natom))
            # copy extend and pos to prevent them from destroying
            extend_minus = extend.copy(); newpos = pos.copy()
            newpos[k][alpha] -= dr
            extend_minus.set_positions(newpos) # displase k-th atom along alpha-axis
            R_minus.append(distance_ij(rank, extend_minus, natom))
            angle_minus.append(angle_ijk(rank, extend_minus, natom))
        
        G = np.empty((ninput, natom))
        dG = np.empty((ninput, 3*natom, natom))
        n = 0
        for Rc in Rcs:
            # prepare cutoff functions with Rc
            fc = cutoff_func(R, Rc)
            fc_plus,fc_minus = [],[]
            for r in range(3*natom):
                fc_plus.append(cutoff_func(R_plus[r], Rc))
                fc_minus.append(cutoff_func(R_minus[r], Rc))
            
            # G1
            G[n] = G1(comm, rank, natom, fc)
            for r in range(3*natom):
                G1_plus = G1(comm, rank, natom, fc_plus[r])
                G1_minus = G1(comm, rank, natom, fc_minus[r])
                dG[n][r] = (G1_plus - G1_minus) / (2 * dr)
            n += 1
            
            for eta in etas:
                # G2
                for Rs in Rss:
                    G[n] = G2(comm, rank, R, natom, fc, Rs, eta)
                    for r in range(3*natom):
                        G2_plus = G2(comm, rank, R_plus[r], natom, fc_plus[r], Rs, eta)
                        G2_minus = G2(comm, rank, R_minus[r], natom, fc_minus[r], Rs, eta)
                        dG[n][r] = (G2_plus - G2_minus) / (2 * dr)
                    n += 1
                
                # G4
                for lam in lams:
                    for zeta in zetas:
                        G[n] = G4(comm, rank, R, angle, natom, fc, eta, lam, zeta)
                        for r in range(3*natom):
                            G4_plus = G4(comm, rank, R_plus[r], angle_plus[r], natom, fc, eta, lam, zeta)
                            G4_minus = G4(comm, rank, R_minus[r], angle_minus[r], natom, fc, eta, lam, zeta)
                            dG[n][r] = (G4_plus - G4_minus) / (2 * dr)
                        n += 1
        Gs[m] = G.T
        dGs[m] = dG.T
    return Gs, dGs

# calculate symmetric function type-1
# 2-bodies
def G1(comm, i, natom, fc):
    G,Gi = np.empty(natom),np.zeros(natom)
    filter = np.array([j == i for j in range(8*natom)])
    fc[filter] = 0.0
    Gi[i] = np.sum(fc)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate symmetric function type-2
# 2-bodies
def G2(comm, i, R, natom, fc, Rs, eta):
    G,Gi = np.empty(natom),np.empty(natom)
    gi = np.exp(- eta * (R - Rs) ** 2) * fc
    filter = np.array([j == i for j in range(8*natom)])
    gi[filter] = 0.0
    Gi[i] = np.sum(gi)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate symmetric function type-4
# 3-bodies
def G4(comm, i, R, angle, natom, fc, eta, lam, zeta):
    G,Gi = np.empty(natom),np.empty(natom)
    R_res = np.exp(- eta * (R**2)).reshape((-1,1))
    fc_res = fc.reshape((-1,1))
    gi = ((1+lam * np.cos(angle*math.pi))**zeta) * np.dot(R_res, R_res.T) * np.dot(fc_res, fc_res.T)
    filter = np.array([j == i or k == i or k == j for j in range(8*natom) for k in range(8*natom)]).reshape((8*natom,8*natom))
    gi[filter] = 0.0
    Gi[i] = (2 ** (1-zeta)) * np.sum(gi)
    comm.Allreduce(Gi, G, op=MPI.SUM)
    return G

# calculate interatomic distances
def distance_ij(i, atoms_obj, natom):
    R = atoms_obj.get_distances(i, range(8*natom), mic=True)
    return R

# calculate anlges
def angle_ijk(i, atoms_obj, natom):
    angle = np.zeros((8*natom,8*natom))
    for j in range(8*natom):
        for k in range(8*natom):
            if j == i or k == i or k == j:
                pass
            else:
                angle[j][k] = atoms_obj.get_angle(j,i,k)
    return angle

# calculate cutoff function
def cutoff_func(R, Rc):
    filter = R > Rc
    fc = np.tanh(1-R/Rc)**3
    fc[filter] = 0.0
    return fc
