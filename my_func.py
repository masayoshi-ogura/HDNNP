# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import os.path as path

# calculate the symmetry functions and derivatives of it
### input
# comm,size,rank: MPI object
# atoms_objs: list of Atoms Object
# train_npy_dir,name: string
# Rcs,etas,Rss: list of float
# lams,zetas: list of int
# natoms,nsample,ninput: int
### output
# Gs: numpy array (nsample x natom x ninput)
# dGs: numpy array (nsample x natom x 3*natom * ninput)
def load_or_calc_G(comm, size, rank, atoms_objs, train_npy_dir, name, Rcs, etas, Rss, lams, zetas, natom, nsample, ninput):
    dr = 0.01
    quo,rem = nsample/size,nsample%size
    if rank < rem:
        min,max = rank*(quo+1),(rank+1)*(quo+1)
    else:
        min,max = rank*quo+rem,(rank+1)*quo+rem
    Gs_T = np.empty((ninput, natom, nsample))
    dGs_T = np.empty((ninput, 3*natom, natom, nsample))
    n = 0
    for Rc in Rcs:
        # G1
        prefix = train_npy_dir+name+'-G1-'+str(Rc)
        if path.exists(prefix+'-Gs.npy'):
            Gs_T[n] = np.load(prefix+'-Gs.npy').T
            dGs_T[n] = np.load(prefix+'-dGs.npy').T
        else:
            G,dG = calc_G1(comm, atoms_objs, min, max, dr, Rc, natom, nsample) # calc R and fc, and then G
            np.save(prefix+'-Gs.npy', G)
            np.save(prefix+'-dGs.npy', dG)
            Gs_T[n] = G.T; dGs_T[n] = dG.T
        n += 1
        
        for eta in etas:
            # G2
            for Rs in Rss:
                prefix = train_npy_dir+name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs)
                if path.exists(prefix+'-Gs.npy'):
                    Gs_T[n] = np.load(prefix+'-Gs.npy').T
                    dGs_T[n] = np.load(prefix+'-dGs.npy').T
                else:
                    G,dG = calc_G2(comm, atoms_objs, min, max, dr, Rc, eta, Rs, natom, nsample) # calc R and fc, and then G
                    np.save(prefix+'-Gs.npy', G)
                    np.save(prefix+'-dGs.npy', dG)
                    Gs_T[n] = G.T; dGs_T[n] = dG.T
                n += 1
            
            # G4
            for lam in lams:
                for zeta in zetas:
                    prefix = train_npy_dir+name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta)
                    if path.exists(prefix+'-Gs.npy'):
                        Gs_T[n] = np.load(prefix+'-Gs.npy').T
                        dGs_T[n] = np.load(prefix+'-dGs.npy').T
                    else:
                        G,dG = calc_G4(comm, atoms_objs, min, max, dr, Rc, eta, lam, zeta, natom, nsample) # calc R and cosine and fc, and then G
                        np.save(prefix+'-Gs.npy', G)
                        np.save(prefix+'-dGs.npy', dG)
                        Gs_T[n] = G.T; dGs_T[n] = dG.T
                    n += 1
    return Gs_T.T, dGs_T.T

# load the symmetry functions and derivatives of it
### input
# train_npy_dir,name: string
# Rcs,etas,Rss: list of float
# lams,zetas: list of int
### output
# Gs: numpy array (nsample x natom x ninput)
# dGs: numpy array (nsample x natom x 3*natom * ninput)
def load_G(train_npy_dir, name, Rcs, etas, Rss, lams, zetas):
    loaded_G,loaded_dG = [],[]
    for Rc in Rcs:
        prefix = train_npy_dir+name+'-G1-'+str(Rc)
        if path.exists(prefix+'-Gs.npy'):
            loaded_G.append(np.load(prefix+'-Gs.npy').T)
            loaded_dG.append(np.load(prefix+'-dGs.npy').T)
        
        for eta in etas:
            for Rs in Rss:
                prefix = train_npy_dir+name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs)
                if path.exists(prefix+'-Gs.npy'):
                    loaded_G.append(np.load(prefix+'-Gs.npy').T)
                    loaded_dG.append(np.load(prefix+'-dGs.npy').T)
                
            for lam in lams:
                for zeta in zetas:
                    prefix = train_npy_dir+name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta)
                    if path.exists(prefix+'-Gs.npy'):
                        loaded_G.append(np.load(prefix+'-Gs.npy').T)
                        loaded_dG.append(np.load(prefix+'-dGs.npy').T)
    
    G = np.c_[loaded_G].T
    dG = np.c_[loaded_dG].T
    return G,dG

# calculate interatomic distances, cutoff function, and cosine of triplet angle
### input
# atoms: Atoms Object
# m: int
# dr,Rc: float
# natom: int
### output
# R_array,fc_array,cosine_array: list of list of numpy array
def calc_geometry(atoms, m, dr, Rc, natom):
    R_array,fc_array,cosine_array = [],[],[]
    # prepare R and cosine
    atoms.set_cutoff(Rc)
    atoms.calc_connect()
    R,fc = distance_ij(atoms, m, 0, Rc, natom)
    cosine = cosine_ijk(atoms, m, 0, Rc, natom)
    
    # prepare just slightly deviated R and cosine for numerical derivatives
    # assume that neighbour atoms are not changed after displacing since dr is too small
    R_array.append(R); fc_array.append(fc); cosine_array.append(cosine),
    dr = 0.01
    for r in range(3*natom):
        k=r/3; alpha=r%3
        # prepare displaced R and cosine
        atoms_plus = atoms.copy(); atoms_plus.pos[k+1][alpha+1] += dr
        atoms_plus.calc_connect()
        R_plus,fc_plus = distance_ij(atoms_plus, m, 2*r+1, Rc, natom)
        R_array.append(R_plus); fc_array.append(fc_plus)
        cosine_array.append(cosine_ijk(atoms_plus, m, 2*r+1, Rc, natom))
        atoms_minus = atoms.copy(); atoms_minus.pos[k+1][alpha+1] -= dr
        atoms_minus.calc_connect()
        R_minus,fc_minus = distance_ij(atoms_minus, m, 2*r+2, Rc, natom)
        R_array.append(R_minus); fc_array.append(fc_minus)
        cosine_array.append(cosine_ijk(atoms_minus, m, 2*r+2, Rc, natom))
        
    return R_array,fc_array,cosine_array

def calc_G1(comm, atoms_objs, min, max, dr, Rc, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        R,fc,cosine = calc_geometry(atoms, m, dr, Rc, natom)
        G_para[m] = G1(fc[0], natom)
        tmp = np.empty((3*natom,natom))
        for r in range(3*natom):
            G1_plus = G1(fc[2*r+1], natom)
            G1_minus = G1(fc[2*r+2], natom)
            tmp[r] = (G1_plus - G1_minus) / (2 * dr)
        dG_para[m] = tmp.T
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

def calc_G2(comm, atoms_objs, min, max, dr, Rc, eta, Rs, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        R,fc,cosine = calc_geometry(atoms, m, dr, Rc, natom)
        G_para[m] = G2(R[0], fc[0], eta, Rs, natom)
        tmp = np.empty((3*natom,natom))
        for r in range(3*natom):
            G2_plus = G2(R[2*r+1], fc[2*r+1], eta, Rs, natom)
            G2_minus = G2(R[2*r+2], fc[2*r+2], eta, Rs, natom)
            tmp[r] = (G2_plus - G2_minus) / (2 * dr)
        dG_para[m] = tmp.T
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

def calc_G4(comm, atoms_objs, min, max, dr, Rc, eta, lam, zeta, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        R,fc,cosine = calc_geometry(atoms, m, dr, Rc, natom)
        G_para[m] = G4(R[0], fc[0], cosine[0], eta, lam, zeta, natom)
        tmp = np.empty((3*natom,natom))
        for r in range(3*natom):
            G4_plus = G4(R[2*r+1], fc[2*r+1], cosine[2*r+1], eta, lam, zeta, natom)
            G4_minus = G4(R[2*r+2], fc[2*r+2], cosine[2*r+2], eta, lam, zeta, natom)
            tmp[r] = (G4_plus - G4_minus) / (2 * dr)
        dG_para[m] = tmp.T
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

# calculate symmetric function type-1
# 2-bodies
def G1(fc, natom):
    G = np.empty(natom)
    for i in range(natom):
        G[i] = np.sum(fc[i])
    return G

# calculate symmetric function type-2
# 2-bodies
def G2(R, fc, eta, Rs, natom):
    G = np.empty(natom)
    for i in range(natom):
        gi = np.exp(- eta * (R[i] - Rs) ** 2) * fc[i]
        G[i] = np.sum(gi)
    return G

# calculate symmetric function type-4
# 3-bodies
def G4(R, fc, cosine, eta, lam, zeta, natom):
    G = np.empty(natom)
    for i in range(natom):
        gauss = np.exp(- eta * (R[i]**2)).reshape((-1,1))
        cutoff = fc[i].reshape((-1,1))
        gi = ((1+lam*cosine[i])**zeta) * np.dot(gauss, gauss.T) * np.dot(cutoff, cutoff.T)
        filter = np.identity(len(R), dtype=bool)
        gi[filter] = 0.0
        G[i] = (2 ** (1-zeta)) * np.sum(gi)
    return G

# memorize variables
def memorize(f):
    if f.func_name == 'distance_ij':
        cache = {}
        def helper(atoms, m, r, Rc, natom):
            if (m,r,Rc) not in cache:
                cache[(m,r,Rc)] = f(atoms, m, r, Rc, natom)
            return cache[(m,r,Rc)]
        return helper
    elif f.func_name == 'cosine_ijk':
        cache = {}
        def helper(atoms, m, r, Rc, natom):
            if (m,r,Rc) not in cache:
                cache[(m,r,Rc)] = f(atoms, m, r, Rc, natom)
            return cache[(m,r,Rc)]
        return helper

# calculate interatomic distances with neighbours
### input
# atoms: Atoms Object
# m,r: int
#      used for indexing cache with nsample index and displacement index
#      m ... different for each nodes
#      r ... 0..2*3*natom
# Rc: float
#     used for the same reason
# natom: int
### output
# R,fc: list of numpy array
@memorize
def distance_ij(atoms, m, r, Rc, natom):
    R,fc = [],[]
    for i in range(natom):
        R.append(np.array([con.distance for con in atoms.connect[i+1]]))
        fc.append(np.tanh(1-R[i]/Rc)**3)
    return R,fc

# calculate anlges with neighbours
### input
# atoms: Atoms Object
# m,r: int
#      used for indexing cache with nsample index and displacement index
#      m ... different for each nodes
#      r ... 0..2*3*natom
# Rc: float
#     used for the same reason
# natom: int
### output
# R,fc: list of numpy array
@memorize
def cosine_ijk(atoms, m, r, Rc, natom):
    cosine_ret = []
    for i in range(natom):
        n_neighb = atoms.n_neighbours(i+1)
        cosine = np.zeros((n_neighb,n_neighb))
        for j in range(n_neighb):
            for k in range(n_neighb):
                if k == j:
                    pass
                else:
                    cosine[j][k] = atoms.cosine_neighbour(i+1,j+1,k+1)
        cosine_ret.append(cosine)
    return cosine_ret
