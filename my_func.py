# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import os.path as path
from quippy import farray,fzeros,frange

def calc_EF(atoms_objs, train_npy_dir, name, natom, nsample):
    Es = np.array([data.cohesive_energy for data in atoms_objs]) 
    Fs = np.array([np.array(data.force).T for data in atoms_objs]).reshape((nsample,3*natom))
    np.save(train_npy_dir+name+'-Es.npy', Es)
    np.save(train_npy_dir+name+'-Fs.npy', Fs)
    return Es,Fs

def load_EF(train_npy_dir, name):
    Es = np.load(train_npy_dir+name+'-Es.npy')
    Fs = np.load(train_npy_dir+name+'-Fs.npy')
    return Es,Fs

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
        if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
            Gs_T[n] = np.load(prefix+'-Gs.npy').T
            dGs_T[n] = np.load(prefix+'-dGs.npy').T
        else:
            G,dG = calc_G1(comm, atoms_objs, min, max, Rc, natom, nsample)
            np.save(prefix+'-Gs.npy', G)
            np.save(prefix+'-dGs.npy', dG)
            Gs_T[n] = G.T; dGs_T[n] = dG.T
        n += 1
        
        for eta in etas:
            # G2
            for Rs in Rss:
                prefix = train_npy_dir+name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs)
                if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
                    Gs_T[n] = np.load(prefix+'-Gs.npy').T
                    dGs_T[n] = np.load(prefix+'-dGs.npy').T
                else:
                    G,dG = calc_G2(comm, atoms_objs, min, max, Rc, eta, Rs, natom, nsample)
                    np.save(prefix+'-Gs.npy', G)
                    np.save(prefix+'-dGs.npy', dG)
                    Gs_T[n] = G.T; dGs_T[n] = dG.T
                n += 1
            
            # G4
            for lam in lams:
                for zeta in zetas:
                    prefix = train_npy_dir+name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta)
                    if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
                        Gs_T[n] = np.load(prefix+'-Gs.npy').T
                        dGs_T[n] = np.load(prefix+'-dGs.npy').T
                    else:
                        G,dG = calc_G4(comm, atoms_objs, min, max, Rc, eta, lam, zeta, natom, nsample)
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

# calculate G and dG for all structure
def calc_G1(comm, atoms_objs, min, max, Rc, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        index,r,R,fc,tanh,cosine = calc_geometry(atoms, m, Rc, natom)
        dR = deriv_R(m, Rc, index, r, R, natom)
        G_para[m],dG_para[m] = G1(fc, tanh, dR, Rc, natom)
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

def calc_G2(comm, atoms_objs, min, max, Rc, eta, Rs, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        index,r,R,fc,tanh,cosine = calc_geometry(atoms, m, Rc, natom)
        dR = deriv_R(m, Rc, index, r, R, natom)
        G_para[m],dG_para[m] = G2(R, fc, tanh, dR, Rc, eta, Rs, natom)
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

def calc_G4(comm, atoms_objs, min, max, Rc, eta, lam, zeta, natom, nsample):
    G,dG = np.empty((nsample,natom)),np.empty((nsample,natom,3*natom))
    G_para,dG_para = np.zeros((nsample,natom)),np.zeros((nsample,natom,3*natom))
    for m in range(min,max):
        atoms = atoms_objs[m]
        index,r,R,fc,tanh,cosine = calc_geometry(atoms, m, Rc, natom)
        dR = deriv_R(m, Rc, index, r, R, natom)
        dcos = deriv_cosine(m, Rc, index, r, R, cosine, natom)
        G_para[m],dG_para[m] = G4(R, fc, tanh, cosine, dR, dcos, Rc, eta, lam, zeta, natom)
    comm.Allreduce(G_para, G, op=MPI.SUM)
    comm.Allreduce(dG_para, dG, op=MPI.SUM)
    return G,dG

# calculate G and dG for each one structure
# calculate symmetric function type-1
# 2-bodies
def G1(fc, tanh, dR, Rc, natom):
    G = np.empty(natom)
    dG = np.empty((natom,3*natom))
    for i in range(natom):
        G[i] = np.sum(fc[i])
        
        dgi = - 3/Rc * (1 - tanh[i][:,None]**2) * tanh[i][:,None]**2 * dR[i]
        dG[i] = np.sum(dgi, axis=0)
    return G,dG

# calculate symmetric function type-2
# 2-bodies
def G2(R, fc, tanh, dR, Rc, eta, Rs, natom):
    G = np.empty(natom)
    dG = np.empty((natom,3*natom))
    for i in range(natom):
        gi = np.exp(- eta * (R[i] - Rs) ** 2) * fc[i]
        G[i] = np.sum(gi)
        
        dgi = gi[:,None] * ((-2*Rc*eta*(R[i][:,None]-Rs)*tanh[i][:,None] + 3*tanh[i][:,None]**2 - 3) / (Rc * tanh[i][:,None])) * dR[i]
        dG[i] = np.sum(dgi, axis=0)
    return G,dG

# calculate symmetric function type-4
# 3-bodies
def G4(R, fc, tanh, cosine, dR, dcos, Rc, eta, lam, zeta, natom):
    G = np.empty(natom)
    dG = np.empty((natom,3*natom))
    for i in range(natom):
        angular = (1+lam*cosine[i])
        gi_ = (2**(1-zeta)) * (angular**(zeta-1)) * np.exp(-eta*(R[i][:,None]**2+R[i][None,:]**2)) * fc[i][:,None] * fc[i][None,:]
        gi = gi_ * angular # separate calculation in order to prevent zero division
        filter = np.identity(len(R[i]), dtype=bool)
        gi[filter] = 0.0
        G[i]  = np.sum(gi)
        
        dgi_R   = ((-2*Rc*eta*R[i][:,None]*tanh[i][:,None] + 3*tanh[i][:,None]**2 - 3) / (Rc * tanh[i][:,None])) * dR[i]
        dgi_cos = zeta * lam * dcos[i]
        dgi = gi_[:,:,None] * (angular[:,:,None] * (dgi_R[None,:,:] + dgi_R[:,None,:]) + dgi_cos)
        dG[i] = np.sum(dgi, axis=(0,1))
    return G,dG

# memorize variables
def memorize(f):
    if f.__name__ == 'calc_geometry':
        cache = {}
        def helper(atoms, m, Rc, natom):
            if (m,Rc) not in cache:
                cache[(m,Rc)] = f(atoms, m, Rc, natom)
            return cache[(m,Rc)]
        return helper
    elif f.__name__ == 'deriv_R':
        cache = {}
        def helper(m, Rc, index, r, R, natom):
            if (m,Rc) not in cache:
                cache[(m,Rc)] = f(m, Rc, index, r, R, natom)
            return cache[(m,Rc)]
        return helper
    elif f.__name__ == 'deriv_cosine':
        cache = {}
        def helper(m, Rc, index, r, R, cosine, natom):
            if (m,Rc) not in cache:
                cache[(m,Rc)] = f(m, Rc, index, r, R, cosine, natom)
            return cache[(m,Rc)]
        return helper

# get various neighbours data(index, vector, distance), and calculate cutoff function, tanh, and cosine
### input
# atoms: Atoms Object
# m:  int
# Rc: float
#     m,Rc is used for the key of memorization
# natom: int
### output
# index,r,R,fc,tanh,cosine: list of numpy array
#                           can't use numpy array because the number of the neighbours of each atoms is different
@memorize
def calc_geometry(atoms, m, Rc, natom):
    atoms.set_cutoff(Rc)
    atoms.calc_connect()
    index = [ atoms.connect.get_neighbours(i)[0] - 1 for i in frange(natom) ]
    r,R,fc,tanh = distance_ij(atoms, Rc, natom)
    cosine = cosine_ijk(atoms, Rc, natom)
    
    return index,r,R,fc,tanh,cosine

### input
# atoms: Atoms Object
# Rc: float
# natom: int
### output
# r,R,fc,tanh: list of numpy array
def distance_ij(atoms, Rc, natom):
    r,R,fc,tanh = [],[],[],[]
    for i in frange(natom):
        ri,Ri = [],[]
        for n in frange(atoms.n_neighbours(i)):
            dist = farray(0.0)
            diff = fzeros(3)
            atoms.neighbour(i,n,distance=dist,diff=diff)
            ri.append(diff)
            Ri.append(dist)
        r.append(np.array(ri))
        R.append(np.array(Ri))
        fc.append(np.tanh(1-R[i-1]/Rc)**3)
        tanh.append(np.tanh(1-R[i-1]/Rc))
    return r,R,fc,tanh

# atoms: Atoms Object
# Rc: float
# natom: int
### output
# cosine: list of numpy array
def cosine_ijk(atoms, Rc, natom):
    cosine = []
    for i in frange(natom):
        n_neighb = atoms.n_neighbours(i)
        cosi = np.zeros((n_neighb,n_neighb))
        for j in frange(n_neighb):
            for k in frange(n_neighb):
                if k == j:
                    pass
                else:
                    cosi[j-1][k-1] = atoms.cosine_neighbour(i,j,k)
        cosine.append(cosi)
    return cosine

# calculate derivative of atomic distance between focused atom i and its neighbour atom j
### input
# m: int
# Rc: float
#     m,Rc is used for the key of memorization
# index: list of numpy array(n_neighb)
# r:     list of numpy array(n_neighb,3)
# R:     list of numpy array(n_neighb)
# natom: int
### output
# dR:    list of numpy array(n_neighb,3*natom)
@memorize
def deriv_R(m, Rc, index, r, R, natom):
    dR = []
    for i in range(natom):
        n_neighb = len(R[i])
        dRi = np.zeros((n_neighb,3*natom))
        for j in index[i]:
            for l in range(natom):
                if l == i:
                    for alpha in range(3):
                        dRi[j][3*l+alpha] = - r[i][j][alpha] / R[i][j]
                elif l == j:
                    for alpha in range(3):
                        dRi[j][3*l+alpha] = + r[i][j][alpha] / R[i][j]
        dR.append(dRi)
    return dR

# calculate derivative of cosine of triplet between focused atom i and its neighbour atom j,k
### input
# m: int
# Rc: float
#     m,Rc is used for the key of memorization
# index:  list of numpy array(n_neighb)
# r:      list of numpy array(n_neighb,3)
# R:      list of numpy array(n_neighb)
# cosine: list of numpy array(n_neighb,n_neighb)
# natom: int
### output
# dcos:   list of numpy array(n_neighb,n_neighb,3*natom)
@memorize
def deriv_cosine(m, Rc, index, r, R, cosine, natom):
    dcos = []
    for i in range(natom):
        n_neighb = len(R[i])
        dcosi = np.zeros((n_neighb,n_neighb,3*natom))
        for j in index[i]:
            for k in index[i]:
                for l in range(natom):
                    if l == i:
                        for alpha in range(3):
                            dcosi[j][k][3*l+alpha] = (+ r[i][j][alpha] / R[i][j]**2) * cosine[i][j][k] + \
                                                     (+ r[i][k][alpha] / R[i][k]**2) * cosine[i][j][k] + \
                                                     (- (r[i][j][alpha] + r[i][k][alpha])) / (R[i][j] * R[i][k])
                    elif l == j :
                        for alpha in range(3):
                            dcosi[j][k][3*l+alpha] = (- r[i][j][alpha] / R[i][j]**2) * cosine[i][j][k] + \
                                                     (+ r[i][k][alpha]) / (R[i][j] * R[i][k])
                    elif l == k :
                        for alpha in range(3):
                            dcosi[j][k][3*l+alpha] = (- r[i][k][alpha] / R[i][k]**2) * cosine[i][j][k] + \
                                                     (+ r[i][j][alpha]) / (R[i][j] * R[i][k])
        dcos.append(dcosi)
    return dcos
