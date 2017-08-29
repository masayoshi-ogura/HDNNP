# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
from os import path
from quippy import farray,fzeros,frange

class Generator:
    """Energy, Force, and Symmetric Fuction generator"""
    def __init__(self, train_npy_dir, name, Rcs, etas, Rss, lams, zetas):
        # set instance variables
        self.train_npy_dir = train_npy_dir
        self.name  = name
        self.Rcs   = Rcs
        self.etas  = etas
        self.Rss   = Rss
        self.lams  = lams
        self.zetas = zetas
    
    def calc_EF(self, atoms_objs, natom, nsample):
        Es = np.array([data.cohesive_energy for data in atoms_objs]) 
        Fs = np.array([np.array(data.force).T for data in atoms_objs]).reshape((nsample,3*natom))
        np.save(path.join(self.train_npy_dir, self.name+'-Es.npy'), Es)
        np.save(path.join(self.train_npy_dir, self.name+'-Fs.npy'), Fs)
        return Es,Fs

    def load_EF(self):
        Es = np.load(path.join(self.train_npy_dir, self.name+'-Es.npy'))
        Fs = np.load(path.join(self.train_npy_dir, self.name+'-Fs.npy'))
        return Es,Fs

    def calc_G(self, comm, size, rank, atoms_objs, natom, nsample, ninput):
        # set instance variables
        self.comm = comm
        self.natom = natom
        self.nsample = nsample
        quo,rem = self.nsample/size,self.nsample%size
        if rank < rem:
            self.atoms_objs = atoms_objs[rank*(quo+1):(rank+1)*(quo+1)]
        else:
            self.atoms_objs = atoms_objs[rank*quo+rem:(rank+1)*quo+rem]
        
        Gs_T = np.empty((ninput, self.natom, self.nsample))
        dGs_T = np.empty((ninput, 3*natom, self.natom, self.nsample))
        n = 0
        for Rc in self.Rcs:
            # G1
            prefix = path.join(self.train_npy_dir, self.name+'-G1-'+str(Rc))
            if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
                Gs_T[n] = np.load(prefix+'-Gs.npy').T
                dGs_T[n] = np.load(prefix+'-dGs.npy').T
            else:
                G,dG = self.calc_G1(Rc)
                np.save(prefix+'-Gs.npy', G)
                np.save(prefix+'-dGs.npy', dG)
                Gs_T[n] = G.T; dGs_T[n] = dG.T
            n += 1
            
            for eta in self.etas:
                # G2
                for Rs in self.Rss:
                    prefix = path.join(self.train_npy_dir, self.name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs))
                    if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
                        Gs_T[n] = np.load(prefix+'-Gs.npy').T
                        dGs_T[n] = np.load(prefix+'-dGs.npy').T
                    else:
                        G,dG = self.calc_G2(Rc, eta, Rs)
                        np.save(prefix+'-Gs.npy', G)
                        np.save(prefix+'-dGs.npy', dG)
                        Gs_T[n] = G.T; dGs_T[n] = dG.T
                    n += 1
                
                # G4
                for lam in self.lams:
                    for zeta in self.zetas:
                        prefix = path.join(self.train_npy_dir, self.name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta))
                        if path.exists(prefix+'-Gs.npy') and Gs_T[n].shape == np.load(prefix+'-Gs.npy').T.shape:
                            Gs_T[n] = np.load(prefix+'-Gs.npy').T
                            dGs_T[n] = np.load(prefix+'-dGs.npy').T
                        else:
                            G,dG = self.calc_G4(Rc, eta, lam, zeta)
                            np.save(prefix+'-Gs.npy', G)
                            np.save(prefix+'-dGs.npy', dG)
                            Gs_T[n] = G.T; dGs_T[n] = dG.T
                        n += 1
        return Gs_T.T, dGs_T.T

    def load_G(self):
        loaded_G,loaded_dG = [],[]
        for Rc in self.Rcs:
            prefix = path.join(self.train_npy_dir, self.name+'-G1-'+str(Rc))
            if path.exists(prefix+'-Gs.npy'):
                loaded_G.append(np.load(prefix+'-Gs.npy').T)
                loaded_dG.append(np.load(prefix+'-dGs.npy').T)
            
            for eta in self.etas:
                for Rs in self.Rss:
                    prefix = path.join(self.train_npy_dir, self.name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs))
                    if path.exists(prefix+'-Gs.npy'):
                        loaded_G.append(np.load(prefix+'-Gs.npy').T)
                        loaded_dG.append(np.load(prefix+'-dGs.npy').T)
                    
                for lam in self.lams:
                    for zeta in self.zetas:
                        prefix = path.join(self.train_npy_dir, self.name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta))
                        if path.exists(prefix+'-Gs.npy'):
                            loaded_G.append(np.load(prefix+'-Gs.npy').T)
                            loaded_dG.append(np.load(prefix+'-dGs.npy').T)
        
        G = np.c_[loaded_G].T
        dG = np.c_[loaded_dG].T
        return G,dG

    def calc_G1(self, Rc):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m,atoms in enumerate(self.atoms_objs):
            index,r,R,fc,tanh,cosine = self.calc_geometry(m, atoms, Rc)
            dR = self.deriv_R(m, index, r, R, Rc)
            G_para[m],dG_para[m] = self.G1(fc, tanh, dR, Rc)
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    def calc_G2(self, Rc, eta, Rs):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m,atoms in enumerate(self.atoms_objs):
            index,r,R,fc,tanh,cosine = self.calc_geometry(m, atoms, Rc)
            dR = self.deriv_R(m, index, r, R, Rc)
            G_para[m],dG_para[m] = self.G2(R, fc, tanh, dR, Rc, eta, Rs)
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    def calc_G4(self, Rc, eta, lam, zeta):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m,atoms in enumerate(self.atoms_objs):
            index,r,R,fc,tanh,cosine = self.calc_geometry(m, atoms, Rc)
            dR = self.deriv_R(m, index, r, R, Rc)
            dcos = self.deriv_cosine(m, index, r, R, cosine, Rc)
            G_para[m],dG_para[m] = self.G4(R, fc, tanh, cosine, dR, dcos, Rc, eta, lam, zeta)
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    def G1(self, fc, tanh, dR, Rc):
        G = np.empty(self.natom)
        dG = np.empty((self.natom,3*self.natom))
        for i in range(self.natom):
            G[i] = np.sum(fc[i])
            
            dgi = - 3/Rc * (1 - tanh[i][:,None]**2) * tanh[i][:,None]**2 * dR[i]
            dG[i] = np.sum(dgi, axis=0)
        return G,dG

    def G2(self, R, fc, tanh, dR, Rc, eta, Rs):
        G = np.empty(self.natom)
        dG = np.empty((self.natom,3*self.natom))
        for i in range(self.natom):
            gi = np.exp(- eta * (R[i] - Rs) ** 2) * fc[i]
            G[i] = np.sum(gi)
            
            dgi = gi[:,None] * ((-2*Rc*eta*(R[i][:,None]-Rs)*tanh[i][:,None] + 3*tanh[i][:,None]**2 - 3) / (Rc * tanh[i][:,None])) * dR[i]
            dG[i] = np.sum(dgi, axis=0)
        return G,dG

# calculate symmetric function type-4
# 3-bodies
    def G4(self, R, fc, tanh, cosine, dR, dcos, Rc, eta, lam, zeta):
        G = np.empty(self.natom)
        dG = np.empty((self.natom,3*self.natom))
        for i in range(self.natom):
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

    def memorize(f):
        if f.__name__ == 'calc_geometry':
            cache = {}
            def helper(self, m, atoms, Rc):
                if (m,Rc) not in cache:
                    cache[(m,Rc)] = f(self, m, atoms, Rc)
                return cache[(m,Rc)]
            return helper
        elif f.__name__ == 'deriv_R':
            cache = {}
            def helper(self, m, index, r, R, Rc):
                if (m,Rc) not in cache:
                    cache[(m,Rc)] = f(self, m, index, r, R, Rc)
                return cache[(m,Rc)]
            return helper
        elif f.__name__ == 'deriv_cosine':
            cache = {}
            def helper(self, m, index, r, R, cosine, Rc):
                if (m,Rc) not in cache:
                    cache[(m,Rc)] = f(self, m, index, r, R, cosine, Rc)
                return cache[(m,Rc)]
            return helper

    @memorize
    def calc_geometry(self, m, atoms, Rc):
        atoms.set_cutoff(Rc)
        atoms.calc_connect()
        index = [ atoms.connect.get_neighbours(i)[0] - 1 for i in frange(self.natom) ]
        r,R,fc,tanh = self.distance_ij(atoms, Rc)
        cosine = self.cosine_ijk(atoms, Rc)
        
        return index,r,R,fc,tanh,cosine

    def distance_ij(self, atoms, Rc):
        r,R,fc,tanh = [],[],[],[]
        for i in frange(self.natom):
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

    def cosine_ijk(self, atoms, Rc):
        cosine = []
        for i in frange(self.natom):
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

    @memorize
    def deriv_R(self, m, index, r, R, Rc):
        dR = []
        for i in range(self.natom):
            n_neighb = len(R[i])
            dRi = np.zeros((n_neighb,3*self.natom))
            for j in index[i]:
                for l in range(self.natom):
                    if l == i:
                        for alpha in range(3):
                            dRi[j][3*l+alpha] = - r[i][j][alpha] / R[i][j]
                    elif l == j:
                        for alpha in range(3):
                            dRi[j][3*l+alpha] = + r[i][j][alpha] / R[i][j]
            dR.append(dRi)
        return dR

    @memorize
    def deriv_cosine(self, m, index, r, R, cosine, Rc):
        dcos = []
        for i in range(self.natom):
            n_neighb = len(R[i])
            dcosi = np.zeros((n_neighb,n_neighb,3*self.natom))
            for j in index[i]:
                for k in index[i]:
                    for l in range(self.natom):
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
