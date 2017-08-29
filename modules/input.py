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
        self.atoms_objs = atoms_objs
        self.natom = natom
        self.nsample = nsample
        quo,rem = self.nsample/size,self.nsample%size
        if rank < rem:
            self.min,self.max = rank*(quo+1),(rank+1)*(quo+1)
        else:
            self.min,self.max = rank*quo+rem,(rank+1)*quo+rem
        
        Gs  = np.empty((ninput,self.nsample,self.natom))
        dGs = np.empty((ninput,self.nsample,self.natom,3*self.natom))

        n = 0
        for Rc in self.Rcs:
            prefix = path.join(self.train_npy_dir, self.name+'-G1-'+str(Rc))
            if path.exists(prefix+'-Gs.npy') and Gs[n].shape == np.load(prefix+'-Gs.npy').shape:
                Gs[n] = np.load(prefix+'-Gs.npy')
                dGs[n] = np.load(prefix+'-dGs.npy')
            else:
                G,dG  = self.calc_G1(Rc)
                np.save(prefix+'-Gs.npy', G)
                np.save(prefix+'-dGs.npy', dG)
                Gs[n]  = G
                dGs[n] = dG
            n += 1

            for eta in self.etas:
                for Rs in self.Rss:
                    prefix = path.join(self.train_npy_dir, self.name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs))
                    if path.exists(prefix+'-Gs.npy') and Gs[n].shape == np.load(prefix+'-Gs.npy').shape:
                        Gs[n]  = np.load(prefix+'-Gs.npy')
                        dGs[n] = np.load(prefix+'-dGs.npy')
                    else:
                        G,dG  = self.calc_G2(Rc, eta, Rs)
                        np.save(prefix+'-Gs.npy', G)
                        np.save(prefix+'-dGs.npy', dG)
                        Gs[n]  = G
                        dGs[n] = dG
                    n += 1

                for lam in self.lams:
                    for zeta in self.zetas:
                        prefix = path.join(self.train_npy_dir, self.name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta))
                        if path.exists(prefix+'-Gs.npy') and Gs[n].shape == np.load(prefix+'-Gs.npy').shape:
                            Gs[n]  = np.load(prefix+'-Gs.npy')
                            dGs[n] = np.load(prefix+'-dGs.npy')
                        else:
                            G,dG  = self.calc_G4(Rc, eta, lam, zeta)
                            np.save(prefix+'-Gs.npy', G)
                            np.save(prefix+'-dGs.npy', dG)
                            Gs[n]  = G
                            dGs[n] = dG
                        n += 1
        return Gs.transpose(1,2,0),dGs.transpose(1,2,3,0)

    def load_G(self):
        loaded_G,loaded_dG = [],[]
        for Rc in self.Rcs:
            prefix = path.join(self.train_npy_dir, self.name+'-G1-'+str(Rc))
            if path.exists(prefix+'-Gs.npy'):
                loaded_G.append(np.load(prefix+'-Gs.npy'))
                loaded_dG.append(np.load(prefix+'-dGs.npy'))
            
            for eta in self.etas:
                for Rs in self.Rss:
                    prefix = path.join(self.train_npy_dir, self.name+'-G2-'+str(Rc)+'-'+str(eta)+'-'+str(Rs))
                    if path.exists(prefix+'-Gs.npy'):
                        loaded_G.append(np.load(prefix+'-Gs.npy'))
                        loaded_dG.append(np.load(prefix+'-dGs.npy'))
                    
                for lam in self.lams:
                    for zeta in self.zetas:
                        prefix = path.join(self.train_npy_dir, self.name+'-G4-'+str(Rc)+'-'+str(eta)+'-'+str(lam)+'-'+str(zeta))
                        if path.exists(prefix+'-Gs.npy'):
                            loaded_G.append(np.load(prefix+'-Gs.npy'))
                            loaded_dG.append(np.load(prefix+'-dGs.npy'))
        
        Gs  = np.c_[loaded_G].transpose(1,2,0)
        dGs = np.c_[loaded_dG].transpose(1,2,3,0)
        return Gs,dGs

    def calc_G1(self, Rc):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m in range(self.min,self.max):
            atoms = self.atoms_objs[m]
            G1_generator = self.G1(self, m, atoms, Rc)
            G_para[m]  = G1_generator.calc()
            dG_para[m] = G1_generator.deriv()
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    class G1:
        def __init__(self, generator, m, atoms, Rc):
            self.natom = generator.natom
            self.Rc = Rc
            self.index,self.r,self.R,self.tanh,self.fc,self.cosine = generator.neighbour(m, atoms, Rc)
            self.dR = generator.deriv_R(m, self.index, self.r, self.R)

        def calc(self):
            G = np.empty((self.natom))
            for i in range(self.natom):
                G[i] = np.sum(self.fc[i])
            return G
        
        def deriv(self):
            dG = np.empty((self.natom,3*self.natom))
            for i in range(self.natom):
                dgi = - 3/self.Rc * (1 - self.tanh[i][:,None]**2) * self.tanh[i][:,None]**2 * self.dR[i]
                dG[i] = np.sum(dgi, axis=0)
            return dG

    def calc_G2(self, Rc, eta, Rs):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m in range(self.min,self.max):
            atoms = self.atoms_objs[m]
            G2_generator = self.G2(self, m, atoms, Rc, eta, Rs)
            G_para[m]  = G2_generator.calc()
            dG_para[m] = G2_generator.deriv()
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    class G2:
        def __init__(self, generator, m, atoms, Rc, eta, Rs):
            self.natom = generator.natom
            self.Rc,self.eta,self.Rs = Rc,eta,Rs
            self.index,self.r,self.R,self.tanh,self.fc,self.cosine = generator.neighbour(m, atoms, Rc)
            self.dR = generator.deriv_R(m, self.index, self.r, self.R)
        
        def calc(self):
            G = np.empty((self.natom))
            for i in range(self.natom):
                gi = np.exp(- self.eta * (self.R[i] - self.Rs) ** 2) * self.fc[i]
                G[i] = np.sum(gi)
            return G
        
        def deriv(self):
            dG = np.empty((self.natom,3*self.natom))
            for i in range(self.natom):
                gi = np.exp(- self.eta * (self.R[i] - self.Rs) ** 2) * self.fc[i]
                dgi = gi[:,None] * ((-2*self.Rc*self.eta*(self.R[i][:,None]-self.Rs)*self.tanh[i][:,None] + 3*self.tanh[i][:,None]**2 - 3) / (self.Rc * self.tanh[i][:,None])) * self.dR[i]
                dG[i] = np.sum(dgi, axis=0)
            return dG

    def calc_G4(self, Rc, eta, lam, zeta):
        G,dG = np.empty((self.nsample,self.natom)),np.empty((self.nsample,self.natom,3*self.natom))
        G_para,dG_para = np.zeros((self.nsample,self.natom)),np.zeros((self.nsample,self.natom,3*self.natom))
        for m in range(self.min,self.max):
            atoms = self.atoms_objs[m]
            G4_generator = self.G4(self, m, atoms, Rc, eta, lam, zeta)
            G_para[m]  = G4_generator.calc()
            dG_para[m] = G4_generator.deriv()
        self.comm.Allreduce(G_para, G, op=MPI.SUM)
        self.comm.Allreduce(dG_para, dG, op=MPI.SUM)
        return G,dG

    class G4:
        def __init__(self, generator, m, atoms, Rc, eta, lam, zeta):
            self.natom = generator.natom
            self.Rc,self.eta,self.lam,self.zeta = Rc,eta,lam,zeta
            self.index,self.r,self.R,self.tanh,self.fc,self.cosine = generator.neighbour(m, atoms, Rc)
            self.dR = generator.deriv_R(m, self.index, self.r, self.R)
            self.dcos = generator.deriv_cosine(m, self.index, self.r, self.R, self.cosine)
        
        def calc(self):
            G = np.empty((self.natom))
            for i in range(self.natom):
                gi = (2**(1-self.zeta)) * ((1+self.lam*self.cosine[i])**self.zeta) * np.exp(-self.eta*(self.R[i][:,None]**2+self.R[i][None,:]**2)) * self.fc[i][:,None] * self.fc[i][None,:]
                filter = np.identity(len(self.index[i]), dtype=bool)
                gi[filter] = 0.0
                G[i]  = np.sum(gi)
            return G
        
        def deriv(self):
            dG = np.empty((self.natom,3*self.natom))
            for i in range(self.natom):
                angular = (1+self.lam*self.cosine[i])
                common  = (2**(1-self.zeta)) * (angular**(self.zeta-1)) * np.exp(-self.eta*(self.R[i][:,None]**2+self.R[i][None,:]**2)) * self.fc[i][:,None] * self.fc[i][None,:]
                dgi_R   = ((-2*self.Rc*self.eta*self.R[i][:,None]*self.tanh[i][:,None] + 3*self.tanh[i][:,None]**2 - 3) / (self.Rc * self.tanh[i][:,None])) * self.dR[i]
                dgi_cos = self.zeta * self.lam * self.dcos[i]
                dgi = common[:,:,None] * (angular[:,:,None] * (dgi_R[None,:,:] + dgi_R[:,None,:]) + dgi_cos)
                filter = np.identity(len(self.index[i]), dtype=bool)[:,:,None].repeat(3*self.natom, axis=2)
                dgi[filter] = 0.0
                dG[i] = np.sum(dgi, axis=(0,1))
            return dG

    def memorize(f):
        if f.__name__ == 'neighbour':
            cache = {}
            def helper(self, m, atoms, Rc):
                if (m,Rc) not in cache:
                    cache[(m,Rc)] = f(self, m, atoms, Rc)
                return cache[(m,Rc)]
            return helper
        elif f.__name__ == 'deriv_R':
            cache = {}
            def helper(self, m, index, r, R):
                if m not in cache:
                    cache[m] = f(self, m, index, r, R)
                return cache[m]
            return helper
        elif f.__name__ == 'deriv_cosine':
            cache = {}
            def helper(self, m, index, r, R, cosine):
                if m not in cache:
                    cache[m] = f(self, m, index, r, R, cosine)
                return cache[m]
            return helper

    @memorize
    def neighbour(self, m, atoms, Rc):
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
    def deriv_R(self, m, index, r, R):
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
    def deriv_cosine(self, m, index, r, R, cosine):
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
