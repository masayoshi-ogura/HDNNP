# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import bool_
from config import file_

# import python modules
from os import path
from os import mkdir
from collections import defaultdict
import json
import numpy as np
from mpi4py import MPI
try:
    from quippy import AtomsReader
    from quippy import farray
    from quippy import fzeros
    from quippy import frange
except ImportError:
    print 'Warning: can\'t import quippy, so can\'t calculate symmetric functions, but load them.'
    bool_.CALC_INPUT = False


class LabelGenerator(object):
    def __init__(self, train_npy_dir):
        self.train_npy_dir = train_npy_dir

    def make(self, atoms_objs, natom, nsample):
        Es = np.array([data.cohesive_energy for data in atoms_objs]).reshape((nsample, 1))
        Fs = np.array([np.array(data.force).T for data in atoms_objs]).reshape((nsample, 3*natom, 1))
        np.savez(path.join(self.train_npy_dir, 'Energy_Force.npz'), Energy=Es, Force=Fs)
        return Es, Fs

    def load(self):
        ndarray = np.load(path.join(self.train_npy_dir, 'Energy_Force.npz'))
        Es = ndarray['Energy']
        Fs = ndarray['Force']
        return Es, Fs


class EFGenerator(LabelGenerator):
    pass


class InputGenerator(object):
    def __init__(self, train_npy_dir):
        self.train_npy_dir = train_npy_dir

    def make(self, comm, size, rank, atoms_objs, natom, nsample, ninput, composition):
        self.comm = comm
        self.atoms_objs = atoms_objs
        self.natom = natom
        self.nsample = nsample
        self.composition = composition
        quo, rem = self.nsample/size, self.nsample % size
        if rank < rem:
            self.min, self.max = rank*(quo+1), (rank+1)*(quo+1)
        else:
            self.min, self.max = rank*quo+rem, (rank+1)*quo+rem

        Gs = np.empty((ninput, self.nsample, self.natom))
        dGs = np.empty((ninput, self.nsample, self.natom, 3*self.natom))

        n = 0
        for Rc in hp.Rcs:
            filename = path.join(self.train_npy_dir, 'G1-{}.npz'.format(Rc))
            if path.exists(filename) and Gs[n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[n:n+2] = ndarray['G']
                dGs[n:n+2] = ndarray['dG']
            else:
                G, dG = self.__calc_G(1, Rc)
                np.savez(filename, G=G, dG=dG)
                Gs[n:n+2] = G
                dGs[n:n+2] = dG
            n += 2

            for eta in hp.etas:
                for Rs in hp.Rss:
                    filename = path.join(self.train_npy_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
                    if path.exists(filename) and Gs[n:n+2].shape == np.load(filename)['G'].shape:
                        ndarray = np.load(filename)
                        Gs[n:n+2] = ndarray['G']
                        dGs[n:n+2] = ndarray['dG']
                    else:
                        G, dG = self.__calc_G(2, Rc, eta, Rs)
                        np.savez(filename, G=G, dG=dG)
                        Gs[n:n+2] = G
                        dGs[n:n+2] = dG
                    n += 2

                for lam in hp.lams:
                    for zeta in hp.zetas:
                        filename = path.join(self.train_npy_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
                        if path.exists(filename) and Gs[n:n+3].shape == np.load(filename)['G'].shape:
                            ndarray = np.load(filename)
                            Gs[n:n+3] = ndarray['G']
                            dGs[n:n+3] = ndarray['dG']
                        else:
                            G, dG = self.__calc_G(4, Rc, eta, lam, zeta)
                            np.savez(filename, G=G, dG=dG)
                            Gs[n:n+3] = G
                            dGs[n:n+3] = dG
                        n += 3
        return Gs.transpose(1, 2, 0), dGs.transpose(1, 2, 3, 0)

    def load(self):
        loaded_G, loaded_dG = [], []
        for Rc in hp.Rcs:
            filename = path.join(self.train_npy_dir, 'G1-{}.npz'.format(Rc))
            if path.exists(filename):
                ndarray = np.load(filename)
                loaded_G.append(ndarray['G'])
                loaded_dG.append(ndarray['dG'])

            for eta in hp.etas:
                for Rs in hp.Rss:
                    filename = path.join(self.train_npy_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
                    if path.exists(filename):
                        ndarray = np.load(filename)
                        loaded_G.append(ndarray['G'])
                        loaded_dG.append(ndarray['dG'])

                for lam in hp.lams:
                    for zeta in hp.zetas:
                        filename = path.join(self.train_npy_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
                        if path.exists(filename):
                            ndarray = np.load(filename)
                            loaded_G.append(ndarray['G'])
                            loaded_dG.append(ndarray['dG'])

        Gs = np.c_[loaded_G].transpose(1, 2, 0)
        dGs = np.c_[loaded_dG].transpose(1, 2, 3, 0)
        return Gs, dGs

    def __calc_G(self, type, *params):
        if type == 1:
            num = 2
            generator = self.__G1_generator(*params)
        elif type == 2:
            num = 2
            generator = self.__G2_generator(*params)
        elif type == 4:
            num = 3
            generator = self.__G4_generator(*params)
        else:
            raise ValueError('available symmetric function type is 1, 2, or 4.')

        # prepare np arrays
        # homo(, mix), hetero
        Gs = np.empty((num, self.nsample, self.natom))
        Gs_para = np.zeros((num, self.nsample, self.natom))
        dGs = np.empty((num, self.nsample, self.natom, 3*self.natom))
        dGs_para = np.zeros((num, self.nsample, self.natom, 3*self.natom))
        # calculate
        # n:1..num, m:self.min..self.max
        for m, n, G, dG in generator:
            Gs_para[n][m] = G
            dGs_para[n][m] = dG
        # MPI Allreduce
        self.comm.Allreduce(Gs_para, Gs, op=MPI.SUM)
        self.comm.Allreduce(dGs_para, dGs, op=MPI.SUM)
        # sclaing
        for Gs_, dGs_ in zip(Gs, dGs):
            min, max = np.min(Gs_), np.max(Gs_)
            Gs_ = 2 * (Gs_ - min) / (max - min) - 1
            dGs_ = (2 * dGs_) / (max - min)
        return Gs, dGs

    def __G1_generator(self, Rc):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'hetero']):
                G = np.empty((self.natom))
                dG = np.empty((self.natom, 3*self.natom))
                for i, radial, _, _ in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R, fc, tanh, dR = radial
                    G[i] = np.sum(fc)
                    dgi = - 3/Rc * (1 - tanh[:, None]**2) * tanh[:, None]**2 * dR
                    dG[i] = np.sum(dgi, axis=0)
                yield m, n, G, dG

    def __G2_generator(self, Rc, eta, Rs):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'hetero']):
                G = np.empty((self.natom))
                dG = np.empty((self.natom, 3*self.natom))
                for i, radial, _, _ in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R, fc, tanh, dR = radial
                    gi = np.exp(- eta * (R - Rs) ** 2) * fc
                    G[i] = np.sum(gi)
                    dgi = gi[:, None] * ((-2*Rc*eta*(R-Rs)*tanh + 3*tanh**2 - 3) / (Rc * tanh))[:, None] * dR
                    dG[i] = np.sum(dgi, axis=0)
                yield m, n, G, dG

    def __G4_generator(self, Rc, eta, lam, zeta):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'mix', 'hetero']):
                G = np.empty((self.natom))
                dG = np.empty((self.natom, 3*self.natom))
                for i, radial1, radial2, angular in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R1, fc1, tanh1, dR1 = radial1
                    R2, fc2, tanh2, dR2 = radial2
                    cos, dcos = angular

                    gi = (2**(1-zeta)) * ((1+lam*cos)**zeta) * \
                        np.exp(-eta*(R1[:, None]**2+R2[None, :]**2)) * \
                        fc1[:, None] * fc2[None, :]
                    dgi_radial1 = gi[:, :, None] * ((-2*Rc*eta*R1*tanh1 + 3*tanh1**2 - 3) / (Rc * tanh1))[:, None, None] * dR1[:, None, :]
                    dgi_radial2 = gi[:, :, None] * ((-2*Rc*eta*R2*tanh2 + 3*tanh2**2 - 3) / (Rc * tanh2))[None, :, None] * dR2[None, :, :]
                    dgi_angular = zeta * lam * (2**(1-zeta)) * ((1+lam*cos)**(zeta-1)) * \
                        np.exp(-eta*(R1[:, None, None]**2+R2[None, :, None]**2)) * \
                        fc1[:, None, None] * fc2[None, :, None] * dcos
                    dgi = dgi_radial1 + dgi_radial2 + dgi_angular

                    if con in ['homo', 'hetero']:
                        filter1 = np.identity(len(R1), dtype=bool)
                        filter2 = filter1[:, :, None].repeat(3*self.natom, axis=2)
                        gi[filter1] = 0.
                        dgi[filter2] = 0.
                        G[i] = np.sum(gi) / 2
                        dG[i] = np.sum(dgi, axis=(0, 1)) / 2
                    elif con == 'mix':
                        G[i] = np.sum(gi)
                        dG[i] = np.sum(dgi, axis=(0, 1))
                yield m, n, G, dG

    def memorize(f):
        if f.__name__ == '__neighbour':
            cache = {}

            def helper(self, m, con, atoms, Rc):
                if (m, con, Rc) not in cache:
                    cache[(m, con, Rc)] = f(self, m, atoms, Rc)
                return cache[(m, con, Rc)]
            return helper
        if f.__name__ == '__distance_ij':
            cache = {}

            def helper(self, m, i, atoms, Rc):
                if (m, i, Rc) not in cache:
                    cache[(m, i, Rc)] = f(self, m, i, atoms, Rc)
                return cache[(m, i, Rc)]
            return helper
        if f.__name__ == '__cosine_ijk':
            cache = {}

            def helper(self, m, i, atoms, Rc):
                if (m, i, Rc) not in cache:
                    cache[(m, i, Rc)] = f(self, m, i, atoms, Rc)
                return cache[(m, i, Rc)]
            return helper
        if f.__name__ == '__deriv_R':
            cache = {}

            def helper(self, m, i, neighbours, r, R, Rc):
                if (m, i, Rc) not in cache:
                    cache[(m, i, Rc)] = f(self, m, i, neighbours, r, R, Rc)
                return cache[(m, i, Rc)]
            return helper
        if f.__name__ == '__deriv_cosine':
            cache = {}

            def helper(self, m, i, neighbours, r, R, cos, Rc):
                if (m, i, Rc) not in cache:
                    cache[(m, i, Rc)] = f(self, m, i, neighbours, r, R, cos, Rc)
                return cache[(m, i, Rc)]
            return helper

    @memorize
    def __neighbour(self, m, con, atoms, Rc):
        atoms.set_pbc(bool_.pbc)  # temporary flags TODO: set periodic boundary conditions correctly in xyz file.
        atoms.set_cutoff(Rc)
        atoms.calc_connect()

        for i in range(self.natom):
            neighbours = atoms.connect.get_neighbours(i+1) - 1
            r, R, fc, tanh = self.__distance_ij(m, i, atoms, Rc)
            cos = self.__cosine_ijk(m, i, atoms, Rc)
            dR = self.__deriv_R(m, i, neighbours, r, R, Rc)
            dcos = self.__deriv_cosine(m, i, neighbours, r, R, cos, Rc)

            symbol = self.composition['symbol'][i]  # symbol of the focused atom
            index = self.composition['index'][symbol]  # index of the same species of the focused atom
            homo_neighbours = [j for j, n in enumerate(neighbours) if n in index]
            hetero_neighbours = [j for j, n in enumerate(neighbours) if n not in index]

            homo_R = np.delete(R, hetero_neighbours)
            homo_fc = np.delete(fc, hetero_neighbours)
            homo_tanh = np.delete(tanh, hetero_neighbours)
            homo_dR = np.delete(dR, hetero_neighbours, axis=0)
            homo_cos = np.delete(np.delete(cos, hetero_neighbours, axis=0), hetero_neighbours, axis=1)
            homo_dcos = np.delete(np.delete(dcos, hetero_neighbours, axis=0), hetero_neighbours, axis=1)
            homo_radial = (homo_R, homo_fc, homo_tanh, homo_dR)
            homo_angular = (homo_cos, homo_dcos)

            hetero_R = np.delete(R, homo_neighbours)
            hetero_fc = np.delete(fc, homo_neighbours)
            hetero_tanh = np.delete(tanh, homo_neighbours)
            hetero_dR = np.delete(dR, homo_neighbours, axis=0)
            hetero_cos = np.delete(np.delete(cos, homo_neighbours, axis=0), homo_neighbours, axis=1)
            hetero_dcos = np.delete(np.delete(dcos, homo_neighbours, axis=0), homo_neighbours, axis=1)
            hetero_radial = (hetero_R, hetero_fc, hetero_tanh, hetero_dR)
            hetero_angular = (hetero_cos, hetero_dcos)

            mix_cos = np.delete(np.delete(cos, hetero_neighbours, axis=0), homo_neighbours, axis=1)
            mix_dcos = np.delete(np.delete(dcos, hetero_neighbours, axis=0), homo_neighbours, axis=1)
            mix_angular = (mix_cos, mix_dcos)

            if con == 'homo':
                yield i, homo_radial, homo_radial, homo_angular
            elif con == 'hetero':
                yield i,  hetero_radial, hetero_radial, hetero_angular
            elif con == 'mix':
                yield i,  homo_radial, hetero_radial, mix_angular

    @memorize
    def __distance_ij(self, m, i, atoms, Rc):
        ri, Ri = [], []
        for n in frange(atoms.n_neighbours(i+1)):
            dist = farray(0.0)
            diff = fzeros(3)
            atoms.neighbour(i+1, n, distance=dist, diff=diff)
            ri.append(diff)
            Ri.append(dist)
        r = np.array(ri)
        R = np.array(Ri)
        fc = np.tanh(1-R/Rc)**3
        tanh = np.tanh(1-R/Rc)
        return r, R, fc, tanh

    @memorize
    def __cosine_ijk(self, m, i, atoms, Rc):
        n_neighb = atoms.n_neighbours(i+1)
        cos = np.zeros((n_neighb, n_neighb))
        for j in range(n_neighb):
            for k in range(n_neighb):
                if k == j:
                    pass
                else:
                    cos[j][k] = atoms.cosine_neighbour(i+1, j+1, k+1)
        return cos

    @memorize
    def __deriv_R(self, m, i, neighbours, r, R, Rc):
        n_neighb = len(neighbours)
        dR = np.zeros((n_neighb, 3*self.natom))
        for j in range(n_neighb):
            for l in range(self.natom):
                if l == i:
                    for alpha in range(3):
                        dR[j][3*l+alpha] = - r[j][alpha] / R[j]
                elif l == neighbours[j]:
                    for alpha in range(3):
                        dR[j][3*l+alpha] = + r[j][alpha] / R[j]
        return dR

    @memorize
    def __deriv_cosine(self, m, i, neighbours, r, R, cos, Rc):
        n_neighb = len(neighbours)
        dcos = np.zeros((n_neighb, n_neighb, 3*self.natom))
        for j in range(n_neighb):
            for k in range(n_neighb):
                for l in range(self.natom):
                    if l == i:
                        for alpha in range(3):
                            dcos[j][k][3*l+alpha] = (+ r[j][alpha] / R[j]**2) * cos[j][k] + \
                                                     (+ r[k][alpha] / R[k]**2) * cos[j][k] + \
                                                     (- (r[j][alpha] + r[k][alpha])) / (R[j] * R[k])
                    elif l == neighbours[j]:
                        for alpha in range(3):
                            dcos[j][k][3*l+alpha] = (- r[j][alpha] / R[j]**2) * cos[j][k] + \
                                                     (+ r[k][alpha]) / (R[j] * R[k])
                    elif l == neighbours[k]:
                        for alpha in range(3):
                            dcos[j][k][3*l+alpha] = (- r[k][alpha] / R[k]**2) * cos[j][k] + \
                                                     (+ r[j][alpha]) / (R[j] * R[k])
        return dcos


class SFGenerator(InputGenerator):
    pass


def make_dataset(comm, rank, size):
    train_xyz_file = path.join(file_.train_dir, 'xyz', file_.xyzfile)
    train_npy_dir = path.join(file_.train_dir, 'npy', file_.name)
    train_composition_file = path.join(train_npy_dir, 'composition.json')
    if rank == 0 and not path.exists(train_npy_dir):
        mkdir(train_npy_dir)
    label = LabelGenerator(train_npy_dir)
    input = InputGenerator(train_npy_dir)

    if bool_.CALC_INPUT:
        dataset = AtomsReader(train_xyz_file)
        coordinates = []
        for data in dataset:
            if data.config_type == file_.name and data.force.min() > -10. and data.force.max() < 10.:
                coordinates.append(data)
        natom = coordinates[0].n
        nsample = len(coordinates)
        Es, Fs = label.make(coordinates, natom, nsample)
        ninput = 2 * len(hp.Rcs) + \
            2 * len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
            3 * len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
        composition = {'number': defaultdict(lambda: 0), 'index': defaultdict(set), 'symbol': []}
        for i, atom in enumerate(coordinates[0]):
            composition['number'][atom.symbol] += 1
            composition['index'][atom.symbol].add(i)
            composition['symbol'].append(atom.symbol)
        with open(train_composition_file, 'w') as f:
            json.dump(composition, f)
        Gs, dGs = input.make(comm, size, rank, coordinates, natom, nsample, ninput, composition)
    else:
        Es, Fs = label.load()
        Gs, dGs = input.load()
        nsample, natom, ninput = Gs.shape
        with open(train_composition_file, 'r') as f:
            composition = json.load(f)

    return Es, Fs, Gs, dGs, natom, nsample, ninput, composition


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    make_dataset(comm, rank, size)
