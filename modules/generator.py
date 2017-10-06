# -*- coding: utf-8 -*-

# define variables
from config import hp
from config import bool_
from config import file_

# import python modules
from os import path
from os import makedirs
from re import match
from random import shuffle
from collections import defaultdict
import dill
import numpy as np
from mpi4py import MPI
try:
    from quippy import AtomsReader
    from quippy import AtomsWriter
    from quippy import farray
    from quippy import fzeros
    from quippy import frange
except ImportError:
    print 'Warning: can\'t import quippy.'


class Generator(object):
    def __init__(self, comm, size, rank, data_dir, name, atoms_objs, natom, nsample, ninput, composition):
        self.comm = comm
        self.rank = rank
        self.data_dir = data_dir
        self.name = name
        self.atoms_objs = atoms_objs
        self.natom = natom
        self.nsample = nsample
        self.ninput = ninput
        self.composition = composition
        quo, rem = self.nsample/size, self.nsample % size
        if rank < rem:
            self.min, self.max = rank*(quo+1), (rank+1)*(quo+1)
        else:
            self.min, self.max = rank*quo+rem, (rank+1)*quo+rem

    def make_label(self):
        EF_file = path.join(self.data_dir, 'Energy_Force.npz')
        if self.rank == 0 and not path.exists(EF_file):
            Es = np.array([data.cohesive_energy for data in self.atoms_objs]).reshape((self.nsample, 1))
            Fs = np.array([np.array(data.force).T for data in self.atoms_objs]).reshape((self.nsample, 3*self.natom, 1))
            np.savez(EF_file, Energy=Es, Force=Fs)
        self.comm.Barrier()
        ndarray = np.load(EF_file)
        Es = ndarray['Energy']
        Fs = ndarray['Force']
        return Es, Fs

    def make_input(self):
        Gs = np.empty((self.ninput, self.nsample, self.natom))
        dGs = np.empty((self.ninput, self.nsample, self.natom, 3*self.natom))

        n = 0
        for Rc in hp.Rcs:
            filename = path.join(self.data_dir, 'G1-{}.npz'.format(Rc))
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
                    filename = path.join(self.data_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
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
                        filename = path.join(self.data_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
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

        Gs = np.empty((num, self.nsample, self.natom))
        Gs_para = np.zeros((num, self.nsample, self.natom))
        dGs = np.empty((num, self.nsample, self.natom, 3*self.natom))
        dGs_para = np.zeros((num, self.nsample, self.natom, 3*self.natom))
        # n:1..num, m:self.min..self.max
        for m, n, G, dG in generator:
            Gs_para[n][m] = G
            dGs_para[n][m] = dG
        self.comm.Allreduce(Gs_para, Gs, op=MPI.SUM)
        self.comm.Allreduce(dGs_para, dGs, op=MPI.SUM)
        # # sclaing
        # for i in range(num):
        #     min, max = np.min(Gs[i]), np.max(Gs[i])
        #     if min == 0. and max == 0.:  # When there are zero heteroatoms, 'hetero' and 'mix' are zero matrices
        #         continue
        #     Gs[i] = 2 * (Gs[i] - min) / (max - min) - 1
        #     dGs[i] = (2 * dGs[i]) / (max - min)
        return Gs, dGs

    def __G1_generator(self, Rc):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'hetero']):
                G = np.zeros((self.natom))
                dG = np.zeros((self.natom, 3*self.natom))
                for i, radial, _, _ in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R, fc, tanh, dR = radial
                    if len(R) == 0:
                        continue
                    G[i] = np.sum(fc)
                    dG[i] = - 3/Rc * np.dot((1 - tanh**2) * tanh**2, dR)
                yield m, n, G, dG

    def __G2_generator(self, Rc, eta, Rs):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'hetero']):
                G = np.zeros((self.natom))
                dG = np.zeros((self.natom, 3*self.natom))
                for i, radial, _, _ in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R, fc, tanh, dR = radial
                    if len(R) == 0:
                        continue
                    gi = np.exp(- eta * (R - Rs) ** 2) * fc
                    G[i] = np.sum(gi)
                    dG[i] = np.dot(gi * (-2*eta*(R-Rs) + 3/Rc*(tanh - 1/tanh)), dR)
                yield m, n, G, dG

    def __G4_generator(self, Rc, eta, lam, zeta):
        for m in range(self.min, self.max):
            for n, con in enumerate(['homo', 'mix', 'hetero']):
                G = np.zeros((self.natom))
                dG = np.zeros((self.natom, 3*self.natom))
                for i, radial1, radial2, angular in self.__neighbour(m, con, self.atoms_objs[m], Rc):
                    R1, fc1, tanh1, dR1 = radial1
                    R2, fc2, tanh2, dR2 = radial2
                    cos, dcos = angular
                    if len(R2) == 0:
                        continue

                    ang = 1 + lam * cos
                    if con in ['homo', 'hetero']:
                        ang[np.identity(len(R1), dtype=bool)] = 0.0
                        common = 2**(-zeta) * ang**(zeta-1) * (np.exp(-eta * R1**2) * fc1)[:, None] * (np.exp(-eta * R2**2) * fc2)[None, :]
                    elif con == 'mix':
                        common = 2**(1-zeta) * ang**(zeta-1) * (np.exp(-eta * R1**2) * fc1)[:, None] * (np.exp(-eta * R2**2) * fc2)[None, :]
                    G[i] = np.tensordot(common, ang)
                    dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2*eta*R1 + 3/Rc*(tanh1 - 1/tanh1)), dR1)
                    dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2*eta*R2 + 3/Rc*(tanh2 - 1/tanh2)), dR2)
                    dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
                    dG[i] = dgi_radial1 + dgi_radial2 + dgi_angular
                yield m, n, G, dG

    def memorize(f):
        cache = {}
        done = []

        def helper(self, m, i, Rc, *args):
            if self.name not in done:
                cache.clear()
                done.append(self.name)
            if (m, i, Rc) not in cache:
                cache[(m, i, Rc)] = f(self, m, i, Rc, *args)
            return cache[(m, i, Rc)]
        return helper

    # yieldと合わせるのよくなさそう(動作がどうなるかわからん)ので外す
    # @memorize
    def __neighbour(self, m, con, atoms, Rc):
        atoms.set_cutoff(Rc)
        atoms.calc_connect()

        for i in range(self.natom):
            neighbours = atoms.connect.get_neighbours(i+1)[0] - 1
            r, R = self.__distance_ij(m, i, Rc, atoms)
            cos = self.__cosine_ijk(m, i, Rc, atoms)
            dR = self.__deriv_R(m, i, Rc, neighbours, r, R)
            dcos = self.__deriv_cosine(m, i, Rc, neighbours, r, R, cos)
            R = np.array(R)
            fc = np.tanh(1-R/Rc)**3
            tanh = np.tanh(1-R/Rc)
            cos = np.array(cos)

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
    def __distance_ij(self, m, i, Rc, atoms):
        r, R = [], []
        for n in frange(atoms.n_neighbours(i+1)):
            dist = farray(0.0)
            diff = fzeros(3)
            atoms.neighbour(i+1, n, distance=dist, diff=diff)
            r.append(diff.tolist())
            R.append(dist.tolist())
        return r, R

    @memorize
    def __cosine_ijk(self, m, i, Rc, atoms):
        n_neighb = atoms.n_neighbours(i+1)
        cos = [[0.0 if k == j else atoms.cosine_neighbour(i+1, j+1, k+1)
                for k in range(n_neighb)]
               for j in range(n_neighb)]
        return cos

    @memorize
    def __deriv_R(self, m, i, Rc, neighbours, r, R):
        n_neighb = len(neighbours)
        dR = np.array([[- r[j][l % 3] / R[j] if l/3 == i
                        else + r[j][l % 3] / R[j] if l/3 == neighbours[j]
                        else 0.0
                        for l in range(3*self.natom)]
                       for j in range(n_neighb)])
        return dR

    @memorize
    def __deriv_cosine(self, m, i, Rc, neighbours, r, R, cos):
        n_neighb = len(neighbours)
        dcos = np.array([[[(r[j][l % 3] / R[j]**2 + r[k][l % 3] / R[k]**2) * cos[j][k] - (r[j][l % 3] + r[k][l % 3]) / (R[j] * R[k]) if l/3 == i
                           else - (r[j][l % 3] / R[j]**2) * cos[j][k] + r[k][l % 3] / (R[j] * R[k]) if l/3 == neighbours[j]
                           else - (r[k][l % 3] / R[k]**2) * cos[j][k] + r[k][l % 3] / (R[j] * R[k]) if l/3 == neighbours[k]
                           else 0.0
                           for l in range(3*self.natom)]
                          for k in range(n_neighb)]
                         for j in range(n_neighb)])
        return dcos


def make_dataset(comm, rank, size, mode):
    xyz_file = path.join(file_.data_dir, file_.xyz_file)
    config_type_file = path.join(file_.data_dir, 'config_type.dill')
    if not path.exists(config_type_file) and rank == 0:
        print 'config_type.dill is not found.\nLoad all data from xyz file ...'
        config_type = set()
        alldataset = defaultdict(list)
        rawdata = AtomsReader(xyz_file)
        for data in rawdata:
            # config_typeが構造と組成を表していないものをスキップ
            config = data.config_type
            if match('Single', config) or match('Sheet', config) or match('Interface', config) or \
                    match('Cluster', config) or match('Amorphous', config):
                continue
            config_type.add(config)
            alldataset[config].append(data)
        with open(config_type_file, 'w') as f:
            dill.dump(config_type, f)
        print 'Separate all dataset to training & validation & test dataset ...'
        for config in config_type:
            composition = {'number': defaultdict(lambda: 0), 'index': defaultdict(set), 'symbol': []}
            for i, data in enumerate(alldataset[config][0]):
                composition['number'][data.symbol] += 1
                composition['index'][data.symbol].add(i)
                composition['symbol'].append(data.symbol)

            data_dir = path.join(file_.data_dir, config)
            shuffle(alldataset[config])
            if not path.exists(data_dir):
                makedirs(path.join(data_dir, 'training'))
                makedirs(path.join(data_dir, 'validation'))
                makedirs(path.join(data_dir, 'test'))
            training_writer = AtomsWriter(path.join(data_dir, 'training', 'structure.xyz'))
            validation_writer = AtomsWriter(path.join(data_dir, 'validation', 'structure.xyz'))
            test_writer = AtomsWriter(path.join(data_dir, 'test', 'structure.xyz'))
            sep1 = len(alldataset[config]) * 7 / 10
            sep2 = len(alldataset[config]) * 9 / 10
            for i, data in enumerate(alldataset[config]):
                if data.cohesive_energy > 0.0:
                    continue
                # 2~4体の構造はpbcをFalseに
                if match('Quadruple', config) or match('Triple', config) or match('Dimer', config):
                    data.set_pbc([False, False, False])
                if i < sep1:
                    training_writer.write(data)
                elif i < sep2:
                    validation_writer.write(data)
                else:
                    test_writer.write(data)
            with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                dill.dump(composition, f)

    comm.Barrier()
    with open(config_type_file, 'r') as f:
        config_type = dill.load(f)
    for config in config_type:
        for type in file_.train_config:
            if match(type, config) or type == 'all':
                break
        else:
            continue
        if rank == 0:
            print '----------------------{}-----------------------'.format(config)
        data_dir = path.join(file_.data_dir, config, mode)
        xyz_file = path.join(data_dir, 'structure.xyz')
        # if not path.exists(xyz_file):
        #     if rank == 0:
        #         print 'xyz file {} is not found. maybe there are too few samples.'.format(xyz_file)
        #     continue

        dataset = AtomsReader(xyz_file)
        natom = dataset[0].n
        nsample = len(dataset)
        ninput = 2 * len(hp.Rcs) + \
            2 * len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
            3 * len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
        with open(path.join(data_dir, '..', 'composition.dill')) as f:
            composition = dill.load(f)
        generator = Generator(comm, size, rank, data_dir, config, dataset, natom, nsample, ninput, composition)
        Es, Fs = generator.make_label()
        Gs, dGs = generator.make_input()
        yield config, Es, Fs, Gs, dGs, natom, nsample, ninput, composition


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for _ in make_dataset(comm, rank, size, 'training'):
        pass
