# -*- coding: utf-8 -*-

# define variables
from config import file_
from config import mpi

# import python modules
from os import path
from os import makedirs
from re import match
from random import shuffle
from collections import defaultdict
from itertools import product
import dill
import numpy as np
from mpi4py import MPI
from chainer.datasets import TupleDataset
from chainer.datasets import get_cross_validation_datasets_random
try:
    from quippy import AtomsReader
    from quippy import AtomsList
    from quippy import AtomsWriter
    from quippy import farray
    from quippy import fzeros
except ImportError:
    print 'Warning: can\'t import quippy.'

from preconditioning import PRECOND
from util import mpiprint


def get_simple_function(name, nsample=1000):
    def make_complex(nsample):
        mesh = int(round(nsample**(1./3)))
        lin = np.linspace(0.1, 1.0, mesh, dtype=np.float32)
        x, y, z = np.meshgrid(lin, lin, lin)
        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        input = np.c_[x, y, z]
        label = (x**2 + np.sin(y) + 3.*np.exp(z) - np.log(x*y)/2 - y/z).reshape(-1, 1)
        dinput = np.identity(3, dtype=np.float32)[None, :, :].repeat(mesh**3, axis=0)
        dlabel = np.c_[2**x - 1/(2*x), np.cos(y) - 1/(2*y) - 1/z, 3.*np.exp(z) + y/z**2].reshape(-1, 3, 1)
        return TupleDataset(input, dinput, label, dlabel)

    def make_LJ(nsample):
        input = np.linspace(0.1, 1.0, nsample, dtype=np.float32).reshape(-1, 1)
        label = 0.001/input**4 - 0.009/input**3
        dinput = np.ones((nsample, 1, 1), dtype=np.float32)
        dlabel = (0.027/input**4 - 0.004/input**5).reshape(-1, 1, 1)
        return TupleDataset(input, dinput, label, dlabel)

    def make_sin(nsample):
        input = np.linspace(-2*3.14, 2*3.14, nsample, dtype=np.float32).reshape(-1, 1)
        label = np.sin(input)
        dinput = np.ones((nsample, 1, 1), dtype=np.float32)
        dlabel = np.cos(input).reshape(-1, 1, 1)
        return TupleDataset(input, dinput, label, dlabel)

    if name == 'complex':
        dataset = make_complex(nsample)
    elif name == 'LJ':
        dataset = make_LJ(nsample)
    elif name == 'sin':
        dataset = make_sin(nsample)
    else:
        raise ValueError("function '{}' is not implemented.")
    dataset.config = name
    return dataset


class AtomicStructureDataset(TupleDataset):
    def __init__(self, hp, config):
        self._hp = hp
        self._data_dir = path.join(file_.data_dir, config)
        with open(path.join(self._data_dir, 'composition.dill')) as f:
            self._composition = dill.load(f)
        xyz_file = path.join(self._data_dir, 'structure.xyz')
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = AtomsReader(xyz_file)[0].n
        self._nforce = 3 * self._natom
        self._config = config
        self._ninput = 2 * len(hp.Rc) + \
            2 * len(hp.Rc)*len(hp.eta)*len(hp.Rs) + \
            3 * len(hp.Rc)*len(hp.eta)*len(hp.lambda_)*len(hp.zeta)
        quo = self._nsample / mpi.size
        rem = self._nsample % mpi.size
        self._count = np.array([quo+1 if i < rem else quo
                                for i in xrange(mpi.size)], dtype=np.int32)
        self._disps = np.array([np.sum(self._count[:i])
                                for i in xrange(mpi.size)], dtype=np.int32)
        self._n = self._count[mpi.rank]  # the number of allocated samples in this node
        if mpi.rank < self._nsample:
            self._atoms_objs = AtomsList(xyz_file, start=mpi.rank, step=mpi.size)
        else:
            self._atoms_objs = []

        Es, Fs = self._make_label()
        Gs, dGs = self._make_input()
        self._datasets = (Gs, dGs, Es, Fs)
        self._length = self._nsample

    @property
    def composition(self):
        return self._composition

    @property
    def config(self):
        return self._config

    @property
    def input(self):
        return self._datasets[0]

    @property
    def dinput(self):
        return self._datasets[1]

    def reset_inputs(self, input, dinput):
        self._datasets = (input, dinput, self._datasets[2], self._datasets[3])

    def _make_label(self):
        EF_file = path.join(self._data_dir, 'Energy_Force.npz')
        if path.exists(EF_file):
            ndarray = np.load(EF_file)
            return ndarray['E'], ndarray['F']
        else:
            mpiprint('{} doesn\'t exist. calculating ...'.format(EF_file))
            Es = np.empty((self._nsample, 1), dtype=np.float32)
            Fs = np.empty((self._nsample, self._nforce, 1), dtype=np.float32)
            Es_send = np.array([data.cohesive_energy for data in self._atoms_objs], dtype=np.float32).reshape(-1, 1)
            Fs_send = np.array([data.force for data in self._atoms_objs], dtype=np.float32).reshape(-1, self._nforce, 1)
            mpi.comm.Allgatherv((Es_send, (self._n), MPI.FLOAT),
                                (Es, (self._count, self._disps), MPI.FLOAT))
            num = self._nforce
            mpi.comm.Allgatherv((Fs_send, (self._n*num), MPI.FLOAT),
                                (Fs, (self._count*num, self._disps*num), MPI.FLOAT))
            np.savez(EF_file, E=Es, F=Fs)
            return Es, Fs

    def _make_input(self):
        Gs = np.empty((self._nsample, self._ninput, self._natom), dtype=np.float32)
        dGs = np.empty((self._nsample, self._ninput, self._natom, self._nforce), dtype=np.float32)
        n = 0

        # type 1
        for Rc in self._hp.Rc:
            filename = path.join(self._data_dir, 'G1-{}.npz'.format(Rc))
            if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+2] = ndarray['G']
                dGs[:, n:n+2] = ndarray['dG']
            else:
                mpiprint('{} doesn\'t exist. calculating ...'.format(filename))
                G = np.empty((self._nsample, 2, self._natom), dtype=np.float32)
                dG = np.empty((self._nsample, 2, self._natom, self._nforce), dtype=np.float32)
                G_send, dG_send = self._calc_G1(Rc)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 2 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+2] = G
                dGs[:, n:n+2] = dG
            n += 2

        # type 2
        for Rc, eta, Rs in product(self._hp.Rc, self._hp.eta, self._hp.Rs):
            filename = path.join(self._data_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
            if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+2] = ndarray['G']
                dGs[:, n:n+2] = ndarray['dG']
            else:
                mpiprint('{} doesn\'t exist. calculating ...'.format(filename))
                G = np.empty((self._nsample, 2, self._natom), dtype=np.float32)
                dG = np.empty((self._nsample, 2, self._natom, self._nforce), dtype=np.float32)
                G_send, dG_send = self._calc_G2(Rc, eta, Rs)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 2 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+2] = G
                dGs[:, n:n+2] = dG
            n += 2

        # type 4
        for Rc, eta, lam, zeta in product(self._hp.Rc, self._hp.eta, self._hp.lambda_, self._hp.zeta):
            filename = path.join(self._data_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
            if path.exists(filename) and Gs[:, n:n+3].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+3] = ndarray['G']
                dGs[:, n:n+3] = ndarray['dG']
            else:
                mpiprint('{} doesn\'t exist. calculating ...'.format(filename))
                G = np.empty((self._nsample, 3, self._natom), dtype=np.float32)
                dG = np.empty((self._nsample, 3, self._natom, self._nforce), dtype=np.float32)
                G_send, dG_send = self._calc_G4(Rc, eta, lam, zeta)
                num = 3 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 3 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+3] = G
                dGs[:, n:n+3] = dG
            n += 3

        return Gs.transpose(0, 2, 1), dGs.transpose(0, 2, 3, 1)

    def _calc_G1(self, Rc):
        G = np.zeros((self._n, 2, self._natom), dtype=np.float32)
        dG = np.zeros((self._n, 2, self._natom, self._nforce), dtype=np.float32)
        for index, (R, fc, tanh, dR), _, _ in self._calc_geometry(['homo', 'hetero'], Rc):
            G[index] = np.sum(fc)
            dG[index] = - 3./Rc * np.dot((1. - tanh**2) * tanh**2, dR)
        return G, dG

    def _calc_G2(self, Rc, eta, Rs):
        G = np.zeros((self._n, 2, self._natom), dtype=np.float32)
        dG = np.zeros((self._n, 2, self._natom, self._nforce), dtype=np.float32)
        for index, (R, fc, tanh, dR), _, _ in self._calc_geometry(['homo', 'hetero'], Rc):
            gi = np.exp(- eta * (R - Rs)**2) * fc
            G[index] = np.sum(gi)
            dG[index] = np.dot(gi * (-2.*eta*(R-Rs) + 3./Rc*(tanh - 1./tanh)), dR)
        return G, dG

    def _calc_G4(self, Rc, eta, lam, zeta):
        G = np.zeros((self._n, 3, self._natom), dtype=np.float32)
        dG = np.zeros((self._n, 3, self._natom, self._nforce), dtype=np.float32)
        for index, (R1, fc1, tanh1, dR1), (R2, fc2, tanh2, dR2), (cos, dcos) in self._calc_geometry(['homo', 'hetero'], Rc):
            ang = 1. + lam * cos
            ang[np.identity(len(R1), dtype=bool)] = 0.
            common = 2.**(-zeta) * ang**(zeta-1) \
                * (np.exp(-eta * R1**2) * fc1)[:, None] \
                * (np.exp(-eta * R2**2) * fc2)[None, :]
            dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2.*eta*R1 + 3./Rc*(tanh1 - 1./tanh1)), dR1)
            dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2.*eta*R2 + 3./Rc*(tanh2 - 1./tanh2)), dR2)
            dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
            G[index] = np.tensordot(common, ang)
            dG[index] = dgi_radial1 + dgi_radial2 + dgi_angular
        for index, (R1, fc1, tanh1, dR1), (R2, fc2, tanh2, dR2), (cos, dcos) in self._calc_geometry(['mix'], Rc):
            ang = 1. + lam * cos
            common = 2.**(1-zeta) * ang**(zeta-1) \
                * (np.exp(-eta * R1**2) * fc1)[:, None] \
                * (np.exp(-eta * R2**2) * fc2)[None, :]
            dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2.*eta*R1 + 3./Rc*(tanh1 - 1./tanh1)), dR1)
            dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2.*eta*R2 + 3./Rc*(tanh2 - 1./tanh2)), dR2)
            dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
            G[index] = np.tensordot(common, ang)
            dG[index] = dgi_radial1 + dgi_radial2 + dgi_angular
        return G, dG

    def memorize_generator(f):
        cache = defaultdict(list)
        done = []

        def helper(self, connect_list, Rc):
            if self._config not in done:
                cache.clear()
                done.append(self._config)
            for con in connect_list:
                if (con, Rc) not in cache:
                    for i, k, homo_rad, hetero_rad, homo_ang, hetero_ang, mix_ang in f(self, Rc):
                        cache[('homo', Rc)].append(((i, 0, k), homo_rad, homo_rad, homo_ang))
                        cache[('hetero', Rc)].append(((i, 1, k), hetero_rad, hetero_rad, hetero_ang))
                        cache[('mix', Rc)].append(((i, 2, k), homo_rad, hetero_rad, mix_ang))
                        yield cache[(con, Rc)][i*self._natom+k]
                else:
                    for ret in cache[(con, Rc)]:
                        yield ret
        return helper

    @memorize_generator
    def _calc_geometry(self, Rc):
        zeros_radial = (np.zeros(1, dtype=np.float32),
                        np.zeros(1, dtype=np.float32),
                        np.ones(1, dtype=np.float32),
                        np.zeros((1, self._nforce), dtype=np.float32))
        zeros_angular = (np.zeros((1, 1), dtype=np.float32),
                         np.zeros((1, 1, self._nforce), dtype=np.float32))
        for i, atoms in enumerate(self._atoms_objs):
            atoms.set_cutoff(Rc)
            atoms.calc_connect()
            for k in xrange(self._natom):
                neighbours = atoms.connect.get_neighbours(k+1)[0] - 1
                if len(neighbours) == 0:
                    yield i, k, zeros_radial, zeros_radial, zeros_angular, zeros_angular, zeros_angular
                    continue

                n_neighb = len(neighbours)
                r, R, cos = self._neighbour(k, n_neighb, atoms)
                dR = self._deriv_R(k, n_neighb, neighbours, r, R)
                dcos = self._deriv_cosine(k, n_neighb, neighbours, r, R, cos)
                R = np.array(R, dtype=np.float32)
                fc = np.tanh(1-R/Rc)**3
                tanh = np.tanh(1-R/Rc)
                cos = np.array(cos, dtype=np.float32)

                element = self._composition['element'][k]  # element of the focused atom
                homo_indices = [index for index, n in enumerate(neighbours) if n in self._composition['index'][element]]
                n_homo = len(homo_indices)
                if n_homo == 0:
                    yield i, k, zeros_radial, (R, fc, tanh, dR), zeros_angular, (cos, dcos), \
                        (np.zeros((1, n_neighb), dtype=np.float32), np.zeros((1, n_neighb, self._nforce), dtype=np.float32))
                    continue
                elif n_homo == n_neighb:
                    yield i, k, (R, fc, tanh, dR), zeros_radial, (cos, dcos), zeros_angular, \
                        (np.zeros((n_neighb, 1), dtype=np.float32), np.zeros((n_neighb, 1, self._nforce), dtype=np.float32))
                    continue

                homo_R = np.take(R, homo_indices)
                homo_fc = np.take(fc, homo_indices)
                homo_tanh = np.take(tanh, homo_indices)
                homo_dR = np.take(dR, homo_indices, axis=0)
                homo_cos = np.take(np.take(cos, homo_indices, axis=0), homo_indices, axis=1)
                homo_dcos = np.take(np.take(dcos, homo_indices, axis=0), homo_indices, axis=1)
                homo_radial = (homo_R, homo_fc, homo_tanh, homo_dR)
                homo_angular = (homo_cos, homo_dcos)

                hetero_R = np.delete(R, homo_indices)
                hetero_fc = np.delete(fc, homo_indices)
                hetero_tanh = np.delete(tanh, homo_indices)
                hetero_dR = np.delete(dR, homo_indices, axis=0)
                hetero_cos = np.delete(np.delete(cos, homo_indices, axis=0), homo_indices, axis=1)
                hetero_dcos = np.delete(np.delete(dcos, homo_indices, axis=0), homo_indices, axis=1)
                hetero_radial = (hetero_R, hetero_fc, hetero_tanh, hetero_dR)
                hetero_angular = (hetero_cos, hetero_dcos)

                mix_cos = np.delete(np.take(cos, homo_indices, axis=0), homo_indices, axis=1)
                mix_dcos = np.delete(np.take(dcos, homo_indices, axis=0), homo_indices, axis=1)
                mix_angular = (mix_cos, mix_dcos)

                yield i, k, homo_radial, hetero_radial, homo_angular, hetero_angular, mix_angular

    def _neighbour(self, k, n_neighb, atoms):
        r = []
        R = []
        ra = r.append
        Ra = R.append
        for l in xrange(n_neighb):
            dist = farray(0.0)
            diff = fzeros(3)
            atoms.neighbour(k+1, l+1, distance=dist, diff=diff)
            ra(diff.tolist())
            Ra(dist.tolist())
        cos = [[0. if l == m else atoms.cosine_neighbour(k+1, l+1, m+1)
                for m in xrange(n_neighb)]
               for l in xrange(n_neighb)]
        return r, R, cos

    def _deriv_R(self, k, n_neighb, neighbours, r, R):
        dR = np.zeros((n_neighb, self._nforce), dtype=np.float32)
        for l, n in product(xrange(n_neighb), xrange(self._nforce)):
            if n % self._natom == k:
                dR[l, n] = - r[l][n/self._natom] / R[l]
            elif n % self._natom == neighbours[l]:
                dR[l, n] = + r[l][n/self._natom] / R[l]
        return dR

    def _deriv_cosine(self, k, n_neighb, neighbours, r, R, cos):
        dcos = np.zeros((n_neighb, n_neighb, self._nforce), dtype=np.float32)
        for l, m, n in product(xrange(n_neighb), xrange(n_neighb), xrange(self._nforce)):
            if n % self._natom == k:
                dcos[l, m, n] = (r[l][n/self._natom] / R[l]**2 + r[m][n/self._natom] / R[m]**2) * cos[l][m] \
                    - (r[l][n/self._natom] + r[m][n/self._natom]) / (R[l] * R[m])
            elif n % self._natom == neighbours[l]:
                dcos[l, m, n] = - (r[l][n/self._natom] / R[l]**2) * cos[l][m] \
                    + r[m][n/self._natom] / (R[l] * R[m])
            elif n % self._natom == neighbours[m]:
                dcos[l, m, n] = - (r[m][n/self._natom] / R[m]**2) * cos[l][m] \
                    + r[l][n/self._natom] / (R[l] * R[m])
        return dcos


class DataGenerator(object):
    def __init__(self, hp):
        self._hp = hp
        self._precond = PRECOND[hp.preconditioning]()
        self._config_type_file = path.join(file_.data_dir, 'config_type.dill')
        if mpi.rank == 0 and not path.exists(self._config_type_file):
            self._parse_xyzfile()

    def __iter__(self):
        with open(self._config_type_file, 'r') as f:
            config_type = dill.load(f)
        alldataset = []
        elements = set()
        for type in file_.train_config:
            for config in filter(lambda config: match(type, config) or type == 'all', config_type):
                dataset = AtomicStructureDataset(self._hp, config)
                self._precond.decompose(dataset)
                alldataset.append(dataset)
                elements.update(dataset.composition['element'])

        if self._hp.cross_validation:
            splited = [[(train, val, d.config, d.composition)
                        for train, val in get_cross_validation_datasets_random(d, n_fold=self._hp.cross_validation)]
                       for d in alldataset]
            for dataset in zip(*splited):
                yield dataset, elements
        else:
            yield [[(d, None, d.config, d.composition) for d in alldataset]]

    def save(self, save_dir):
        with open(path.join(save_dir, 'preconditioning.dill'), 'w') as f:
            dill.dump(self._precond, f)

    def load(self, save_dir):
        with open(path.join(save_dir, 'preconditioning.dill'), 'r') as f:
            self._precond = dill.load(f)

    def _parse_xyzfile(self):
        mpiprint('config_type.dill is not found.\nLoad all data from xyz file ...')
        config_type = set()
        alldataset = defaultdict(list)
        rawdata = AtomsReader(path.join(file_.data_dir, file_.xyz_file))
        for data in rawdata:
            # config_typeが構造と組成を表していないものをスキップ
            config = data.config_type
            if match('Single', config) or match('Sheet', config) or match('Interface', config) or \
                    match('Cluster', config) or match('Amorphous', config):
                continue
            config_type.add(config)
            alldataset[config].append(data)
        with open(self._config_type_file, 'w') as f:
            dill.dump(config_type, f)

        for config in config_type:
            composition = {'index': defaultdict(set), 'element': []}
            for i, atom in enumerate(alldataset[config][0]):
                composition['index'][atom.symbol].add(i)
                composition['element'].append(atom.symbol)

            data_dir = path.join(file_.data_dir, config)
            makedirs(data_dir)
            shuffle(alldataset[config])
            writer = AtomsWriter(path.join(data_dir, 'structure.xyz'))
            for i, data in enumerate(alldataset[config]):
                if data.cohesive_energy > 0.0:
                    continue
                # 2~4体の構造はpbcをFalseに
                if match('Quadruple', config) or match('Triple', config) or match('Dimer', config):
                    data.set_pbc([False, False, False])
                writer.write(data)
            writer.close()
            with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                dill.dump(composition, f)


if __name__ == '__main__':
    for _ in DataGenerator():
        pass
