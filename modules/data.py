# -*- coding: utf-8 -*-

# define variables
from config import hp
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
try:
    from quippy import AtomsReader
    from quippy import AtomsList
    from quippy import AtomsWriter
    from quippy import farray
    from quippy import fzeros
except ImportError:
    print 'Warning: can\'t import quippy.'

from preconditioning import PRECOND


class DataSet(object):
    def __init__(self):
        pass

    @property
    def nsample(self):
        return self._nsample

    @property
    def ninput(self):
        return self._ninput

    @ninput.setter
    def ninput(self, ninput):
        self._ninput = ninput

    @property
    def nderivative(self):
        return self._nderivative

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def label(self):
        return self._label

    @property
    def dinput(self):
        return self._dinput

    @dinput.setter
    def dinput(self, dinput):
        self._dinput = dinput

    @property
    def dlabel(self):
        return self._dlabel


class FunctionData(DataSet):
    def __init__(self, name, type):
        if name == 'complex':
            input, label, dinput, dlabel = self._make_complex()
        elif name == 'LJ':
            input, label, dinput, dlabel = self._make_LJ()
        elif name == 'sin':
            input, label, dinput, dlabel = self._make_sin()

        np.random.seed(0)
        np.random.shuffle(input)
        np.random.seed(0)
        np.random.shuffle(label)
        np.random.seed(0)
        np.random.shuffle(dinput)
        np.random.seed(0)
        np.random.shuffle(dlabel)

        sep = self._nsample * 7 / 10
        if type == 'training':
            self._nsample = self._nsample * 7 / 10
            self._input = input[:sep]
            self._label = label[:sep]
            self._dinput = dinput[:sep]
            self._dlabel = dlabel[:sep]
        elif type == 'validation':
            self._nsample = self._nsample - self._nsample * 7 / 10
            self._input = input[sep:]
            self._label = label[sep:]
            self._dinput = dinput[sep:]
            self._dlabel = dlabel[sep:]

    def _make_complex(self):
        mesh = 10
        self._nsample = mesh**3
        self._ninput = 3
        self._nderivative = 3
        lin = np.linspace(0.1, 1.0, mesh)
        x, y, z = np.meshgrid(lin, lin, lin)
        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        input = np.c_[x, y, z]
        label = (x**2 + np.sin(y) + 3.*np.exp(z) - np.log(x*y)/2 - y/z).reshape(self._nsample, 1)
        dinput = np.identity(3)[None, :, :].repeat(self._nsample, axis=0)
        dlabel = np.c_[2**x - 1/(2*x),
                       np.cos(y) - 1/(2*y) - 1/z,
                       3.*np.exp(z) + y/z**2].reshape(self._nsample, 3, 1)
        return input, label, dinput, dlabel

    def _make_LJ(self):
        self._nsample = 1000
        self._ninput = 1
        self._nderivative = 1
        input = np.linspace(0.1, 1.0, self._nsample).reshape(self._nsample, 1)
        label = 0.001/input**4 - 0.009/input**3
        dinput = np.ones(self._nsample).reshape(self._nsample, 1, 1)
        dlabel = (0.027/input**4 - 0.004/input**5).reshape(self._nsample, 1, 1)
        return input, label, dinput, dlabel

    def _make_sin(self):
        self._nsample = 1000
        self._ninput = 1
        self._nderivative = 1
        input = np.linspace(-2*3.14, 2*3.14, self._nsample).reshape(self._nsample, 1)
        label = np.sin(input)
        dinput = np.ones(self._nsample).reshape(self._nsample, 1, 1)
        dlabel = np.cos(input).reshape(self._nsample, 1, 1)
        return input, label, dinput, dlabel


class AtomicStructureData(DataSet):
    def __init__(self, config, type):
        self._data_dir = path.join(file_.data_dir, config, type)
        with open(path.join(self._data_dir, '..', 'composition.dill')) as f:
            self._composition = dill.load(f)
        xyz_file = path.join(self._data_dir, 'structure.xyz')
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = AtomsReader(xyz_file)[0].n
        self._nforce = 3 * self._natom
        self._name = config + type
        self._ninput = 2 * len(hp.Rcs) + \
            2 * len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
            3 * len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
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

        self._make_label()
        self._make_input()

    @property
    def composition(self):
        return self._composition

    @property
    def natom(self):
        return self._natom

    @property
    def nderivative(self):
        return self._nforce

    @property
    def input(self):
        return self._Gs

    @input.setter
    def input(self, input):
        self._Gs = input

    @property
    def label(self):
        return self._Es

    @property
    def dinput(self):
        return self._dGs

    @dinput.setter
    def dinput(self, dinput):
        self._dGs = dinput

    @property
    def dlabel(self):
        return self._Fs

    def _make_label(self):
        EF_file = path.join(self._data_dir, 'Energy_Force.npz')
        if path.exists(EF_file):
            ndarray = np.load(EF_file)
            self._Es = ndarray['E']
            self._Fs = ndarray['F']
        else:
            self._Es = np.empty((self._nsample, 1))
            self._Fs = np.empty((self._nsample, self._nforce, 1))
            Es_send = np.array([data.cohesive_energy for data in self._atoms_objs]).reshape(-1, 1)
            Fs_send = np.array([data.force for data in self._atoms_objs]).reshape(-1, self._nforce, 1)
            mpi.comm.Allgatherv((Es_send, (self._n), MPI.DOUBLE),
                                (self._Es, (self._count, self._disps), MPI.DOUBLE))
            num = self._nforce
            mpi.comm.Allgatherv((Fs_send, (self._n*num), MPI.DOUBLE),
                                (self._Fs, (self._count*num, self._disps*num), MPI.DOUBLE))
            np.savez(EF_file, E=self._Es, F=self._Fs)

    def _make_input(self):
        Gs = np.empty((self._nsample, self._ninput, self._natom))
        dGs = np.empty((self._nsample, self._ninput, self._natom, self._nforce))
        n = 0

        # type 1
        for Rc in hp.Rcs:
            filename = path.join(self._data_dir, 'G1-{}.npz'.format(Rc))
            if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+2] = ndarray['G']
                dGs[:, n:n+2] = ndarray['dG']
            else:
                G = np.empty((self._nsample, 2, self._natom))
                dG = np.empty((self._nsample, 2, self._natom, self._nforce))
                G_send, dG_send = self._calc_G1(Rc)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                    (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                num = 2 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                    (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+2] = G
                dGs[:, n:n+2] = dG
            n += 2

        # type 2
        for Rc, eta, Rs in product(hp.Rcs, hp.etas, hp.Rss):
            filename = path.join(self._data_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
            if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+2] = ndarray['G']
                dGs[:, n:n+2] = ndarray['dG']
            else:
                G = np.empty((self._nsample, 2, self._natom))
                dG = np.empty((self._nsample, 2, self._natom, self._nforce))
                G_send, dG_send = self._calc_G2(Rc, eta, Rs)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                    (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                num = 2 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                    (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+2] = G
                dGs[:, n:n+2] = dG
            n += 2

        # type 4
        for Rc, eta, lam, zeta in product(hp.Rcs, hp.etas, hp.lams, hp.zetas):
            filename = path.join(self._data_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
            if path.exists(filename) and Gs[:, n:n+3].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+3] = ndarray['G']
                dGs[:, n:n+3] = ndarray['dG']
            else:
                G = np.empty((self._nsample, 3, self._natom))
                dG = np.empty((self._nsample, 3, self._natom, self._nforce))
                G_send, dG_send = self._calc_G4(Rc, eta, lam, zeta)
                num = 3 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                    (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                num = 3 * self._natom * self._nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                    (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+3] = G
                dGs[:, n:n+3] = dG
            n += 3

        self._Gs = Gs.transpose(0, 2, 1)
        self._dGs = dGs.transpose(0, 2, 3, 1)

    def _calc_G1(self, Rc):
        G = np.zeros((self._n, 2, self._natom))
        dG = np.zeros((self._n, 2, self._natom, self._nforce))
        for index, (R, fc, tanh, dR), _, _ in self._calc_geometry(['homo', 'hetero'], Rc):
            G[index] = np.sum(fc)
            dG[index] = - 3./Rc * np.dot((1. - tanh**2) * tanh**2, dR)
        return G, dG

    def _calc_G2(self, Rc, eta, Rs):
        G = np.zeros((self._n, 2, self._natom))
        dG = np.zeros((self._n, 2, self._natom, self._nforce))
        for index, (R, fc, tanh, dR), _, _ in self._calc_geometry(['homo', 'hetero'], Rc):
            gi = np.exp(- eta * (R - Rs)**2) * fc
            G[index] = np.sum(gi)
            dG[index] = np.dot(gi * (-2.*eta*(R-Rs) + 3./Rc*(tanh - 1./tanh)), dR)
        return G, dG

    def _calc_G4(self, Rc, eta, lam, zeta):
        G = np.zeros((self._n, 3, self._natom))
        dG = np.zeros((self._n, 3, self._natom, self._nforce))
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
            if self._name not in done:
                cache.clear()
                done.append(self._name)
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
        zeros_radial = (np.zeros(1), np.zeros(1), np.ones(1), np.zeros((1, self._nforce)))
        zeros_angular = (np.zeros((1, 1)), np.zeros((1, 1, self._nforce)))
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
                R = np.array(R)
                fc = np.tanh(1-R/Rc)**3
                tanh = np.tanh(1-R/Rc)
                cos = np.array(cos)

                symbol = self._composition['symbol'][k]  # symbol of the focused atom
                homo_indices = [index for index, n in enumerate(neighbours) if n in self._composition['index'][symbol]]
                n_homo = len(homo_indices)
                if n_homo == 0:
                    yield i, k, zeros_radial, (R, fc, tanh, dR), zeros_angular, (cos, dcos), \
                        (np.zeros((1, n_neighb)), np.zeros((1, n_neighb, self._nforce)))
                    continue
                elif n_homo == n_neighb:
                    yield i, k, (R, fc, tanh, dR), zeros_radial, (cos, dcos), zeros_angular, \
                        (np.zeros((n_neighb, 1)), np.zeros((n_neighb, 1, self._nforce)))
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
        dR = np.zeros((n_neighb, self._nforce))
        for l, n in product(xrange(n_neighb), xrange(self._nforce)):
            if n % self._natom == k:
                dR[l, n] = - r[l][n/self._natom] / R[l]
            elif n % self._natom == neighbours[l]:
                dR[l, n] = + r[l][n/self._natom] / R[l]
        return dR

    def _deriv_cosine(self, k, n_neighb, neighbours, r, R, cos):
        dcos = np.zeros((n_neighb, n_neighb, self._nforce))
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
    def __init__(self, mode, precond=None):
        self._mode = mode
        self._precond = PRECOND[precond]()
        self._config_type_file = path.join(file_.data_dir, 'config_type.dill')
        if mpi.rank == 0 and not path.exists(self._config_type_file):
            self._parse_xyzfile()
        mpi.comm.Barrier()

    def __iter__(self):
        with open(self._config_type_file, 'r') as f:
            config_type = dill.load(f)
        if self._mode == 'training':
            for type in file_.train_config:
                for config in filter(lambda config: match(type, config) or type == 'all', config_type):
                    training_data = AtomicStructureData(config, 'training')
                    self._precond.decompose(training_data)
                    validation_data = AtomicStructureData(config, 'validation')
                    self._precond.decompose(validation_data)
                    yield config, training_data, validation_data
        elif self._mode == 'test':
            for type in file_.train_config:
                for config in filter(lambda config: match(type, config) or type == 'all', config_type):
                    test_data = AtomicStructureData(config, 'test')
                    self._precond.decompose(test_data)
                    yield config, test_data

    def save(self, save_dir):
        with open(path.join(save_dir, 'preconditioning.dill'), 'w') as f:
            dill.dump(self._precond, f)

    def load(self, save_dir):
        with open(path.join(save_dir, 'preconditioning.dill'), 'r') as f:
            self._precond = dill.load(f)

    def _parse_xyzfile(self):
        print 'config_type.dill is not found.\nLoad all data from xyz file ...'
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

        print 'Separate all dataset to training & validation & test dataset ...'
        for config in config_type:
            composition = {'number': defaultdict(lambda: 0), 'index': defaultdict(set), 'symbol': []}
            for i, atom in enumerate(alldataset[config][0]):
                composition['number'][atom.symbol] += 1
                composition['index'][atom.symbol].add(i)
                composition['symbol'].append(atom.symbol)

            data_dir = path.join(file_.data_dir, config)
            shuffle(alldataset[config])
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
            training_writer.close()
            validation_writer.close()
            test_writer.close()
            with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                dill.dump(composition, f)


if __name__ == '__main__':
    generator = DataGenerator('training')
    for config, _ in generator:
        if mpi.rank == 0:
            print config
