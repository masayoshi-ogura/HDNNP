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
import dill
import numpy as np
from mpi4py import MPI
try:
    from quippy import AtomsReader
    from quippy import AtomsList
    from quippy import AtomsWriter
    from quippy import farray
    from quippy import fzeros
    from quippy import frange
except ImportError:
    print 'Warning: can\'t import quippy.'


class DataSet(object):
    def __init__(self, config, type):
        self._data_dir = path.join(file_.data_dir, config, type)
        with open(path.join(self._data_dir, '..', 'composition.dill')) as f:
            self._composition = dill.load(f)
        self._atoms_objs = AtomsList(path.join(self._data_dir, 'structure.xyz'), start=mpi.rank, step=mpi.size)
        self._nsample = len(AtomsReader(path.join(self._data_dir, 'structure.xyz')))
        self._natom = self._atoms_objs[0].n
        self._nforce = 3 * self._natom
        self._name = config + type
        self._ninput = 2 * len(hp.Rcs) + \
            2 * len(hp.Rcs)*len(hp.etas)*len(hp.Rss) + \
            3 * len(hp.Rcs)*len(hp.etas)*len(hp.lams)*len(hp.zetas)
        quo = self.nsample / mpi.size
        rem = self.nsample % mpi.size
        self._count = np.array([quo+1 if i < rem else quo
                                for i in range(mpi.size)], dtype=np.int8)
        self._disps = np.array([np.sum(self._count[:i])
                                for i in range(mpi.size)], dtype=np.int8)
        self._n = self._count[mpi.rank]  # the number of allocated samples in this node

        self._make_label()
        self._make_input()

    @property
    def composition(self):
        return self._composition

    @property
    def nsample(self):
        return self._nsample

    @property
    def natom(self):
        return self._natom

    @property
    def nforce(self):
        return self._nforce

    @property
    def ninput(self):
        return self._ninput

    @property
    def input(self):
        return self._Gs

    @property
    def label(self):
        return self._Es

    @property
    def dinput(self):
        return self._dGs

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
            self._Es = np.empty((self.nsample, 1))
            self._Fs = np.empty((self.nsample, self.nforce, 1))
            Es_send = np.array([data.cohesive_energy for data in self._atoms_objs]).reshape(-1, 1)
            Fs_send = np.array([data.force for data in self._atoms_objs]).reshape(-1, self.nforce, 1)
            mpi.comm.Allgatherv((Es_send, (self._n), MPI.DOUBLE),
                                (self._Es, (self._count, self._disps), MPI.DOUBLE))
            num = self.nforce
            mpi.comm.Allgatherv((Fs_send, (self._n*num), MPI.DOUBLE),
                                (self._Fs, (self._count*num, self._disps*num), MPI.DOUBLE))
            np.savez(EF_file, E=self._Es, F=self._Fs)

    def _make_input(self):
        Gs = np.empty((self.nsample, self.ninput, self.natom))
        dGs = np.empty((self.nsample, self.ninput, self.natom, self.nforce))

        n = 0
        for Rc in hp.Rcs:
            filename = path.join(self._data_dir, 'G1-{}.npz'.format(Rc))
            if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                ndarray = np.load(filename)
                Gs[:, n:n+2] = ndarray['G']
                dGs[:, n:n+2] = ndarray['dG']
            else:
                G = np.empty((self.nsample, 2, self.natom))
                dG = np.empty((self.nsample, 2, self.natom, self.nforce))
                G_send, dG_send = self._calc_G1(Rc)
                num = 2 * self.natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                    (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                num = 2 * self.natom * self.nforce
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                    (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                np.savez(filename, G=G, dG=dG)
                Gs[:, n:n+2] = G
                dGs[:, n:n+2] = dG
            n += 2

            for eta in hp.etas:
                for Rs in hp.Rss:
                    filename = path.join(self._data_dir, 'G2-{}-{}-{}.npz'.format(Rc, eta, Rs))
                    if path.exists(filename) and Gs[:, n:n+2].shape == np.load(filename)['G'].shape:
                        ndarray = np.load(filename)
                        Gs[:, n:n+2] = ndarray['G']
                        dGs[:, n:n+2] = ndarray['dG']
                    else:
                        G = np.empty((self.nsample, 2, self.natom))
                        dG = np.empty((self.nsample, 2, self.natom, self.nforce))
                        G_send, dG_send = self._calc_G2(Rc, eta, Rs)
                        num = 2 * self.natom
                        mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                            (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                        num = 2 * self.natom * self.nforce
                        mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                            (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                        np.savez(filename, G=G, dG=dG)
                        Gs[:, n:n+2] = G
                        dGs[:, n:n+2] = dG
                    n += 2

                for lam in hp.lams:
                    for zeta in hp.zetas:
                        filename = path.join(self._data_dir, 'G4-{}-{}-{}-{}.npz'.format(Rc, eta, lam, zeta))
                        if path.exists(filename) and Gs[:, n:n+3].shape == np.load(filename)['G'].shape:
                            ndarray = np.load(filename)
                            Gs[:, n:n+3] = ndarray['G']
                            dGs[:, n:n+3] = ndarray['dG']
                        else:
                            G = np.empty((self.nsample, 3, self.natom))
                            dG = np.empty((self.nsample, 3, self.natom, self.nforce))
                            G_send, dG_send = self._calc_G4(Rc, eta, lam, zeta)
                            num = 3 * self.natom
                            mpi.comm.Allgatherv((G_send, (self._n*num), MPI.DOUBLE),
                                                (G, (self._count*num, self._disps*num), MPI.DOUBLE))
                            num = 3 * self.natom * self.nforce
                            mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.DOUBLE),
                                                (dG, (self._count*num, self._disps*num), MPI.DOUBLE))
                            np.savez(filename, G=G, dG=dG)
                            Gs[:, n:n+3] = G
                            dGs[:, n:n+3] = dG
                        n += 3
        self._Gs = Gs.transpose(1, 0, 2)
        self._dGs = dGs.transpose(1, 0, 2, 3)

    def _calc_G1(self, Rc):
        if len(self.composition['number']) == 2:
            G = np.empty((self._n, 2, self.natom))
            dG = np.empty((self._n, 2, self.natom, self.nforce))
            for i, con in enumerate(['homo', 'hetero']):
                for j, atoms in enumerate(self._atoms_objs):
                    for k, radial, _, _ in self._neighbour(con, j, Rc, atoms):
                        R, fc, tanh, dR = radial
                        G[j, i, k] = np.sum(fc)
                        dG[j, i, k] = - 3/Rc * np.dot((1 - tanh**2) * tanh**2, dR)
        else:
            G = np.zeros((self._n, 2, self.natom))
            dG = np.zeros((self._n, 2, self.natom, self.nforce))
            for j, atoms in enumerate(self._atoms_objs):
                for k, radial, _, _ in self._neighbour('homo', j, Rc, atoms):
                    R, fc, tanh, dR = radial
                    G[j, 0, k] = np.sum(fc)
                    dG[j, 0, k] = -3./Rc * np.dot((1 - tanh**2) * tanh**2, dR)
        return G, dG

    def _calc_G2(self, Rc, eta, Rs):
        if len(self.composition['number']) == 2:
            G = np.empty((self._n, 2, self.natom))
            dG = np.empty((self._n, 2, self.natom, self.nforce))
            for i, con in enumerate(['homo', 'hetero']):
                for j, atoms in enumerate(self._atoms_objs):
                    for k, radial, _, _ in self._neighbour(con, j, Rc, atoms):
                        R, fc, tanh, dR = radial
                        gi = np.exp(- eta * (R - Rs) ** 2) * fc
                        G[j, i, k] = np.sum(gi)
                        dG[j, i, k] = np.dot(gi * (-2*eta*(R-Rs) + 3/Rc*(tanh - 1/tanh)), dR)
        else:
            G = np.zeros((self._n, 2, self.natom))
            dG = np.zeros((self._n, 2, self.natom, self.nforce))
            for j, atoms in enumerate(self._atoms_objs):
                for k, radial, _, _ in self._neighbour('homo', j, Rc, atoms):
                    R, fc, tanh, dR = radial
                    gi = np.exp(- eta * (R - Rs) ** 2) * fc
                    G[j, 0, k] = np.sum(gi)
                    dG[j, 0, k] = np.dot(gi * (-2*eta*(R-Rs) + 3/Rc*(tanh - 1/tanh)), dR)
        return G, dG

    def _calc_G4(self, Rc, eta, lam, zeta):
        if len(self.composition['number']) == 2:
            G = np.empty((self._n, 3, self.natom))
            dG = np.empty((self._n, 3, self.natom, self.nforce))
            for i, con in enumerate(['homo', 'hetero']):
                for j, atoms in enumerate(self._atoms_objs):
                    for k, radial1, radial2, angular in self._neighbour(con, j, Rc, atoms):
                        R1, fc1, tanh1, dR1 = radial1
                        R2, fc2, tanh2, dR2 = radial2
                        cos, dcos = angular
                        ang = 1 + lam * cos
                        ang[np.identity(len(R1), dtype=bool)] = 0.0
                        common = 2**(-zeta) * ang**(zeta-1) \
                            * (np.exp(-eta * R1**2) * fc1)[:, None] \
                            * (np.exp(-eta * R2**2) * fc2)[None, :]
                        G[j, i, k] = np.tensordot(common, ang)
                        dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2*eta*R1 + 3/Rc*(tanh1 - 1/tanh1)), dR1)
                        dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2*eta*R2 + 3/Rc*(tanh2 - 1/tanh2)), dR2)
                        dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
                        dG[j, i, k] = dgi_radial1 + dgi_radial2 + dgi_angular
            for j, atoms in enumerate(self._atoms_objs):
                for k, radial1, radial2, angular in self._neighbour('mix', j, Rc, atoms):
                    R1, fc1, tanh1, dR1 = radial1
                    R2, fc2, tanh2, dR2 = radial2
                    cos, dcos = angular
                    ang = 1 + lam * cos
                    common = 2**(1-zeta) * ang**(zeta-1) \
                        * (np.exp(-eta * R1**2) * fc1)[:, None] \
                        * (np.exp(-eta * R2**2) * fc2)[None, :]
                    G[j, 2, k] = np.tensordot(common, ang)
                    dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2*eta*R1 + 3/Rc*(tanh1 - 1/tanh1)), dR1)
                    dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2*eta*R2 + 3/Rc*(tanh2 - 1/tanh2)), dR2)
                    dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
                    dG[j, 2, k] = dgi_radial1 + dgi_radial2 + dgi_angular
        else:
            G = np.zeros((self._n, 3, self.natom))
            dG = np.zeros((self._n, 3, self.natom, self.nforce))
            for j, atoms in enumerate(self._atoms_objs):
                for k, radial1, radial2, angular in self._neighbour('homo', j, Rc, atoms):
                    R1, fc1, tanh1, dR1 = radial1
                    R2, fc2, tanh2, dR2 = radial2
                    cos, dcos = angular
                    ang = 1 + lam * cos
                    ang[np.identity(len(R1), dtype=bool)] = 0.0
                    common = 2**(-zeta) * ang**(zeta-1) \
                        * (np.exp(-eta * R1**2) * fc1)[:, None] \
                        * (np.exp(-eta * R2**2) * fc2)[None, :]
                    G[j, 0, k] = np.tensordot(common, ang)
                    dgi_radial1 = np.dot(np.sum(common*ang, axis=1) * (-2*eta*R1 + 3/Rc*(tanh1 - 1/tanh1)), dR1)
                    dgi_radial2 = np.dot(np.sum(common*ang, axis=0) * (-2*eta*R2 + 3/Rc*(tanh2 - 1/tanh2)), dR2)
                    dgi_angular = zeta * lam * np.tensordot(common, dcos, ((0, 1), (0, 1)))
                    dG[j, 0, k] = dgi_radial1 + dgi_radial2 + dgi_angular
        return G, dG

    def memorize(f):
        cache = {}
        done = []

        def helper(self, j, k, Rc, *args):
            if self._name not in done:
                cache.clear()
                done.append(self._name)
            if (j, k, Rc) not in cache:
                cache[(j, k, Rc)] = f(self, j, k, Rc, *args)
            return cache[(j, k, Rc)]
        return helper

    # yieldと合わせるのよくなさそう(動作がどうなるかわからん)ので外す
    # @memorize
    def _neighbour(self, con, j, Rc, atoms):
        atoms.set_cutoff(Rc)
        atoms.calc_connect()

        for k in range(self.natom):
            neighbours = atoms.connect.get_neighbours(k+1)[0] - 1
            r, R = self._distance_ij(j, k, Rc, atoms)
            cos = self._cosine_ijk(j, k, Rc, atoms)
            dR = self._deriv_R(j, k, Rc, neighbours, r, R)
            dcos = self._deriv_cosine(j, k, Rc, neighbours, r, R, cos)
            R = np.array(R)
            fc = np.tanh(1-R/Rc)**3
            tanh = np.tanh(1-R/Rc)
            cos = np.array(cos)

            symbol = self.composition['symbol'][k]  # symbol of the focused atom
            index = self.composition['index'][symbol]  # index of the same species of the focused atom
            homo_neighbours = [i for i, n in enumerate(neighbours) if n in index]
            hetero_neighbours = [i for i, n in enumerate(neighbours) if n not in index]

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
                yield k, homo_radial, homo_radial, homo_angular
            elif con == 'hetero':
                yield k,  hetero_radial, hetero_radial, hetero_angular
            elif con == 'mix':
                yield k,  homo_radial, hetero_radial, mix_angular

    @memorize
    def _distance_ij(self, j, k, Rc, atoms):
        r, R = [], []
        for n in frange(atoms.n_neighbours(k+1)):
            dist = farray(0.0)
            diff = fzeros(3)
            atoms.neighbour(k+1, n, distance=dist, diff=diff)
            r.append(diff.tolist())
            R.append(dist.tolist())
        return r, R

    @memorize
    def _cosine_ijk(self, j, k, Rc, atoms):
        n_neighb = atoms.n_neighbours(k+1)
        cos = [[0.0 if l == m else atoms.cosine_neighbour(k+1, l+1, m+1)
                for m in range(n_neighb)]
               for l in range(n_neighb)]
        return cos

    @memorize
    def _deriv_R(self, j, k, Rc, neighbours, r, R):
        n_neighb = len(neighbours)
        dR = np.array([[- r[l][n/self.natom] / R[l] if n % self.natom == k
                        else + r[l][n/self.natom] / R[l] if n % self.natom == neighbours[l]
                        else 0.0
                        for n in range(self.nforce)]
                       for l in range(n_neighb)])
        return dR

    @memorize
    def _deriv_cosine(self, j, k, Rc, neighbours, r, R, cos):
        n_neighb = len(neighbours)
        dcos = np.array([[[(r[l][n/self.natom] / R[l]**2 + r[m][n/self.natom] / R[m]**2) * cos[l][m]
                           - (r[l][n/self.natom] + r[m][n/self.natom]) / (R[l] * R[m]) if n % self.natom == k
                           else - (r[l][n/self.natom] / R[l]**2) * cos[l][m]
                           + r[m][n/self.natom] / (R[l] * R[m]) if n % self.natom == neighbours[l]
                           else - (r[m][n/self.natom] / R[m]**2) * cos[l][m]
                           + r[l][n/self.natom] / (R[l] * R[m]) if n % self.natom == neighbours[m]
                           else 0.0
                           for n in range(self.nforce)]
                          for m in range(n_neighb)]
                         for l in range(n_neighb)])
        return dcos


class DataGenerator(object):
    def __init__(self, mode):
        self._mode = mode
        self._config_type_file = path.join(file_.data_dir, 'config_type.dill')
        if mpi.rank == 0 and not path.exists(self._config_type_file):
            self._parse_xyzfile()
        mpi.comm.Barrier()

    def __iter__(self):
        with open(self._config_type_file, 'r') as f:
            config_type = dill.load(f)
        if self._mode == 'training':
            for config in config_type:
                for type in file_.train_config:
                    if type == 'all' or match(type, config):
                        break
                else:
                    continue
                training_data = DataSet(config, 'training')
                validation_data = DataSet(config, 'validation')
                yield training_data, validation_data
        elif self._mode == 'test':
            for config in config_type:
                for type in file_.train_config:
                    if type == 'all' or match(type, config):
                        break
                else:
                    continue
                test_data = DataSet(config, 'test')
                yield test_data

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
            with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                dill.dump(composition, f)


if __name__ == '__main__':
    generator = DataGenerator('training')
    for _ in generator:
        pass
