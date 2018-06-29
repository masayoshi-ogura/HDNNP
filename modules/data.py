# -*- coding: utf-8 -*-

# define variables
from config import file_
from config import mpi

# import python modules
from os import path
from re import match
from collections import defaultdict
from itertools import product
import dill
import numpy as np
from mpi4py import MPI
from chainer.datasets import split_dataset_random
from chainer.datasets import get_cross_validation_datasets_random
from quippy import Atoms
from quippy import AtomsReader
from quippy import AtomsList
from quippy import AtomsWriter
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToCm

from .util import pprint, mkdir
from .util import DictAsAttributes


def memorize_generator(f):
    cache = defaultdict(list)
    identifier = ['']

    def helper(self, at, Rc):
        if identifier[0] != str(id(self)) + str(id(at)):
            cache.clear()
            identifier[0] = str(id(self)) + str(id(at))

        if Rc not in cache:
            for ret in f(self, at, Rc):
                cache[Rc].append(ret)
                yield cache[Rc][-1]
        else:
            for ret in cache[Rc]:
                yield ret

    return helper


def scatter(dataset):
    sub_nsample = (dataset.nsample + mpi.size - 1) / mpi.size
    slice_ = np.array([(dataset.nsample * i / mpi.size, dataset.nsample * i / mpi.size + sub_nsample)
                       for i in range(mpi.size)], dtype=np.int32)
    for attr in ['input', 'dinput', 'label', 'dlabel']:
        if mpi.rank != 0:
            shape = mpi.comm.bcast(None, root=0)
            recv = np.empty((sub_nsample,) + shape, dtype=np.float32)
            mpi.comm.Recv(recv, source=0)
            setattr(dataset, attr, recv)
        else:
            data = getattr(dataset, attr)
            mpi.comm.bcast(data.shape[1:], root=0)
            setattr(dataset, attr, data[:sub_nsample])
            for i in xrange(1, mpi.size):
                mpi.comm.Send(data[slice_[i][0]:slice_[i][1]], dest=i)


def bcast(dataset):
    for attr in ['input', 'dinput', 'label', 'dlabel']:
        if mpi.rank != 0:
            shape = mpi.comm.bcast(None, root=0)
            data = np.empty(shape, dtype=np.float32)
            mpi.comm.Bcast(data, root=0)
            setattr(dataset, attr, data)
        else:
            data = getattr(dataset, attr)
            mpi.comm.bcast(data.shape, root=0)
            mpi.comm.Bcast(data, root=0)


class AtomicStructureDataset(object):
    def __init__(self, hp, filename, file_format, *args, **kwargs):
        assert file_format in ['xyz', 'POSCAR']
        self._hp = hp
        if file_format == 'xyz':
            self._load_xyz(filename)
        elif file_format == 'POSCAR':
            self._load_poscar(filename, *args, **kwargs)

    def __getitem__(self, index):
        batches = [self._Gs[index], self._dGs[index], self._Es[index], self._Fs[index]]
        if isinstance(index, slice):
            return [tuple([batch[i] for batch in batches]) for i in range(len(batches))]
        else:
            return tuple(batches)

    def __len__(self):
        return self._Gs.shape[0]

    @property
    def phonopy(self):
        return self._phonopy

    @property
    def composition(self):
        return self._composition

    @property
    def config(self):
        return self._config

    @property
    def nsample(self):
        return self._nsample

    @property
    def ninput(self):
        return self._Gs.shape[-1]

    @property
    def input(self):
        return self._Gs

    @input.setter
    def input(self, input_):
        self._Gs = input_

    @property
    def dinput(self):
        return self._dGs

    @dinput.setter
    def dinput(self, dinput):
        self._dGs = dinput

    @property
    def label(self):
        return self._Es

    @label.setter
    def label(self, label):
        self._Es = label

    @property
    def dlabel(self):
        return self._Fs

    @dlabel.setter
    def dlabel(self, dlabel):
        self._Fs = dlabel

    def _load_xyz(self, xyz_file):
        self._data_dir = path.dirname(xyz_file)
        self._config = path.basename(self._data_dir)
        with open(path.join(self._data_dir, 'composition.dill')) as f:
            self._composition = dill.load(f)
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = len(self._composition.element)
        self._count = np.array([(self._nsample + i) / mpi.size for i in range(mpi.size)[::-1]], dtype=np.int32)
        self._atoms = AtomsReader(xyz_file)[self._count[:mpi.rank].sum(): self._count[:mpi.rank + 1].sum()]

        self._make_input(save=True)
        self._make_label(save=True)
        del self._hp, self._data_dir, self._natom, self._atoms, self._count

    def _load_poscar(self, poscar, dimension=None, distance=0.03, save=True, scale=1.0):
        if dimension is None:
            dimension = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert self._hp.mode in ['optimize', 'phonon']

        self._data_dir = path.dirname(poscar)
        self._config = path.basename(self._data_dir)
        unitcell, = AtomsList(poscar, format='POSCAR')
        atoms = []
        if self._hp.mode == 'optimize':
            for k in np.linspace(0.9, 1.1, 201):
                supercell = unitcell.copy()
                supercell.set_lattice(unitcell.lattice * k, scale_positions=True)
                atoms.append(supercell)
        elif self._hp.mode == 'phonon':
            unitcell.set_lattice(unitcell.lattice * scale, scale_positions=True)  # scaling
            unitcell = PhonopyAtoms(symbols=unitcell.get_chemical_symbols(),
                                    positions=unitcell.positions,
                                    numbers=unitcell.numbers,
                                    masses=unitcell.get_masses(),
                                    scaled_positions=unitcell.get_scaled_positions(),
                                    cell=unitcell.cell)
            phonon = Phonopy(unitcell,
                             dimension,
                             # primitive_matrix=primitive_matrix,
                             factor=VaspToCm)
            phonon.generate_displacements(distance=distance)
            supercells = phonon.get_supercells_with_displacements()
            self._phonopy = phonon
            for phonopy_at in supercells:
                at = Atoms(cell=phonopy_at.cell,
                           positions=phonopy_at.get_positions(),
                           numbers=phonopy_at.numbers,
                           masses=phonopy_at.masses)
                at.set_chemical_symbols(phonopy_at.get_chemical_symbols())
                atoms.append(at)

        symbols = atoms[0].get_chemical_symbols()
        composition = {'index': {k: set([i for i, s in enumerate(symbols) if s == k]) for k in set(symbols)},
                       'element': symbols}
        self._composition = DictAsAttributes(composition)
        self._nsample = len(atoms)
        self._natom = len(atoms[0])

        self._count = np.array([(self._nsample + i) / mpi.size for i in range(mpi.size)[::-1]], dtype=np.int32)
        self._atoms = atoms[self._count[:mpi.rank].sum(): self._count[:mpi.rank + 1].sum()]
        self._make_input(save=save)
        del self._hp, self._data_dir, self._natom, self._atoms, self._count

    def _make_label(self, save):
        EF_file = path.join(self._data_dir, 'Energy_Force.npz')

        # non-root process
        if mpi.rank != 0 and not path.exists(EF_file):
            Es_send = np.array([data.cohesive_energy for data in self._atoms]).reshape(-1, 1)
            Fs_send = np.array([data.force.T for data in self._atoms]).reshape(-1, self._natom, 3)
            mpi.comm.Gatherv(Es_send, None, 0)
            mpi.comm.Gatherv(Fs_send, None, 0)

        # root node process
        else:
            if path.exists(EF_file):
                ndarray = np.load(EF_file)
                Es = ndarray['E']
                Fs = ndarray['F']
            else:
                pprint('making {} ... '.format(EF_file), end='', flush=True)
                Es = np.empty((self._nsample, 1))
                Fs = np.empty((self._nsample, self._natom, 3))
                Es_send = np.array([data.cohesive_energy for data in self._atoms]).reshape(-1, 1)
                Fs_send = np.array([data.force.T for data in self._atoms]).reshape(-1, self._natom, 3)
                mpi.comm.Gatherv(Es_send, (Es, (self._count, None), MPI.DOUBLE), root=0)
                mpi.comm.Gatherv(Fs_send, (Fs, (self._count * self._natom * 3, None), MPI.DOUBLE), root=0)
                if save:
                    np.savez(EF_file, E=Es, F=Fs)
                pprint('done')
            self._Es = Es.astype(np.float32)
            self._Fs = Fs.astype(np.float32)

    def _make_input(self, save):
        if mpi.rank != 0:
            new_keys = mpi.comm.bcast(None, root=0)
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                mpi.comm.Gatherv(G_send, None, 0)
                mpi.comm.Gatherv(dG_send, None, 0)

        else:
            Gs = {}
            dGs = {}
            SF_file = path.join(self._data_dir, 'Symmetry_Function.npz')

            if save and path.exists(SF_file):
                ndarray = np.load(SF_file)
                if 'nsample' in ndarray and ndarray['nsample'] == self._nsample:
                    existing_keys = set([path.dirname(key) for key in ndarray.iterkeys() if key.endswith('G')])
                    new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys(existing_keys)
                else:
                    pprint("# of samples (or atoms) between Symmetry_Function.npz and structure.xyz don't match.\n"
                           "re-calculate from structure.xyz.")
                    new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys()
                mpi.comm.bcast(new_keys, root=0)
                if new_keys:
                    pprint('making {} ... '.format(SF_file), end='', flush=True)
                for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                    G = np.empty((self._nsample,) + G_send.shape[1:])
                    dG = np.empty((self._nsample,) + dG_send.shape[1:])
                    mpi.comm.Gatherv(G_send, (G, (self._count * G[0].size, None), MPI.DOUBLE), root=0)
                    mpi.comm.Gatherv(dG_send, (dG, (self._count * dG[0].size, None), MPI.DOUBLE), root=0)
                    Gs[path.join(key, 'G')] = G
                    dGs[path.join(key, 'dG')] = dG
                for key in re_used_keys:
                    G_key = path.join(key, 'G')
                    dG_key = path.join(key, 'dG')
                    Gs[G_key] = ndarray[G_key]
                    dGs[dG_key] = ndarray[dG_key]
                if new_keys:
                    pprint('done')
                self._Gs = np.concatenate([v for k, v in sorted(Gs.iteritems())], axis=2).astype(np.float32)
                self._dGs = np.concatenate([v for k, v in sorted(dGs.iteritems())], axis=2).astype(np.float32)
                for key in no_used_keys:
                    G_key = path.join(key, 'G')
                    dG_key = path.join(key, 'dG')
                    Gs[G_key] = ndarray[G_key]
                    dGs[dG_key] = ndarray[dG_key]
                np.savez(SF_file, nsample=self._nsample, **{k: v for dic in [Gs, dGs] for k, v in dic.iteritems()})

            else:
                pprint('1)save flag off, or 2){} not found.\n'
                       'calculate symmetry functions from scratch in this time.')
                keys, _, _ = self._check_uncalculated_keys()
                mpi.comm.bcast(keys, root=0)
                pprint('making {} ... '.format(SF_file), end='', flush=True)
                for key, G_send, dG_send in self._calculate_symmetry_function(keys):
                    G = np.empty((self._nsample,) + G_send.shape[1:])
                    dG = np.empty((self._nsample,) + dG_send.shape[1:])
                    mpi.comm.Gatherv(G_send, (G, (self._count * G[0].size, None), MPI.DOUBLE), root=0)
                    mpi.comm.Gatherv(dG_send, (dG, (self._count * dG[0].size, None), MPI.DOUBLE), root=0)
                    Gs[path.join(key, 'G')] = G
                    dGs[path.join(key, 'dG')] = dG
                self._Gs = np.concatenate([v for k, v in sorted(Gs.iteritems())], axis=2).astype(np.float32)
                self._dGs = np.concatenate([v for k, v in sorted(dGs.iteritems())], axis=2).astype(np.float32)
                if save:
                    np.savez(SF_file, nsample=self._nsample, **{k: v for dic in [Gs, dGs] for k, v in dic.iteritems()})

    def _check_uncalculated_keys(self, existing_keys=set()):
        required_keys = set()
        for Rc in self._hp.Rc:
            key = path.join(*map(str, ['type1', Rc]))
            required_keys.add(key)
        for Rc, eta, Rs in product(self._hp.Rc, self._hp.eta, self._hp.Rs):
            key = path.join(*map(str, ['type2', Rc, eta, Rs]))
            required_keys.add(key)
        for Rc, eta, lambda_, zeta in product(self._hp.Rc, self._hp.eta, self._hp.lambda_, self._hp.zeta):
            key = path.join(*map(str, ['type4', Rc, eta, lambda_, zeta]))
            required_keys.add(key)
        new_keys = sorted(required_keys - existing_keys)
        re_used_keys = sorted(required_keys & existing_keys)
        no_used_keys = sorted(existing_keys - required_keys)
        if new_keys:
            pprint('uncalculated symmetry function parameters are as follows:')
            pprint('\n'.join(new_keys))
        return new_keys, re_used_keys, no_used_keys

    def _calculate_symmetry_function(self, keys):
        Gs = defaultdict(list)
        dGs = defaultdict(list)
        for at in self._atoms:
            for key in keys:
                params = key.split('/')
                G, dG = getattr(self, '_' + params[0])(at, *map(float, params[1:]))
                Gs[key].append(G)
                dGs[key].append(dG)
        for key in keys:
            yield key, np.stack(Gs[key]), np.stack(dGs[key])

    def _type1(self, at, Rc):
        G = np.zeros((self._natom, 2))
        dG = np.zeros((self._natom, 2, self._natom, 3))
        for i, neighbour, homo_all, hetero_all, R, tanh, dR, _, _ in self._neighbour(at, Rc):
            g = tanh ** 3
            dg = -3. / Rc * ((1. - tanh ** 2) * tanh ** 2)[:, None] * dR
            # G
            G[i, 0] = g[homo_all].sum()
            G[i, 1] = g[hetero_all].sum()
            # dG
            for j, (homo, indices) in enumerate(neighbour):
                if homo:
                    dG[i, 0, j] = dg.take(indices, 0).sum(0)
                else:
                    dG[i, 1, j] = dg.take(indices, 0).sum(0)
        return G, dG

    def _type2(self, at, Rc, eta, Rs):
        G = np.zeros((self._natom, 2))
        dG = np.zeros((self._natom, 2, self._natom, 3))
        for i, neighbour, homo_all, hetero_all, R, tanh, dR, _, _ in self._neighbour(at, Rc):
            g = np.exp(- eta * (R - Rs) ** 2) * tanh ** 3
            dg = (np.exp(- eta * (R - Rs) ** 2) * tanh ** 2 * (
                    -2. * eta * (R - Rs) * tanh + 3. / Rc * (tanh ** 2 - 1.0)))[:, None] * dR
            # G
            G[i, 0] = g[homo_all].sum()
            G[i, 1] = g[hetero_all].sum()
            # dG
            for j, (homo, indices) in enumerate(neighbour):
                if homo:
                    dG[i, 0, j] = dg.take(indices, 0).sum(0)
                else:
                    dG[i, 1, j] = dg.take(indices, 0).sum(0)
        return G, dG

    def _type4(self, at, Rc, eta, lambda_, zeta):
        G = np.zeros((self._natom, 3))
        dG = np.zeros((self._natom, 3, self._natom, 3))
        for i, neighbour, homo_all, hetero_all, R, tanh, dR, cos, dcos in self._neighbour(at, Rc):
            ang = 1. + lambda_ * cos
            rad1 = np.exp(-eta * R ** 2) * tanh ** 3
            rad2 = np.exp(-eta * R ** 2) * tanh ** 2 * (-2. * eta * R * tanh + 3. / Rc * (tanh ** 2 - 1.0))
            ang[np.eye(len(R), dtype=bool)] = 0
            g = 2. ** (1 - zeta) * ang ** zeta * rad1[:, None] * rad1[None, :]
            dg_radial_part = 2. ** (1 - zeta) * ang[:, :, None] ** zeta * \
                             rad2[:, None, None] * rad1[None, :, None] * dR[:, None, :]
            dg_angular_part = zeta * lambda_ * 2. ** (1 - zeta) * ang[:, :, None] ** (zeta - 1) * \
                              rad1[:, None, None] * rad1[None, :, None] * dcos
            dg = dg_radial_part + dg_angular_part

            # G
            G[i, 0] = g.take(homo_all, 0).take(homo_all, 1).sum() / 2.0
            G[i, 1] = g.take(hetero_all, 0).take(hetero_all, 1).sum() / 2.0
            G[i, 2] = g.take(homo_all, 0).take(hetero_all, 1).sum()
            # dG
            for j, (homo, indices) in enumerate(neighbour):
                if homo:
                    dG[i, 0, j] = dg.take(indices, 0).take(homo_all, 1).sum((0, 1))
                    dG[i, 2, j] = dg.take(indices, 0).take(hetero_all, 1).sum((0, 1))
                else:
                    dG[i, 1, j] = dg.take(indices, 0).take(hetero_all, 1).sum((0, 1))
                    dG[i, 2, j] = dg.take(indices, 0).take(homo_all, 1).sum((0, 1))
        return G, dG

    @memorize_generator
    def _neighbour(self, at, Rc):
        at.set_cutoff(Rc)
        at.calc_connect()
        for i in xrange(self._natom):
            n_neighb = at.n_neighbours(i + 1)
            if n_neighb == 0:
                continue

            r = np.zeros((n_neighb, 3))
            element = self._composition.element[i]
            neighbour = [[j_prime in self._composition.index[element], []] for j_prime in xrange(self._natom)]
            homo_all, hetero_all = [], []
            for j, neighb in enumerate(at.connect[i + 1]):
                r[j] = neighb.diff
                neighbour[neighb.j - 1][1].append(j)
                if neighbour[neighb.j - 1][0]:
                    homo_all.append(j)
                else:
                    hetero_all.append(j)
            R = np.linalg.norm(r, axis=1)
            tanh = np.tanh(1 - R / Rc)
            dR = r / R[:, None]
            cos = dR.dot(dR.T)
            # cosine(j - i - k) differentiate w.r.t. "j"
            # dcos = - rj * cos / Rj**2 + rk / Rj / Rk
            dcos = - r[:, None, :] / R[:, None, None] ** 2 * cos[:, :, None] \
                   + r[None, :, :] / (R[:, None, None] * R[None, :, None])
            yield i, neighbour, homo_all, hetero_all, R, tanh, dR, cos, dcos


class DataGenerator(object):
    def __init__(self, hp, precond):
        self._hp = hp
        self._precond = precond
        self._data_dir = path.dirname(file_.xyz_file)
        self._config_type_file = path.join(self._data_dir, 'config_type.dill')
        if not path.exists(self._config_type_file):
            self._parse_xyzfile()

        with open(self._config_type_file, 'r') as f:
            config_type = dill.load(f)
        self._datasets = []
        self._elements = set()
        for required_cnf in file_.config:
            for config in filter(lambda cnf: match(required_cnf, cnf) or required_cnf == 'all', config_type):
                pprint('Construct dataset of configuration type: {}'.format(config))

                xyz_file = path.join(self._data_dir, config, 'structure.xyz')
                dataset = AtomicStructureDataset(self._hp, xyz_file, 'xyz')
                if mpi.rank == 0:
                    self._precond.decompose(dataset)

                if hp.mode == 'training':
                    scatter(dataset)
                elif hp.mode == 'cv':
                    bcast(dataset)

                self._datasets.append(dataset)
                self._elements.update(dataset.composition.element)
                pprint('')
                mpi.comm.Barrier()
        self._length = sum([d.nsample for d in self._datasets])

    def __iter__(self):
        """
        first iteration: training - validation set
        second iteration: config_type
        """
        if self._hp.mode == 'cv':
            splited = [[(train, val, d.config, d.composition)
                        for train, val in get_cross_validation_datasets_random(d, n_fold=self._hp.kfold)]
                       for d in self._datasets]
            for dataset in zip(*splited):
                yield dataset, self._elements
        elif self._hp.mode == 'training':
            splited = [split_dataset_random(d, len(d) * 9 / 10) + (d.config, d.composition) for d in self._datasets]
            yield splited, self._elements

    def __len__(self):
        return self._length

    def _parse_xyzfile(self):
        if mpi.rank == 0:
            pprint('config_type.dill is not found.\nparsing {} ... '.format(file_.xyz_file), end='', flush=True)
            config_type = set()
            all_dataset = defaultdict(list)
            for data in AtomsReader(file_.xyz_file):
                config = data.config_type
                config_type.add(config)
                all_dataset[config].append(data)
            with open(self._config_type_file, 'w') as f:
                dill.dump(config_type, f)

            for config in config_type:
                composition = {'index': defaultdict(set), 'element': []}
                for i, atom in enumerate(all_dataset[config][0]):
                    composition['index'][atom.symbol].add(i)
                    composition['element'].append(atom.symbol)
                composition = DictAsAttributes(composition)

                data_dir = path.join(self._data_dir, config)
                mkdir(data_dir)
                writer = AtomsWriter(path.join(data_dir, 'structure.xyz'))
                for data in all_dataset[config]:
                    if data.cohesive_energy < 0.0:
                        writer.write(data)
                writer.close()
                with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                    dill.dump(composition, f)
            pprint('done')

        mpi.comm.Barrier()
