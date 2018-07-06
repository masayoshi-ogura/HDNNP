# -*- coding: utf-8 -*-

# define variables
import settings as stg

# import python modules
from os import path
from re import match
from collections import defaultdict
from itertools import product
import dill
import copy
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import KFold
from quippy import Atoms
from quippy import AtomsReader
from quippy import AtomsList
from quippy import AtomsWriter
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToCm

from .util import pprint, mkdir
from .util import DictAsAttributes


def memorize(f):
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
            return [tuple([batch[i] for batch in batches]) for i in range(len(batches[0]))]
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

    def take(self, slc):
        assert isinstance(slc, slice) or isinstance(slc, np.ndarray)
        sliced = copy.copy(self)
        sliced.input = self.input[slc]
        sliced.dinput = self.dinput[slc]
        sliced.label = self.label[slc]
        sliced.dlabel = self.dlabel[slc]
        return sliced

    def _shuffle(self):
        state = np.random.get_state()
        np.random.shuffle(self._Gs)
        np.random.set_state(state)
        np.random.shuffle(self._dGs)
        np.random.set_state(state)
        np.random.shuffle(self._Es)
        np.random.set_state(state)
        np.random.shuffle(self._Fs)

    def _load_xyz(self, xyz_file):
        data_dir = path.dirname(xyz_file)
        self._config = path.basename(data_dir)
        with open(path.join(data_dir, 'composition.dill')) as f:
            self._composition = dill.load(f)
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = len(self._composition.element)
        count = np.array([(self._nsample + i) / stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = AtomsReader(xyz_file)[count[:stg.mpi.rank].sum(): count[:stg.mpi.rank + 1].sum()]

        self._make_input(data_dir, count, save=True)
        self._make_label(data_dir, count, save=True)
        self._shuffle()  # shuffle dataset at once
        del self._atoms

    def _load_poscar(self, poscar, dimension=None, distance=0.03, save=True, scale=1.0):
        if dimension is None:
            dimension = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert stg.args.mode in ['test', 'phonon', 'optimize']

        data_dir = path.dirname(poscar)
        self._config = path.basename(data_dir)
        unitcell, = AtomsList(poscar, format='POSCAR')
        atoms = []
        if stg.args.mode == 'test':
            atoms.append(unitcell)
        elif stg.args.mode == 'phonon':
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
                             factor=VaspToCm,
                             symprec=1.0e-4)
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
        elif stg.args.mode == 'optimize':
            for k in np.linspace(0.9, 1.1, 201):
                supercell = unitcell.copy()
                supercell.set_lattice(unitcell.lattice * k, scale_positions=True)
                atoms.append(supercell)

        symbols = atoms[0].get_chemical_symbols()
        composition = {'index': {k: set([i for i, s in enumerate(symbols) if s == k]) for k in set(symbols)},
                       'element': symbols}
        self._composition = DictAsAttributes(**composition)
        self._nsample = len(atoms)
        self._natom = len(atoms[0])

        count = np.array([(self._nsample + i) / stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = atoms[count[:stg.mpi.rank].sum(): count[:stg.mpi.rank + 1].sum()]
        self._make_input(data_dir, count, save=save)
        del self._atoms

    def _make_label(self, data_dir, count, save):
        EF_file = path.join(data_dir, 'Energy_Force.npz')

        # non-root process
        if stg.mpi.rank != 0 and not path.exists(EF_file):
            Es_send = np.array([data.cohesive_energy for data in self._atoms]).reshape(-1, 1)
            Fs_send = np.array([data.force.T for data in self._atoms]).reshape(-1, self._natom, 3)
            stg.mpi.comm.Gatherv(Es_send, None, 0)
            stg.mpi.comm.Gatherv(Fs_send, None, 0)
            self._Es = np.empty(0)
            self._Fs = np.empty(0)

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
                stg.mpi.comm.Gatherv(Es_send, (Es, (count, None), MPI.DOUBLE), root=0)
                stg.mpi.comm.Gatherv(Fs_send, (Fs, (count * self._natom * 3, None), MPI.DOUBLE), root=0)
                if save:
                    np.savez(EF_file, E=Es, F=Fs)
                pprint('done')
            self._Es = Es.astype(np.float32)
            self._Fs = Fs.astype(np.float32)

    def _make_input(self, data_dir, count, save):
        if stg.mpi.rank != 0:
            new_keys = stg.mpi.comm.bcast(None, root=0)
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                stg.mpi.comm.Gatherv(G_send, None, 0)
                stg.mpi.comm.Gatherv(dG_send, None, 0)
            self._Gs = np.empty(0)
            self._dGs = np.empty(0)

        else:
            Gs = {}
            dGs = {}
            SF_file = path.join(data_dir, 'Symmetry_Function.npz')

            if save and path.exists(SF_file):
                ndarray = np.load(SF_file)
                if 'nsample' in ndarray and ndarray['nsample'] == self._nsample:
                    existing_keys = set([path.dirname(key) for key in ndarray.iterkeys() if key.endswith('G')])
                    new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys(existing_keys)
                else:
                    pprint("# of samples (or atoms) between Symmetry_Function.npz and structure.xyz don't match.\n"
                           "re-calculate from structure.xyz.")
                    new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys()
                stg.mpi.comm.bcast(new_keys, root=0)
                if new_keys:
                    pprint('making {} ... '.format(SF_file), end='', flush=True)
                for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                    G = np.empty((self._nsample,) + G_send.shape[1:])
                    dG = np.empty((self._nsample,) + dG_send.shape[1:])
                    stg.mpi.comm.Gatherv(G_send, (G, (count * G[0].size, None), MPI.DOUBLE), root=0)
                    stg.mpi.comm.Gatherv(dG_send, (dG, (count * dG[0].size, None), MPI.DOUBLE), root=0)
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
                stg.mpi.comm.bcast(keys, root=0)
                pprint('making {} ... '.format(SF_file), end='', flush=True)
                for key, G_send, dG_send in self._calculate_symmetry_function(keys):
                    G = np.empty((self._nsample,) + G_send.shape[1:])
                    dG = np.empty((self._nsample,) + dG_send.shape[1:])
                    stg.mpi.comm.Gatherv(G_send, (G, (count * G[0].size, None), MPI.DOUBLE), root=0)
                    stg.mpi.comm.Gatherv(dG_send, (dG, (count * dG[0].size, None), MPI.DOUBLE), root=0)
                    Gs[path.join(key, 'G')] = G
                    dGs[path.join(key, 'dG')] = dG
                pprint('done')
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

    @memorize
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
    def __init__(self, preproc):
        self._preproc = preproc
        self._data_dir = path.dirname(stg.file.xyz_file)
        self._config_type_file = path.join(self._data_dir, 'config_type.dill')
        if not path.exists(self._config_type_file):
            self._parse_xyzfile()

        with open(self._config_type_file, 'r') as f:
            config_type = dill.load(f)
        self._datasets = []
        self._elements = set()
        for required_cnf in stg.file.config:
            for config in filter(lambda cnf: match(required_cnf, cnf) or required_cnf == 'all', config_type):
                pprint('Construct dataset of configuration type: {}'.format(config))

                xyz_file = path.join(self._data_dir, config, 'structure.xyz')
                dataset = AtomicStructureDataset(stg.sym_func, xyz_file, 'xyz')
                if stg.mpi.rank == 0:
                    self._preproc.decompose(dataset)

                self._datasets.append(dataset)
                self._elements.update(dataset.composition.element)
                pprint('')
                stg.mpi.comm.Barrier()
        self._length = sum([d.nsample for d in self._datasets])

    def __len__(self):
        return self._length

    @property
    def preproc(self):
        return self._preproc

    def holdout(self, ratio):
        if stg.mpi.rank != 0:
            return [(None, None, dataset.composition) for dataset in self._datasets], self._elements
        else:
            splited = []
            for dataset in self._datasets:
                train = dataset.take(slice(None, int(len(dataset) * ratio), None))
                test = dataset.take(slice(int(len(dataset) * ratio), None, None))
                splited.append((train, test, dataset.composition))
            return splited, self._elements

    def cross_validation(self, ratio, kfold):
        if stg.mpi.rank != 0:
            for i in range(kfold):
                yield [(None, None, dataset.composition) for dataset in self._datasets], self._elements
        else:
            kf = KFold(n_splits=kfold)
            splited = []
            for dataset in self._datasets:
                splited.append([])
                for train_idx, test_idx in kf.split(range(int(len(dataset) * ratio))):
                    train = dataset.take(train_idx)
                    test = dataset.take(test_idx)
                    splited[-1].append((train, test, dataset.composition))
            for dataset in zip(*splited):
                yield dataset, self._elements

    def _parse_xyzfile(self):
        if stg.mpi.rank == 0:
            pprint('config_type.dill is not found.\nparsing {} ... '.format(stg.file.xyz_file), end='', flush=True)
            config_type = set()
            all_dataset = defaultdict(list)
            for data in AtomsReader(stg.file.xyz_file):
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
                composition = DictAsAttributes(**composition)

                data_dir = path.join(self._data_dir, config)
                mkdir(data_dir)
                writer = AtomsWriter(path.join(data_dir, 'structure.xyz'))
                for data in all_dataset[config]:
                    writer.write(data)
                writer.close()
                with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                    dill.dump(composition, f)
            pprint('done')

        stg.mpi.comm.Barrier()
