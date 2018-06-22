# -*- coding: utf-8 -*-

# define variables
from config import file_
from config import mpi
import config


# import python modules
from os import path
from re import match
from collections import defaultdict
from itertools import product
import dill
import numpy as np
from mpi4py import MPI
from chainer.datasets import TupleDataset
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


class AtomicStructureDataset(TupleDataset):
    def __init__(self, hp, file, format, *args, **kwargs):
        self._hp = hp
        if format == 'xyz':
            self._load_xyz(file)
        elif format == 'POSCAR':
            self._load_poscar(file, *args, **kwargs)
        else:
            pprint('unknown file format\navailable file format: .xyz POSCAR')

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
    def input(self, input):
        self._Gs = input

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

    @property
    def _datasets(self):
        return self._Gs, self._dGs, self._Es, self._Fs

    def _load_xyz(self, xyz_file):
        self._data_dir = path.dirname(xyz_file)
        self._config = path.basename(self._data_dir)
        with open(path.join(self._data_dir, 'composition.dill')) as f:
            self._composition = dill.load(f)
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = len(self._composition.element)
        self._count = np.array([(self._nsample+i)/mpi.size for i in range(mpi.size)[::-1]], dtype=np.int32)
        self._atoms_objs = AtomsReader(xyz_file)[self._count[:mpi.rank].sum(): self._count[:mpi.rank+1].sum()]

        self._make_input(save=True)
        self._make_label(save=True)
        del self._hp, self._data_dir, self._natom, self._atoms_objs, self._count

    def _load_poscar(self, poscar, dimension=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], distance=0.03, save=True, scale=1.0):
        self._data_dir = path.dirname(poscar)
        self._config = path.basename(self._data_dir)
        unitcell, = AtomsList(poscar, format='POSCAR')
        if self._hp.mode == 'optimize':
            supercells = []
            for k in np.linspace(0.9, 1.1, 201):
                supercell = unitcell.copy()
                supercell.set_lattice(unitcell.lattice * k, scale_positions=True)
                supercells.append(supercell)
            atoms_objs = supercells
        elif self._hp.mode == 'phonon':
            unitcell.set_lattice(unitcell.lattice*scale, scale_positions=True)  # scaling
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
            atoms_objs = []
            for pa in supercells:
                atoms = Atoms(cell=pa.cell,
                              positions=pa.get_positions(),
                              numbers=pa.numbers,
                              masses=pa.masses)
                atoms.set_chemical_symbols(pa.get_chemical_symbols())
                atoms_objs.append(atoms)

        symbols = atoms_objs[0].get_chemical_symbols()
        composition = {'index': {k: set([i for i, s in enumerate(symbols) if s == k]) for k in set(symbols)},
                       'element': symbols}
        self._composition = DictAsAttributes(composition)
        self._nsample = len(supercells)
        self._natom = supercells[0].get_number_of_atoms()

        self._count = np.array([(self._nsample+i)/mpi.size for i in range(mpi.size)[::-1]], dtype=np.int32)
        self._atoms_objs = atoms_objs[self._count[:mpi.rank].sum(): self._count[:mpi.rank+1].sum()]
        self._make_input(save=save)
        del self._hp, self._data_dir, self._natom, self._atoms_objs, self._count

    def _make_label(self, save):
        #EF_file = path.join(self._data_dir, 'Energy_Force.npz')
        #born_file = path.join(self._data_dir, 'born.npy')
        Es = np.ones((self._nsample, 3))
        #Fs = np.load(born_file)
        Fs = np.load("./born_tensor.npy")
        #Es = np.load("./E.npy")
        print("Born effective charge")
        self._Es = Es.astype(np.float32)
        self._Fs = Fs.astype(np.float32)


        #non-root process
        #if mpi.rank != 0 and not path.exists(EF_file):
        #    Es_send = np.array([data.cohesive_energy for data in self._atoms_objs]).reshape(-1, 1)
        #    Fs_send = np.array([data.force.T for data in self._atoms_objs]).reshape(-1, self._natom, 3)
        #    mpi.comm.Gatherv(Es_send, None, 0)
        #    mpi.comm.Gatherv(Fs_send, None, 0)

        #root node process
        #else:
            #Es = np.ones((self._nsample, 1))
            #Fs = np.load(born_file)
            #print("Born effective charge")
        #    if path.exists(EF_file):
        #        ndarray = np.load(EF_file)
        #        Es = ndarray['E']
        #        Fs = ndarray['F']
        #    else:
                #Es = np.ones((self._nsample, 1))
                #Fs = np.load(born_file)
                #print("Born effective charge")
        #        pprint('making {} ... '.format(EF_file), end='', flush=True)
        #        Es = np.empty((self._nsample, 1))
        #        Fs = np.empty((self._nsample, self._natom, 3))
        #        Es_send = np.array([data.cohesive_energy for data in self._atoms_objs]).reshape(-1, 1)
        #        Fs_send = np.array([data.force.T for data in self._atoms_objs]).reshape(-1, self._natom, 3)
        #        mpi.comm.Gatherv(Es_send, (Es, (self._count, None), MPI.DOUBLE), root=0)
        #        mpi.comm.Gatherv(Fs_send, (Fs, (self._count*self._natom*3, None), MPI.DOUBLE), root=0)
        #        if save:
        #            np.savez(EF_file, E=Es, F=Fs)
        #        pprint('done')
        #    self._Es = Es.astype(np.float32)
        #    self._Fs = Fs.astype(np.float32)


    def _make_input(self, save):
        if mpi.rank != 0:
            new_keys = mpi.comm.bcast(None, root=0)
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                mpi.comm.Gatherv(G_send, None, 0)
                mpi.comm.Gatherv(dG_send, None, 0)

        else:
            Gs = {}
            dGs = {}

            existing_keys = set()
            SF_file = path.join(self._data_dir, 'Symmetry_Function.npz')
            if save and path.exists(SF_file):
                ndarray = np.load(SF_file)
                if 'nsample' in ndarray and ndarray['nsample'] == self._nsample:
                    existing_keys = set([path.dirname(key) for key in ndarray.iterkeys() if key.endswith('G')])
                else:
                    pprint('# of samples (or atoms) between Symmetry_Function.npz and structure.xyz doesn\'t match.\n'
                           're-calculate from structure.xyz.')
            for key in existing_keys:
                G_key = path.join(key, 'G')
                dG_key = path.join(key, 'dG')
                Gs[G_key] = ndarray[G_key]
                dGs[dG_key] = ndarray[dG_key]

            new_keys = self._check_uncalculated_keys(existing_keys)
            mpi.comm.bcast(new_keys, root=0)
            if new_keys:
                pprint('making {} ... '.format(SF_file), end='', flush=True)
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                G = np.empty((self._nsample,) + G_send.shape[1:])
                dG = np.empty((self._nsample,) + dG_send.shape[1:])
                mpi.comm.Gatherv(G_send, (G, (self._count*G[0].size, None), MPI.DOUBLE), root=0)
                mpi.comm.Gatherv(dG_send, (dG, (self._count*dG[0].size, None), MPI.DOUBLE), root=0)
                Gs[path.join(key, 'G')] = G
                dGs[path.join(key, 'dG')] = dG

            if save:
                np.savez(SF_file, nsample=self._nsample, **{k: v for dic in [Gs, dGs] for k, v in dic.iteritems()})
            if new_keys:
                pprint('done')

            self._Gs = np.concatenate([v for k, v in sorted(Gs.iteritems())], axis=2).astype(np.float32)
            self._dGs = np.concatenate([v for k, v in sorted(dGs.iteritems())], axis=2).astype(np.float32)

    def _check_uncalculated_keys(self, existing_keys):
        all_keys = set()
        for Rc in self._hp.Rc:
            key = path.join(*map(str, ['type1', Rc]))
            all_keys.add(key)
        for Rc, eta, Rs in product(self._hp.Rc, self._hp.eta, self._hp.Rs):
            key = path.join(*map(str, ['type2', Rc, eta, Rs]))
            all_keys.add(key)
        for Rc, eta, lambda_, zeta in product(self._hp.Rc, self._hp.eta, self._hp.lambda_, self._hp.zeta):
            key = path.join(*map(str, ['type4', Rc, eta, lambda_, zeta]))
            all_keys.add(key)
        new_keys = sorted(all_keys - existing_keys)
        if new_keys:
            pprint('uncalculated symmetry function parameters are as follows:')
            pprint('\n'.join(new_keys))
        return new_keys

    def _calculate_symmetry_function(self, keys):
        Gs = defaultdict(list)
        dGs = defaultdict(list)
        for at in self._atoms_objs:
            for key in keys:
                params = key.split('/')
                G, dG = getattr(self, '_'+params[0])(at, *map(float, params[1:]))
                Gs[key].append(G)
                dGs[key].append(dG)
        for key in keys:
            yield key, np.stack(Gs[key]), np.stack(dGs[key])

    def _type1(self, at, Rc):
        G = np.zeros((self._natom, 3))
        dG = np.zeros((self._natom, 3, self._natom, 3))
        for i, neighbour, species, R, tanh, dR, _, _ in self._neighbour(at, Rc):
            g = tanh**3
            dg = -3./Rc * ((1.-tanh**2)*tanh**2)[:, None] * dR
            # G
            G[i, 0] = g[species["Li"]].sum()
            G[i, 1] = g[species["P"]].sum()
            G[i, 2] = g[species["O"]].sum()
            # dG
            for j, (atomic_species, indices) in enumerate(neighbour):
                if atomic_species == "Li":
                    dG[i, 0, j] = dg.take(indices, 0).sum(0)
                elif atomic_species == "P":
                    dG[i, 1, j] = dg.take(indices, 0).sum(0)
                else:
                    dG[i, 2, j] = dg.take(indices, 0).sum(0)
        return G, dG

    def _type2(self, at, Rc, eta, Rs):
        G = np.zeros((self._natom, 3))
        dG = np.zeros((self._natom, 3, self._natom, 3))
        for i, neighbour, species, R, tanh, dR, _, _ in self._neighbour(at, Rc):
            g = np.exp(- eta * (R - Rs)**2) * tanh**3
            dg = (np.exp(- eta * (R - Rs)**2) * tanh**2 * (-2.*eta*(R-Rs)*tanh + 3./Rc*(tanh**2 - 1.0)))[:, None] * dR
            # G
            G[i, 0] = g[species["Li"]].sum()
            G[i, 1] = g[species["P"]].sum()
            G[i, 2] = g[species["O"]].sum()
            # dG
            for j, (atomic_species, indices) in enumerate(neighbour):
                if atomic_species == "Li":
                    dG[i, 0, j] = dg.take(indices, 0).sum(0)
                elif atomic_species == "P":
                    dG[i, 1, j] = dg.take(indices, 0).sum(0)
                else:
                    dG[i, 2, j] = dg.take(indices, 0).sum(0)
        return G, dG

    def _type4(self, at, Rc, eta, lambda_, zeta):
        G = np.zeros((self._natom, 6))
        dG = np.zeros((self._natom, 6, self._natom, 3))
        for i, neighbour, species, R, tanh, dR, cos, dcos in self._neighbour(at, Rc):
            ang = 1. + lambda_ * cos
            rad1 = np.exp(-eta * R**2) * tanh**3
            rad2 = np.exp(-eta * R**2) * tanh**2 * (-2.*eta*R*tanh + 3./Rc*(tanh**2 - 1.0))
            ang[np.eye(len(R), dtype=bool)] = 0
            g = 2.**(1-zeta) * ang**zeta * rad1[:, None] * rad1[None, :]
            dg_radial_part = 2.**(1-zeta) * ang[:, :, None]**zeta * rad2[:, None, None] * rad1[None, :, None] * dR[:, None, :]
            dg_angular_part = zeta * lambda_ * 2.**(1-zeta) * ang[:, :, None]**(zeta-1) * rad1[:, None, None] * rad1[None, :, None] * dcos
            dg = dg_radial_part + dg_angular_part

            # G
            G[i, 0] = g.take(species["Li"], 0).take(species["Li"], 1).sum() / 2.0
            G[i, 1] = g.take(species["P"], 0).take(species["P"], 1).sum() / 2.0
            G[i, 2] = g.take(species["O"], 0).take(species["O"], 1).sum() / 2.0
            G[i, 3] = g.take(species["Li"], 0).take(species["P"], 1).sum()
            G[i, 4] = g.take(species["Li"], 0).take(species["O"], 1).sum()
            G[i, 5] = g.take(species["P"], 0).take(species["O"], 1).sum()
            # dG
            for j, (atomic_species, indices) in enumerate(neighbour):
                if atomic_species == "Li":
                    dG[i, 0, j] = dg.take(indices, 0).take(species["Li"], 1).sum((0, 1))
                    dG[i, 3, j] = dg.take(indices, 0).take(species["P"], 1).sum((0, 1))
                    dG[i, 4, j] = dg.take(indices, 0).take(species["O"], 1).sum((0, 1))
                elif atomic_species == "P":
                    dG[i, 3, j] = dg.take(indices, 0).take(species["Li"], 1).sum((0, 1))
                    dG[i, 1, j] = dg.take(indices, 0).take(species["P"], 1).sum((0, 1))
                    dG[i, 5, j] = dg.take(indices, 0).take(species["O"], 1).sum((0, 1))
                else:
                    dG[i, 4, j] = dg.take(indices, 0).take(species["Li"], 1).sum((0, 1))
                    dG[i, 5, j] = dg.take(indices, 0).take(species["P"], 1).sum((0, 1))
                    dG[i, 2, j] = dg.take(indices, 0).take(species["O"], 1).sum((0, 1))
        return G, dG

    def memorize_generator(f):
        cache = defaultdict(list)
        identifier = [None]

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

    @memorize_generator
    def _neighbour(self, at, Rc):
        at.set_cutoff(Rc)
        at.calc_connect()
        for i in xrange(self._natom):
            n_neighb = at.n_neighbours(i+1)
            if n_neighb == 0:
                continue

            r = np.zeros((n_neighb, 3))
            element = self._composition.element[i]
            #neighbour = [(j_prime in self._composition.index[element], []) for j_prime in xrange(self._natom)]
            neighbour = [(self._composition.element[j_prime], []) for j_prime in xrange(self._natom)]
            #print(neighbour)
            species = {"Li":[],"P":[], "O":[]}
            #print(asdfgh)
            #homo_all, hetero_all = [], []
            for j, neighb in enumerate(at.connect[i+1]):
                r[j] = neighb.diff
                #print(neighb.diff)
                #print(neighb.j-1)
                neighbour[neighb.j-1][1].append(j)
                #print(neighbour[neighb.j-1][1].append(j))
                #print(neighbour[neighb.j-1][0])
                if neighbour[neighb.j-1][0] == "Li":
                    species["Li"].append(j)
                elif neighbour[neighb.j-1][0] == "P":
                    species["P"].append(j)
                else:
                    species["O"].append(j)


            #print(homo_all)
            #print(neighbour)
            #print(species)
            #print(asdfg)
            R = np.linalg.norm(r, axis=1)
            tanh = np.tanh(1-R/Rc)
            dR = r / R[:, None]
            cos = dR.dot(dR.T)
            # cosine(j - i - k) differentiate w.r.t. "j"
            # dcos = - rj * cos / Rj**2 + rk / Rj / Rk
            dcos = - r[:, None, :]/R[:, None, None]**2 * cos[:, :, None] \
                + r[None, :, :]/(R[:, None, None] * R[None, :, None])
            yield i, neighbour, species, R, tanh, dR, cos, dcos


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
        for type in file_.config:
            for config in filter(lambda config: match(type, config) or type == 'all', config_type):
                xyz_file = path.join(self._data_dir, config, 'structure.xyz')
                dataset = AtomicStructureDataset(self._hp, xyz_file, 'xyz')
                if mpi.rank == 0:
                    self._precond.decompose(dataset)

                if hp.mode == 'training':
                    self._scatter(dataset)
                elif hp.mode == 'cv':
                    self._bcast(dataset)

                self._datasets.append(dataset)
                self._elements.update(dataset.composition.element)
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
            splited = [split_dataset_random(d, len(d)*9/10) + (d.config, d.composition) for d in self._datasets]
            yield splited, self._elements

    def __len__(self):
        return self._length

    def _parse_xyzfile(self):
        if mpi.rank == 0:
            pprint('config_type.dill is not found.\nparsing {} ... '.format(file_.xyz_file), end='', flush=True)
            config_type = set()
            alldataset = defaultdict(list)
            for data in AtomsReader(file_.xyz_file):
                config = data.config_type
                config_type.add(config)
                alldataset[config].append(data)
            with open(self._config_type_file, 'w') as f:
                dill.dump(config_type, f)

            for config in config_type:
                composition = {'index': defaultdict(set), 'element': []}
                for i, atom in enumerate(alldataset[config][0]):
                    composition['index'][atom.symbol].add(i)
                    composition['element'].append(atom.symbol)
                composition = DictAsAttributes(composition)

                data_dir = path.join(self._data_dir, config)
                mkdir(data_dir)
                writer = AtomsWriter(path.join(data_dir, 'structure.xyz'))
                for data in alldataset[config]:
                    if data.cohesive_energy < 0.0:
                        writer.write(data)
                writer.close()
                with open(path.join(data_dir, 'composition.dill'), 'w') as f:
                    dill.dump(composition, f)
            pprint('done')

        mpi.comm.Barrier()

    def _scatter(self, dataset):
        sub_nsample = (dataset.nsample + mpi.size - 1) / mpi.size
        slice = np.array([(dataset.nsample*i/mpi.size, dataset.nsample*i/mpi.size + sub_nsample)
                          for i in range(mpi.size)], dtype=np.int32)
        for attr in ['input', 'dinput', 'label', 'dlabel']:
            if mpi.rank != 0:
                shape = mpi.comm.bcast(None, root=0)
                recv = np.empty((sub_nsample,)+shape, dtype=np.float32)
                # mpi.comm.Scatterv(None, recv, root=0)
                mpi.comm.Recv(recv, source=0)
                setattr(dataset, attr, recv)
            else:
                data = getattr(dataset, attr)
                shape = mpi.comm.bcast(data.shape[1:], root=0)
                setattr(dataset, attr, data[:sub_nsample])
                for i in xrange(1, mpi.size):
                    mpi.comm.Send(data[slice[i][0]:slice[i][1]], dest=i)
                    # mpi.comm.Scatterv((data, (slice*data[0].size, None), MPI.FLOAT), recv, root=0)

    def _bcast(self, dataset):
        for attr in ['input', 'dinput', 'label', 'dlabel']:
            if mpi.rank != 0:
                shape = mpi.comm.bcast(None, root=0)
                data = np.empty(shape, dtype=np.float32)
                mpi.comm.Bcast(data, root=0)
                setattr(dataset, attr, data)
            else:
                data = getattr(dataset, attr)
                shape = mpi.comm.bcast(data.shape, root=0)
                mpi.comm.Bcast(data, root=0)
