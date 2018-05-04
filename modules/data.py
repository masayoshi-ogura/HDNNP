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
from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.datasets import get_cross_validation_datasets_random
from quippy import Atoms
from quippy import AtomsReader
from quippy import AtomsList
from quippy import AtomsWriter
from quippy import farray
from quippy import fzeros
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToCm

from .util import pprint, mkdir
from .util import DictAsAttributes


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
    def __init__(self, hp):
        self._hp = hp

    def __len__(self):
        return self._nsample

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
    def input(self):
        return self._datasets[0]

    @property
    def dinput(self):
        return self._datasets[1]

    def load_xyz(self, xyz_file):
        self._data_dir = path.dirname(xyz_file)
        self._config = path.basename(self._data_dir)
        with open(path.join(self._data_dir, 'composition.dill')) as f:
            self._composition = dill.load(f)
        self._nsample = len(AtomsReader(xyz_file))
        self._natom = len(self._composition.element)
        self._atoms_objs = AtomsList(xyz_file, start=mpi.rank, step=mpi.size) if mpi.rank < self._nsample else []

        self._configure_mpi()
        Es, Fs = self._make_label(save=True)
        Gs, dGs = self._make_input(save=True)
        self._datasets = (Gs, dGs, Es, Fs)

    def load_poscar(self, poscar, dimension=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], distance=0.03, save=True, scale=1.0):
        self._data_dir = path.dirname(poscar)
        self._config = path.basename(self._data_dir)
        unitcell, = AtomsList(poscar, format='POSCAR')
        if self._hp.mode == 'optimize':
            supercells = []
            for k in np.linspace(0.9, 1.1, 201):
                supercell = unitcell.copy()
                supercell.set_lattice(unitcell.lattice * k, scale_positions=True)
                supercells.append(supercell)
            self._atoms_objs = supercells
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
            self._atoms_objs = []
            for pa in supercells:
                atoms = Atoms(cell=pa.cell,
                              positions=pa.get_positions(),
                              numbers=pa.numbers,
                              masses=pa.masses)
                atoms.set_chemical_symbols(pa.get_chemical_symbols())
                self._atoms_objs.append(atoms)

        symbols = self._atoms_objs[0].get_chemical_symbols()
        composition = {'index': {k: set([i for i, s in enumerate(symbols) if s == k]) for k in set(symbols)},
                       'element': symbols}
        self._composition = DictAsAttributes(composition)
        self._nsample = len(supercells)
        self._natom = supercells[0].get_number_of_atoms()

        self._configure_mpi()
        Gs, dGs = self._make_input(save=save)
        self._datasets = (Gs, dGs)

    def reset_inputs(self, input, dinput):
        self._datasets = (input, dinput) + self._datasets[2:]

    def _configure_mpi(self):
        quo = self._nsample / mpi.size
        rem = self._nsample % mpi.size
        self._count = np.array([quo+1 if i < rem else quo
                                for i in xrange(mpi.size)], dtype=np.int32)
        self._disps = np.array([np.sum(self._count[:i])
                                for i in xrange(mpi.size)], dtype=np.int32)
        self._n = self._count[mpi.rank]  # the number of allocated samples in this node

    def _make_label(self, save):
        EF_file = path.join(self._data_dir, 'Energy_Force.npz')
        if path.exists(EF_file):
            ndarray = np.load(EF_file)
            return ndarray['E'], ndarray['F']
        else:
            pprint('{} doesn\'t exist. calculating ...'.format(EF_file), end='\r')
            Es = np.empty((self._nsample, 1), dtype=np.float32)
            Fs = np.empty((self._nsample, self._natom, 3), dtype=np.float32)
            Es_send = np.array([data.cohesive_energy for data in self._atoms_objs], dtype=np.float32).reshape(-1, 1)
            Fs_send = np.array([data.force.T for data in self._atoms_objs], dtype=np.float32).reshape(-1, self._natom, 3)
            mpi.comm.Allgatherv((Es_send, (self._n), MPI.FLOAT),
                                (Es, (self._count, self._disps), MPI.FLOAT))
            num = self._natom * 3
            mpi.comm.Allgatherv((Fs_send, (self._n*num), MPI.FLOAT),
                                (Fs, (self._count*num, self._disps*num), MPI.FLOAT))
            if save:
                np.savez(EF_file, E=Es, F=Fs)
            return Es, Fs

    def _make_input(self, save):
        Gs = []
        dGs = []
        SF_file = path.join(self._data_dir, 'Symmetry_Function.npz')
        existing = np.load(SF_file) if save and path.exists(SF_file) else {}
        new = {}

        # type 1
        for Rc in self._hp.Rc:
            key = path.join(*map(str, ['type1', Rc]))
            G_key = path.join('G', key)
            dG_key = path.join('dG', key)
            if G_key in existing and existing[G_key].shape[0] == self._nsample:
                Gs.append(existing[G_key])
                dGs.append(existing[dG_key])
            else:
                pprint('calculating symmetry function {} ...'.format(key), end='\r')
                G = np.empty((self._nsample, self._natom, 2), dtype=np.float32)
                dG = np.empty((self._nsample, self._natom, 2, self._natom, 3), dtype=np.float32)
                G_send, dG_send = self._calc_G1(Rc)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 2 * self._natom**2 * 3
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                Gs.append(G)
                dGs.append(dG)
                new[G_key] = G
                new[dG_key] = dG

        # type 2
        for Rc, eta, Rs in product(self._hp.Rc, self._hp.eta, self._hp.Rs):
            key = path.join(*map(str, ['type2', Rc, eta, Rs]))
            G_key = path.join('G', key)
            dG_key = path.join('dG', key)
            if G_key in existing and existing[G_key].shape[0] == self._nsample:
                Gs.append(existing[G_key])
                dGs.append(existing[dG_key])
            else:
                pprint('calculating symmetry function {} ...'.format(key), end='\r')
                G = np.empty((self._nsample, self._natom, 2), dtype=np.float32)
                dG = np.empty((self._nsample, self._natom, 2, self._natom, 3), dtype=np.float32)
                G_send, dG_send = self._calc_G2(Rc, eta, Rs)
                num = 2 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 2 * self._natom**2 * 3
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                Gs.append(G)
                dGs.append(dG)
                new[G_key] = G
                new[dG_key] = dG

        # type 4
        for Rc, eta, lambda_, zeta in product(self._hp.Rc, self._hp.eta, self._hp.lambda_, self._hp.zeta):
            key = path.join(*map(str, ['type4', Rc, eta, lambda_, zeta]))
            G_key = path.join('G', key)
            dG_key = path.join('dG', key)
            if G_key in existing and existing[G_key].shape[0] == self._nsample:
                Gs.append(existing[G_key])
                dGs.append(existing[dG_key])
            else:
                pprint('calculating symmetry function {} ...'.format(key), end='\r')
                G = np.empty((self._nsample, self._natom, 3), dtype=np.float32)
                dG = np.empty((self._nsample, self._natom, 3, self._natom, 3), dtype=np.float32)
                G_send, dG_send = self._calc_G4(Rc, eta, lambda_, zeta)
                num = 3 * self._natom
                mpi.comm.Allgatherv((G_send, (self._n*num), MPI.FLOAT),
                                    (G, (self._count*num, self._disps*num), MPI.FLOAT))
                num = 3 * self._natom**2 * 3
                mpi.comm.Allgatherv((dG_send, (self._n*num), MPI.FLOAT),
                                    (dG, (self._count*num, self._disps*num), MPI.FLOAT))
                Gs.append(G)
                dGs.append(dG)
                new[G_key] = G
                new[dG_key] = dG

        if save and mpi.rank == 0:
            np.savez(SF_file, **dict(existing.items() + new.items()))
        Gs = np.concatenate(Gs, axis=2)  # (sample, atom, feature)
        dGs = np.concatenate(dGs, axis=2)  # (sample, atom, feature, atom, 3)
        return Gs, dGs

    def _calc_G1(self, Rc):
        G = np.zeros((self._n, self._natom, 2), dtype=np.float32)
        dG = np.zeros((self._n, self._natom, 2, self._natom, 3), dtype=np.float32)
        for i, k, neighbour, R, fc, tanh, dR, _, _ in self._calc_geometry(Rc):
            dg = -3./Rc * ((1.-tanh**2)*tanh**2)[:, None] * dR
            # homo
            G[i, k, 0] = np.sum(fc[neighbour['homo_all']])
            dG[i, k, 0] = np.array([dg[indices].sum(axis=0) for indices in neighbour['homo']])
            # hetero
            G[i, k, 1] = np.sum(fc[neighbour['hetero_all']])
            dG[i, k, 1] = np.array([dg[indices].sum(axis=0) for indices in neighbour['hetero']])
        return G, dG

    def _calc_G2(self, Rc, eta, Rs):
        G = np.zeros((self._n, self._natom, 2), dtype=np.float32)
        dG = np.zeros((self._n, self._natom, 2, self._natom, 3), dtype=np.float32)
        for i, k, neighbour, R, fc, tanh, dR, _, _ in self._calc_geometry(Rc):
            g = np.exp(- eta * (R - Rs)**2) * fc
            dg = (g * (-2.*eta*(R-Rs) + 3./Rc*(tanh - 1./tanh)))[:, None] * dR
            # homo
            G[i, k, 0] = np.sum(g[neighbour['homo_all']])
            dG[i, k, 0] = np.array([dg[indices].sum(axis=0) for indices in neighbour['homo']])
            # hetero
            G[i, k, 1] = np.sum(g[neighbour['hetero_all']])
            dG[i, k, 1] = np.array([dg[indices].sum(axis=0) for indices in neighbour['hetero']])
        return G, dG

    def _calc_G4(self, Rc, eta, lambda_, zeta):
        G = np.zeros((self._n, self._natom, 3), dtype=np.float32)
        dG = np.zeros((self._n, self._natom, 3, self._natom, 3), dtype=np.float32)
        for i, k, neighbour, R, fc, tanh, dR, cos, dcos in self._calc_geometry(Rc):
            ang = 1. + lambda_ * cos
            ang[np.identity(len(R), dtype=bool)] = 0.
            g = 2.**(1-zeta) \
                * np.einsum('jk,j,k->jk',
                            ang**zeta,
                            np.exp(-eta * R**2) * fc,
                            np.exp(-eta * R**2) * fc)
            dg_radial_part1 = 2.**(1-zeta) \
                * np.einsum('jk,j,k,ja->jka',
                            ang**zeta,
                            np.exp(-eta * R**2) * fc * (-2.*eta*R + 3./Rc*(tanh - 1./tanh)),
                            np.exp(-eta * R**2) * fc,
                            dR)
            dg_radial_part2 = 2.**(1-zeta) \
                * np.einsum('jk,j,k,ka->jka',
                            ang**zeta,
                            np.exp(-eta * R**2) * fc * (-2.*eta*R + 3./Rc*(tanh - 1./tanh)),
                            np.exp(-eta * R**2) * fc,
                            dR)
            dg_angular_part = zeta * lambda_ * 2.**(1-zeta) \
                * np.einsum('jk,j,k,jka->jka',
                            ang**(zeta-1),
                            np.exp(-eta * R**2) * fc,
                            np.exp(-eta * R**2) * fc,
                            dcos)
            dg = dg_radial_part1 + dg_radial_part2 + dg_angular_part
            # homo
            G[i, k, 0] = np.sum(g[np.ix_(neighbour['homo_all'], neighbour['homo_all'])]) / 2.
            dG[i, k, 0] = np.array([dg[np.ix_(indices, indices)].sum(axis=(0, 1)) for indices in neighbour['homo']]) / 2.
            # hetero
            G[i, k, 1] = np.sum(g[np.ix_(neighbour['hetero_all'], neighbour['hetero_all'])]) / 2.
            dG[i, k, 1] = np.array([dg[np.ix_(indices, indices)].sum(axis=(0, 1)) for indices in neighbour['hetero']]) / 2.
            # mix
            G[i, k, 2] = np.sum(g[np.ix_(neighbour['homo_all'], neighbour['hetero_all'])])
            dG[i, k, 2] = np.array([dg[np.ix_(homo, hetero)].sum(axis=(0, 1)) for homo, hetero in zip(neighbour['homo'], neighbour['hetero'])])
        return G, dG

    def memorize_generator(f):
        cache = defaultdict(list)
        done = []

        def helper(self, Rc):
            if id(self) not in done:
                cache.clear()
                done.append(id(self))

            if Rc not in cache:
                for ret in f(self, Rc):
                    cache[Rc].append(ret)
                    yield cache[Rc][-1]
            else:
                for ret in cache[Rc]:
                    yield ret
        return helper

    @memorize_generator
    def _calc_geometry(self, Rc):
        for i, atoms in enumerate(self._atoms_objs):
            atoms.set_cutoff(Rc)
            atoms.calc_connect()
            for k in xrange(self._natom):
                n_neighb = atoms.n_neighbours(k+1)
                if n_neighb == 0:
                    yield i, k, [], np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), \
                        np.ones(1, dtype=np.float32), np.zeros((1, 3), dtype=np.float32), \
                        np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1, 3), dtype=np.float32)
                    continue

                r, R, cos = self._neighbour(k, n_neighb, atoms)
                dR = r / R[:, None]
                dcos = self._deriv_cosine(n_neighb, r, R, cos)
                fc = np.tanh(1-R/Rc)**3
                tanh = np.tanh(1-R/Rc)

                element = self._composition.element[k]  # element of the focused atom
                neighbour_list = atoms.connect.get_neighbours(k+1)[0] - 1  # [0, 4, 1, 1, 3, 2 ...]; length=n_neighb
                neighbour = {'homo': [], 'hetero': [], 'homo_all': [], 'hetero_all': []}
                for j in xrange(self._natom):
                    if j in self._composition.index[element]:
                        neighbour['homo'].append([m for m, index in enumerate(neighbour_list) if index == j])
                        neighbour['hetero'].append([])
                        neighbour['homo_all'].extend(neighbour['homo'][j])
                    else:
                        neighbour['homo'].append([])
                        neighbour['hetero'].append([m for m, index in enumerate(neighbour_list) if index == j])
                        neighbour['hetero_all'].extend(neighbour['hetero'][j])

                yield i, k, neighbour, R, fc, tanh, dR, cos, dcos

    def _neighbour(self, k, n_neighb, atoms):
        diff = np.zeros((n_neighb, 3))
        cos = np.zeros((n_neighb, n_neighb), dtype=np.float32)
        for l in xrange(n_neighb):
            atoms.neighbour(k+1, l+1, diff=diff[l])
            for m in xrange(l):
                cos[l, m] = atoms.cosine_neighbour(k+1, l+1, m+1)
                cos[m, l] = cos[l, m]
        r = diff.astype(np.float32)
        R = np.linalg.norm(r, axis=1)
        return r, R, cos

    def _deriv_cosine(self, n_neighb, r, R, cos):
        dcos = - r[:, None, :]/R[:, None, None]**2 * cos[:, :, None] \
            + r[None, :, :]/(R[:, None, None] * R[None, :, None])  # derivative of cosine(theta) w.r.t. the position of atom l
        for i in xrange(n_neighb):
            dcos[i, i, :] = 0
        return dcos


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
                dataset = AtomicStructureDataset(self._hp)
                dataset.load_xyz(xyz_file)
                self._precond.decompose(dataset)
                self._datasets.append(dataset)
                self._elements.update(dataset.composition.element)
        self._length = sum([len(d) for d in self._datasets])

    def __iter__(self):
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
            pprint('config_type.dill is not found.\nLoad all data from xyz file ...', end='', flush=True)
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
