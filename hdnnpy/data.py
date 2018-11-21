# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
import pickle
from itertools import product, combinations, combinations_with_replacement
import copy
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import KFold
import ase
import ase.io
import ase.neighborlist
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from . import settings as stg
from .preproc import PREPROC
from .util import pprint, mkdir


RANDOMSTATE = np.random.get_state()  # use the same random state to shuffle datesets on a execution


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
    def __init__(self, file_path, format):
        assert format in ['xyz', 'poscar']
        if format == 'xyz':
            self._load_xyz(file_path)
        elif format == 'poscar':
            self._load_poscar(file_path)

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
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Gs)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._dGs)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Es)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Fs)

    def _load_xyz(self, xyz_file):
        data_dir = xyz_file.parent
        self._config = data_dir.name
        self._composition = pickle.loads((data_dir/'composition.pickle').read_bytes())
        self._ifeat = defaultdict(dict)
        for idx, (jelem, kelem) in enumerate(combinations_with_replacement(self._composition['element'], 2)):
            self._ifeat[jelem][kelem] = self._ifeat[kelem][jelem] = idx
        self._atoms = ase.io.read(str(xyz_file), index=':', format='xyz')
        self._nsample = len(self._atoms)
        self._natom = len(self._composition['atom'])
        count = np.array([(self._nsample + i) // stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = self._atoms[count[:stg.mpi.rank].sum(): count[:stg.mpi.rank + 1].sum()]

        self._make_input(data_dir, count, save=True)
        self._make_label(data_dir, count, save=True)
        self._shuffle()  # shuffle dataset at once
        del self._atoms

    def _load_poscar(self, poscar):
        data_dir = poscar.parent
        unitcell = ase.io.read(str(poscar), format='vasp')
        atoms = []
        if stg.args.mode == 'prediction':
            atoms.append(unitcell)
        elif stg.args.mode == 'phonon':
            unitcell = PhonopyAtoms(cell=unitcell.cell,
                                    positions=unitcell.positions,
                                    symbols=unitcell.get_chemical_symbols())
            phonopy = Phonopy(unitcell, stg.phonopy.dimensions, **stg.phonopy.options)
            phonopy.generate_displacements(distance=stg.phonopy.distance)
            supercells = phonopy.get_supercells_with_displacements()
            self._phonopy = phonopy
            for phonopy_at in supercells:
                at = ase.Atoms(symbols=phonopy_at.get_chemical_symbols(),
                           positions=phonopy_at.get_positions(),
                           cell=phonopy_at.get_cell(),
                           pbc=True)
                atoms.append(at)

        symbols = atoms[0].get_chemical_symbols()
        self._composition = {'indices': {k: set([i for i, s in enumerate(symbols) if s == k])
                                         for k in set(symbols)},
                             'atom': symbols,
                             'element': sorted(set(symbols))}
        self._ifeat = defaultdict(dict)
        for idx, (jelem, kelem) in enumerate(combinations_with_replacement(self._composition['element'], 2)):
            self._ifeat[jelem][kelem] = self._ifeat[kelem][jelem] = idx
        self._nsample = len(atoms)
        self._natom = len(atoms[0])

        count = np.array([(self._nsample + i) // stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = atoms[count[:stg.mpi.rank].sum(): count[:stg.mpi.rank + 1].sum()]
        self._make_input(data_dir, count, save=False)
        del self._atoms

    def _make_label(self, data_dir, count, save):
        """At this point, only root process should have whole dataset.
        non-root processes work as calculators if necessary,
            but discard the calculated data once.
        preprocessed datasets will be scattered to all processes later.
        """
        EF_file = data_dir/'Energy_Force.npz'
        try:
            if not save:
                raise ValueError
            ndarray = np.load(EF_file)
            assert ndarray['nsample'] == self._nsample, \
                '# of samples of {} and given data file do not match.'.format(EF_file)
            pprint('{} is found.'.format(EF_file))
            Es = ndarray['E']
            Fs = ndarray['F']

        except (ValueError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                pprint('{} is not found.'.format(EF_file))
            pprint('calculate energy and forces from scratch.')

            if stg.mpi.rank == 0:
                Es = np.empty((self._nsample, 1))
                Fs = np.empty((self._nsample, self._natom, 3))
                Es_send = np.array([data.get_potential_energy() for data in self._atoms]).reshape(-1, 1)
                Fs_send = np.array([data.get_forces() for data in self._atoms])
                stg.mpi.comm.Gatherv(Es_send, (Es, (count, None), MPI.DOUBLE), root=0)
                stg.mpi.comm.Gatherv(Fs_send, (Fs, (count * self._natom * 3, None), MPI.DOUBLE), root=0)
                if save:
                    np.savez(EF_file, nsample=self._nsample, E=Es, F=Fs)
            else:
                Es_send = np.array([data.get_potential_energy() for data in self._atoms]).reshape(-1, 1)
                Fs_send = np.array([data.get_forces() for data in self._atoms])
                stg.mpi.comm.Gatherv(Es_send, None, 0)
                stg.mpi.comm.Gatherv(Fs_send, None, 0)

        if stg.mpi.rank == 0:
            self._Es = Es.astype(np.float32)
            self._Fs = Fs.astype(np.float32)
        else:
            self._Es = np.empty(0)
            self._Fs = np.empty(0)

    def _make_input(self, data_dir, count, save):
        """At this point, only root process should have whole dataset.
        non-root processes work as calculators if necessary,
            but discard the calculated data once.
        preprocessed datasets will be scattered to all processes later.
        """
        SF_file = data_dir/'Symmetry_Function.npz'
        try:
            if not save:
                raise ValueError
            ndarray = np.load(SF_file)
            assert ndarray['nsample'] == self._nsample, \
                '# of samples of {} and given data file do not match.'.format(SF_file)
            pprint('{} is found.'.format(SF_file))
            existing_keys = set([Path(key).parent for key in ndarray.keys() if key.endswith('G')])
        except (ValueError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                pprint('{} is not found.'.format(SF_file))
            pprint('calculate symmetry functions from scratch.')
            existing_keys = None
        new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys(existing_keys)
        if new_keys and save:
            pprint('uncalculated symmetry function parameters are as follows:')
            pprint('\n'.join(map(str, new_keys)))

        if stg.mpi.rank == 0:
            Gs, dGs = {}, {}
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                G = np.empty((self._nsample,) + G_send.shape[1:])
                dG = np.empty((self._nsample,) + dG_send.shape[1:])
                stg.mpi.comm.Gatherv(G_send, (G, (count * G[0].size, None), MPI.DOUBLE), root=0)
                stg.mpi.comm.Gatherv(dG_send, (dG, (count * dG[0].size, None), MPI.DOUBLE), root=0)
                Gs[str(key/'G')] = G
                dGs[str(key/'dG')] = dG
            for key in re_used_keys:
                Gs[str(key/'G')] = ndarray[str(key/'G')]
                dGs[str(key/'dG')] = ndarray[str(key/'dG')]
            self._Gs = np.concatenate([v for k, v in sorted(Gs.items())], axis=2).astype(np.float32)
            self._dGs = np.concatenate([v for k, v in sorted(dGs.items())], axis=2).astype(np.float32)
            if new_keys and save:
                for key in no_used_keys:
                    Gs[str(key/'G')] = ndarray[str(key/'G')]
                    dGs[str(key/'dG')] = ndarray[str(key/'dG')]
                np.savez(SF_file, nsample=self._nsample, **Gs, **dGs)

        else:
            for _, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                stg.mpi.comm.Gatherv(G_send, None, 0)
                stg.mpi.comm.Gatherv(dG_send, None, 0)
            self._Gs = np.empty(0)
            self._dGs = np.empty(0)

    def _check_uncalculated_keys(self, existing_keys=None):
        if existing_keys is None:
            existing_keys = set()
        required_keys = set()
        for Rc in stg.dataset.Rc:
            required_keys.add(Path('type1', str(Rc)))
        for Rc, eta, Rs in product(stg.dataset.Rc, stg.dataset.eta, stg.dataset.Rs):
            required_keys.add(Path('type2', str(Rc), str(eta), str(Rs)))
        for Rc, eta, lambda_, zeta in product(stg.dataset.Rc, stg.dataset.eta, stg.dataset.lambda_, stg.dataset.zeta):
            required_keys.add(Path('type4', str(Rc), str(eta), str(lambda_), str(zeta)))
        new_keys = sorted(required_keys - existing_keys)
        re_used_keys = sorted(required_keys & existing_keys)
        no_used_keys = sorted(existing_keys - required_keys)
        return new_keys, re_used_keys, no_used_keys

    def _calculate_symmetry_function(self, keys):
        Gs = defaultdict(list)
        dGs = defaultdict(list)
        for at in self._atoms:
            for key in keys:
                params = key.parts
                G, dG = getattr(self, '_' + params[0])(at, *map(float, params[1:]))
                Gs[key].append(G)
                dGs[key].append(dG)
        for key in keys:
            yield key, np.stack(Gs[key]), np.stack(dGs[key])

    def _type1(self, at, Rc):
        size = len(self._composition['element'])
        G = np.zeros((self._natom, size))
        dG = np.zeros((self._natom, size, self._natom, 3))
        for i, indices, R, tanh, dR, _, _ in self._neighbour_info(at, Rc):
            g = tanh ** 3
            dg = -3. / Rc * ((1. - tanh ** 2) * tanh ** 2)[:, None] * dR
            # G
            for jelem in self._composition['element']:
                G[i, self._ifeat[self._composition['element'][0]][jelem]] = g[indices[jelem]].sum()
            # dG
            for j, jelem in enumerate(self._composition['atom']):
                dG[i, self._ifeat[self._composition['element'][0]][jelem], j] = dg.take(indices[j], 0).sum(0)
        return G, dG

    def _type2(self, at, Rc, eta, Rs):
        size = len(self._composition['element'])
        G = np.zeros((self._natom, size))
        dG = np.zeros((self._natom, size, self._natom, 3))
        for i, indices, R, tanh, dR, _, _ in self._neighbour_info(at, Rc):
            g = np.exp(- eta * (R - Rs) ** 2) * tanh ** 3
            dg = (np.exp(- eta * (R - Rs) ** 2) * tanh ** 2 * (
                    -2. * eta * (R - Rs) * tanh + 3. / Rc * (tanh ** 2 - 1.0)))[:, None] * dR
            # G
            for jelem in self._composition['element']:
                G[i, self._ifeat[self._composition['element'][0]][jelem]] = g[indices[jelem]].sum()
            # dG
            for j, jelem in enumerate(self._composition['atom']):
                dG[i, self._ifeat[self._composition['element'][0]][jelem], j] = dg.take(indices[j], 0).sum(0)
        return G, dG

    def _type4(self, at, Rc, eta, lambda_, zeta):
        size = len(self._composition['element']) * (1 + len(self._composition['element'])) // 2
        G = np.zeros((self._natom, size))
        dG = np.zeros((self._natom, size, self._natom, 3))
        for i, indices, R, tanh, dR, cos, dcos in self._neighbour_info(at, Rc):
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
            for jelem in self._composition['element']:
                G[i, self._ifeat[jelem][jelem]] = g.take(indices[jelem], 0).take(indices[jelem], 1).sum() / 2.0
            for jelem, kelem in combinations(self._composition['element'], 2):
                G[i, self._ifeat[jelem][kelem]] = g.take(indices[jelem], 0).take(indices[kelem], 1).sum()
            # dG
            for (j, jelem), kelem in product(enumerate(self._composition['atom']), self._composition['element']):
                dG[i, self._ifeat[jelem][kelem], j] = dg.take(indices[j], 0).take(indices[kelem], 1).sum((0, 1))
        return G, dG

    @memorize
    def _neighbour_info(self, at, Rc):
        i_list, j_list, R_list, r_list = ase.neighborlist.neighbor_list('ijdD', at, Rc)
        for i in range(self._natom):
            i_ind, = np.where(i_list == i)
            if i_ind.size == 0:
                continue

            indices = defaultdict(list)
            for j_ind, j in enumerate(j_list[i_ind]):
                indices[j].append(j_ind)
                indices[self._composition['atom'][j]].append(j_ind)

            R = R_list[i_ind]
            r = r_list[i_ind]
            tanh = np.tanh(1 - R / Rc)
            dR = r / R[:, None]
            cos = dR.dot(dR.T)
            # cosine(j - i - k) differentiate w.r.t. "j"
            # dcos = - rj * cos / Rj**2 + rk / Rj / Rk
            dcos = - r[:, None, :] / R[:, None, None] ** 2 * cos[:, :, None] \
                   + r[None, :, :] / (R[:, None, None] * R[None, :, None])
            yield i, indices, R, tanh, dR, cos, dcos


class DataGenerator(object):
    def __init__(self, data_file, format):
        if format == 'xyz':
            self._construct_training_datasets(data_file, format)
        elif format == 'poscar':
            self._construct_test_datasets(data_file, format)

    def __call__(self):
        return self._datasets[0], self._elements

    @property
    def preproc(self):
        return self._preproc

    def holdout(self, ratio):
        split = []
        while self._datasets:
            dataset = self._datasets.pop(0)
            s = int(len(dataset) * ratio)
            train = dataset.take(slice(None, s, None))
            test = dataset.take(slice(s, None, None))
            split.append((train, test, dataset.composition))
        return split, self._elements

    def cross_validation(self, ratio, kfold):
        kf = KFold(n_splits=kfold)
        kfold_indices = [kf.split(range(int(len(dataset) * ratio)))
                         for dataset in self._datasets]

        for indices in zip(*kfold_indices):
            split = []
            for dataset, (train_idx, test_idx) in zip(self._datasets, indices):
                train = dataset.take(train_idx)
                test = dataset.take(test_idx)
                split.append((train, test, dataset.composition))
            yield split, self._elements

    def _construct_training_datasets(self, original_xyz, format):
        cfg_pickle = original_xyz.with_name('config_type.pickle')
        if not cfg_pickle.exists():
            parse_xyzfile(original_xyz)
        config_type = pickle.loads(cfg_pickle.read_bytes())
        required_config = sorted(config_type) if 'all' in stg.dataset.config else stg.dataset.config

        if stg.mpi.rank == 0:
            self._preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        else:
            self._preproc = PREPROC[None]()
        if stg.args.mode == 'training' and stg.args.resume:
            self._preproc.load(stg.file.out_dir/'preproc.npz')

        self._datasets = []
        elements = set()
        for config in required_config:
            if config not in config_type:
                continue
            pprint('Construct dataset of configuration type: {}'.format(config))

            parsed_xyz = original_xyz.with_name(config)/'structure.xyz'
            dataset = AtomicStructureDataset(parsed_xyz, format)
            self._preproc.decompose(dataset)
            stg.mpi.comm.Barrier()

            dataset = scatter_dataset(dataset)
            self._datasets.append(dataset)
            elements.update(dataset.composition['element'])
            pprint('')

        self._preproc.save(stg.file.out_dir/'preproc.npz')
        self._elements = sorted(elements)
        stg.dataset.nsample = sum([d.nsample for d in self._datasets])

    def _construct_test_datasets(self, poscar, format):
        self._preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        self._preproc.load(stg.file.out_dir/'preproc.npz')
        dataset = AtomicStructureDataset(poscar, format)
        self._preproc.decompose(dataset)
        self._datasets = [dataset]
        self._elements = dataset.composition['element']

def parse_xyzfile(xyz_file):
    if stg.mpi.rank == 0:
        pprint('config_type.pickle is not found.\nparsing {} ... '.format(xyz_file), end='')
        config_type = set()
        dataset = defaultdict(list)
        for atoms in ase.io.iread(str(xyz_file), index=':', format='xyz', parallel=False):
            config = atoms.info['config_type']
            config_type.add(config)
            dataset[config].append(atoms)
        xyz_file.with_name('config_type.pickle').write_bytes(pickle.dumps(config_type))

        for config in config_type:
            composition = {'indices': defaultdict(list), 'atom': [], 'element': []}
            for i, atom in enumerate(dataset[config][0]):
                composition['indices'][atom.symbol].append(i)
                composition['atom'].append(atom.symbol)
            composition['element'] = sorted(set(composition['atom']))

            cfg_dir = xyz_file.with_name(config)
            mkdir(cfg_dir)
            ase.io.write(str(cfg_dir/'structure.xyz'), dataset[config], format='xyz', parallel=False)
            (cfg_dir/'composition.pickle').write_bytes(pickle.dumps(dict(composition)))
        pprint('done')

    stg.mpi.comm.Barrier()


def scatter_dataset(dataset, root=0, max_buf_len=256 * 1024 * 1024):
    """Scatter the given dataset to the workers in the communicator.
    
    refer to chainermn.scatter_dataset()
    change:
        broadcast dataset by split size, NOT whole size, to the workers.
        omit shuffling dataset
        use raw mpi4py method to send/recv dataset
    """
    assert 0 <= root and root < stg.mpi.size

    if stg.mpi.rank == root:
        mine = None
        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + stg.mpi.size - 1) // stg.mpi.size

        for i in range(stg.mpi.size):
            b = n_total_samples * i // stg.mpi.size
            e = b + n_sub_samples

            if i == root:
                mine = dataset.take(slice(b, e, None))
            else:
                send = dataset.take(slice(b, e, None))
                send_chunk(send, dest=i, max_buf_len=max_buf_len)
        assert mine is not None
        return mine

    else:
        recv = recv_chunk(source=root, max_buf_len=max_buf_len)
        assert recv is not None
        return recv


INT_MAX = 2147483647


def send_chunk(obj, dest, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    pickled_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    total_bytes = len(pickled_bytes)
    total_chunk_num = -(-total_bytes // max_buf_len)
    stg.mpi.comm.send((total_chunk_num, max_buf_len, total_bytes), dest=dest, tag=1)

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = pickled_bytes[b:e]
        stg.mpi.comm.Send(buf, dest=dest, tag=2)


def recv_chunk(source, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    data = stg.mpi.comm.recv(source=source, tag=1)
    assert data is not None
    total_chunk_num, max_buf_len, total_bytes = data
    pickled_bytes = bytearray()

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = bytearray(e - b)
        stg.mpi.comm.Recv(buf, source=source, tag=2)
        pickled_bytes[b:e] = buf

    obj = pickle.loads(pickled_bytes)
    return obj
