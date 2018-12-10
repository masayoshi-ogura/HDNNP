# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
import pickle
from itertools import product, combinations_with_replacement
import copy
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import KFold
import ase.io

from .. import settings as stg
from ..preproc import PREPROC
from ..util import pprint
from .file_io import load_xyz, load_poscar, parse_xyzfile
from .symmetry_functions import type1, type2, type4


RANDOMSTATE = np.random.get_state()  # use the same random state to shuffle datesets on a execution


class AtomicStructureDataset(object):
    def __init__(self, file_path, format):
        assert format in ['xyz', 'poscar']

        if format == 'xyz':
            atoms, self._config, self._elemental_composition = load_xyz(file_path)
        elif format == 'poscar':
            atoms, self._config, self._elemental_composition = load_poscar(file_path)

        self._nsample = len(atoms)
        self._count = np.array([(self._nsample + i) // stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = atoms[self._count[:stg.mpi.rank].sum(): self._count[:stg.mpi.rank + 1].sum()]

        if format == 'xyz':
            self._make_input(file_path.parent, save=True)
            self._make_label(file_path.parent, save=True)
            self._shuffle()  # shuffle dataset at once
        elif format == 'poscar':
            self._make_input(Path(), save=False)

        del self._atoms

    def __getitem__(self, index):
        batches = [self._Gs[index], self._dGs[index], self._Es[index], self._Fs[index]]
        if isinstance(index, slice):
            return [tuple([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        else:
            return tuple(batches)

    def __len__(self):
        return self._Gs.shape[0]

    @property
    def elemental_composition(self):
        return self._elemental_composition

    @property
    def elements(self):
        return sorted(set(self._elemental_composition))

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

    def _make_label(self, data_dir, save):
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
                natom = len(self._atoms[0])
                Es = np.empty((self._nsample, 1))
                Fs = np.empty((self._nsample, natom, 3))
                Es_send = np.array([data.get_potential_energy() for data in self._atoms]).reshape(-1, 1)
                Fs_send = np.array([data.get_forces() for data in self._atoms])
                stg.mpi.comm.Gatherv(Es_send, (Es, (self._count, None), MPI.DOUBLE), root=0)
                stg.mpi.comm.Gatherv(Fs_send, (Fs, (self._count * natom * 3, None), MPI.DOUBLE), root=0)
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

    def _make_input(self, data_dir, save):
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
                stg.mpi.comm.Gatherv(G_send, (G, (self._count * G[0].size, None), MPI.DOUBLE), root=0)
                stg.mpi.comm.Gatherv(dG_send, (dG, (self._count * dG[0].size, None), MPI.DOUBLE), root=0)
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
        ifeat = defaultdict(dict)
        for idx, (jelem, kelem) in enumerate(
                combinations_with_replacement(sorted(set(self._atoms[0].get_chemical_symbols())), 2)):
            ifeat[jelem][kelem] = ifeat[kelem][jelem] = idx
        for atoms in self._atoms:
            for key in keys:
                params = key.parts
                G, dG = eval(params[0])(ifeat, atoms, self.elements, *map(float, params[1:]))
                Gs[key].append(G)
                dGs[key].append(dG)
        for key in keys:
            yield key, np.stack(Gs[key]), np.stack(dGs[key])


class AtomicStructureDatasetGenerator(object):
    def __init__(self, data_file, format):
        if format == 'xyz':
            self._construct_training_datasets(data_file, format)
        elif format == 'poscar':
            self._construct_test_datasets(data_file, format)

    def __iter__(self):
        for poscars, dataset in self._datasets:
            yield poscars, dataset, self._elements

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
            split.append((train, test))
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
                split.append((train, test))
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
        if stg.args.mode == 'train' and stg.args.is_resume:
            self._preproc.load(stg.args.resume_dir.with_name('preproc.npz'))

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
            elements.update(dataset.elements)
            pprint('')

        self._preproc.save(stg.file.out_dir/'preproc.npz')
        self._elements = sorted(elements)
        stg.dataset.nsample = sum([d.nsample for d in self._datasets])

    def _construct_test_datasets(self, original_poscars, format):
        configurations = defaultdict(list)
        for poscar in original_poscars:
            formula = ase.io.read(str(poscar), format='vasp').get_chemical_formula()
            configurations[formula].append(poscar)

        self._preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        self._preproc.load(stg.args.masters.with_name('preproc.npz'))

        self._datasets = []
        elements = set()
        for poscars in configurations.values():
            dataset = AtomicStructureDataset(poscars, format)
            self._preproc.decompose(dataset)
            self._datasets.append((poscars, dataset))
            elements.update(dataset.elements)
        self._elements = sorted(elements)


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
