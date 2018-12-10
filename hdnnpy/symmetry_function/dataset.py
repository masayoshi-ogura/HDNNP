# -*- coding: utf-8 -*-

import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
import pickle
from itertools import product, combinations_with_replacement
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import KFold
import ase.io

from .. import settings as stg
from ..preproc import PREPROC
from ..util import pprint
from .file_io import parse_xyz
from .symmetry_functions import type1, type2, type4


RANDOMSTATE = np.random.get_state()  # use the same random state to shuffle datesets on a execution


class SymmetryFunctionDataset(object):
    def __init__(self, elemental_composition=None, tag=None, nsample=None, Gs=None, dGs=None, Es=None, Fs=None):
        self.elemental_composition = elemental_composition or []
        self.tag = tag or None
        self.nsample = nsample or None

        self._Gs = Gs if Gs is not None else np.empty(0)
        self._dGs = dGs if dGs is not None else np.empty(0)
        self._Es = Es if Es is not None else np.empty(0)
        self._Fs = Fs if Fs is not None else np.empty(0)

        self._data_dir_path = Path()
        self._atoms = []
        self._count = None
        self._is_save_input = False
        self._is_save_label = False

    def __getitem__(self, index):
        batches = [self._Gs[index], self._dGs[index], self._Es[index], self._Fs[index]]
        if isinstance(index, slice):
            return [tuple([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        else:
            return tuple(batches)

    def __len__(self):
        return self._Gs.shape[0]

    @property
    def elements(self):
        return sorted(set(self.elemental_composition))

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

    def load(self, xyz_path, verbose=True):
        self._data_dir_path = xyz_path.parent.absolute()
        atoms = ase.io.read(xyz_path, index=':', format='xyz')
        self.elemental_composition = atoms[0].get_chemical_symbols()
        self.tag = atoms[0].info['tag']
        self.nsample = len(atoms)
        self._count = np.array([(self.nsample + i) // stg.mpi.size for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        self._atoms = atoms[self._count[:stg.mpi.rank].sum(): self._count[:stg.mpi.rank + 1].sum()]
        if atoms[0].calc is not None:
            self._make_input(verbose)
            self._make_label(verbose)
            self._shuffle()  # shuffle dataset at once
        else:
            self._make_input(verbose)
        del self._atoms

    def save(self):
        if self._is_save_input:
            shutil.copy(self._temp_SF_file.name,
                        self._data_dir_path/'Symmetry_Function.npz')
        if self._is_save_label:
            shutil.copy(self._temp_EF_file.name,
                        self._data_dir_path/'Energy_Force.npz')

    def take(self, slc):
        assert isinstance(slc, slice) or isinstance(slc, np.ndarray)
        sliced = SymmetryFunctionDataset(
            elemental_composition=self.elemental_composition,
            tag=self.tag,
            nsample=self.nsample,
            Gs=self.input[slc],
            dGs=self.dinput[slc],
            Es=self.label[slc],
            Fs=self.dlabel[slc],
        )
        return sliced

    def _make_input(self, verbose):
        """At this point, only root process should have whole dataset.
        non-root processes work as calculators if necessary,
            but discard the calculated data once.
        preprocessed datasets will be scattered to all processes later.
        """
        SF_path = self._data_dir_path/'Symmetry_Function.npz'
        try:
            ndarray = np.load(SF_path)
            assert ndarray['nsample'] == self.nsample, \
                '# of samples of {} and given data file do not match.'.format(SF_path)
            existing_keys = set([Path(key).parent for key in ndarray.keys() if key.endswith('G')])
            if verbose:
                pprint('Loaded symmetry functions from {}.'.format(SF_path))
        except FileNotFoundError as e:
            if verbose:
                pprint('{} does not exist.'.format(SF_path))
                pprint('Calculate symmetry functions from scratch.')
            existing_keys = None
        new_keys, re_used_keys, no_used_keys = self._check_uncalculated_keys(existing_keys)
        if new_keys and verbose:
            pprint('Uncalculated symmetry function parameters are as follows:')
            pprint('\n'.join(map(str, new_keys)))

        if stg.mpi.rank == 0:
            Gs, dGs = {}, {}
            for key, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                G_send = G_send.astype(np.float32)
                dG_send = dG_send.astype(np.float32)
                G = np.empty((self.nsample,) + G_send.shape[1:], dtype=np.float32)
                dG = np.empty((self.nsample,) + dG_send.shape[1:], dtype=np.float32)
                stg.mpi.comm.Gatherv(
                        G_send, (G, (self._count * G[0].size, None), MPI.FLOAT), root=0)
                stg.mpi.comm.Gatherv(
                        dG_send, (dG, (self._count * dG[0].size, None), MPI.FLOAT), root=0)
                Gs[str(key/'G')] = G
                dGs[str(key/'dG')] = dG
            for key in re_used_keys:
                Gs[str(key/'G')] = ndarray[str(key/'G')]
                dGs[str(key/'dG')] = ndarray[str(key/'dG')]
            self._Gs = np.concatenate([v for k, v in sorted(Gs.items())], axis=2)
            self._dGs = np.concatenate([v for k, v in sorted(dGs.items())], axis=2)
            if new_keys:
                for key in no_used_keys:
                    Gs[str(key/'G')] = ndarray[str(key/'G')]
                    dGs[str(key/'dG')] = ndarray[str(key/'dG')]
                self._is_save_input = True
                self._temp_SF_file = tempfile.NamedTemporaryFile()
                np.savez(self._temp_SF_file, nsample=self.nsample, **Gs, **dGs)

        else:
            for _, G_send, dG_send in self._calculate_symmetry_function(new_keys):
                G_send = G_send.astype(np.float32)
                dG_send = dG_send.astype(np.float32)
                stg.mpi.comm.Gatherv(G_send, None, 0)
                stg.mpi.comm.Gatherv(dG_send, None, 0)

    def _make_label(self, verbose):
        """At this point, only root process should have whole dataset.
        non-root processes work as calculators if necessary,
            but discard the calculated data once.
        preprocessed datasets will be scattered to all processes later.
        """
        EF_path = self._data_dir_path/'Energy_Force.npz'
        try:
            ndarray = np.load(EF_path)
            assert ndarray['nsample'] == self.nsample, \
                '# of samples of {} and given data file do not match.'.format(EF_path)
            self._Es = ndarray['energy']
            self._Fs = ndarray['force']
            if verbose:
                pprint('Loaded energies and forces from {}.'.format(EF_path))

        except FileNotFoundError as e:
            if verbose:
                pprint('{} does not exist.'.format(EF_path))
                pprint('Calculate energies and forces from scratch.')

            if stg.mpi.rank == 0:
                natom = len(self._atoms[0])
                self._Es = np.empty((self.nsample, 1), dtype=np.float32)
                self._Fs = np.empty((self.nsample, natom, 3), dtype=np.float32)
                Es_send = np.array([data.get_potential_energy() for data in self._atoms],
                                   dtype=np.float32).reshape(-1, 1)
                Fs_send = np.array([data.get_forces() for data in self._atoms],
                                   dtype=np.float32)
                stg.mpi.comm.Gatherv(
                        Es_send, (self._Es, (self._count, None), MPI.FLOAT), root=0)
                stg.mpi.comm.Gatherv(
                        Fs_send, (self._Fs, (self._count * natom * 3, None), MPI.FLOAT), root=0)
                self._is_save_label = True
                self._temp_EF_file = tempfile.NamedTemporaryFile()
                np.savez(self._temp_EF_file, nsample=self.nsample, energy=self._Es, force=self._Fs)
            else:
                Es_send = np.array([data.get_potential_energy() for data in self._atoms],
                                   dtype=np.float32).reshape(-1, 1)
                Fs_send = np.array([data.get_forces() for data in self._atoms],
                                   dtype=np.float32)
                stg.mpi.comm.Gatherv(Es_send, None, 0)
                stg.mpi.comm.Gatherv(Fs_send, None, 0)

    def _check_uncalculated_keys(self, existing_keys):
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

    def _shuffle(self):
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Gs)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._dGs)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Es)
        np.random.set_state(RANDOMSTATE)
        np.random.shuffle(self._Fs)


class SymmetryFunctionDatasetGenerator(object):
    def __init__(self):
        self._datasets = []
        self._elements = []
        self._preproc = PREPROC[None]()

    @property
    def preproc(self):
        return self._preproc

    def load_xyz(self, file_path):
        file_path = file_path.absolute()
        all_tags = parse_xyz(file_path)
        included_tags = all_tags if 'all' in stg.dataset.tag else stg.dataset.tag

        if stg.mpi.rank == 0:
            self._preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        if stg.args.mode == 'train' and stg.args.is_resume:
            self._preproc.load(stg.args.resume_dir.with_name('preproc.npz'))

        elements = set()
        stg.dataset.nsample = 0
        for tag in included_tags:
            if tag not in all_tags:
                pprint('Data tagged "{}" does not exist in {}. Skipped.\n'.format(tag, file_path))
                continue
            pprint('Construct dataset tagged "{}"'.format(tag))

            xyz = file_path.with_name(tag)/'Atomic_Structure.xyz'
            dataset = SymmetryFunctionDataset()
            dataset.load(xyz, verbose=True)
            dataset.save()
            elements.update(dataset.elements)
            stg.dataset.nsample += dataset.nsample

            self._preproc.decompose(dataset)
            dataset = scatter_dataset(dataset)
            self._datasets.append(dataset)
            pprint('')

        self._preproc.save(stg.file.out_dir/'preproc.npz')
        self._elements = sorted(elements)

    def load_poscars(self, file_paths):
        tag_poscars_map = defaultdict(list)
        for poscar in file_paths:
            tag = ase.io.read(poscar, format='vasp').get_chemical_formula()
            tag_poscars_map[tag].append(poscar)

        self._preproc = PREPROC[stg.dataset.preproc](stg.dataset.nfeature)
        self._preproc.load(stg.args.masters.with_name('preproc.npz'))

        elements = set()
        for poscars in tag_poscars_map.values():
            with tempfile.NamedTemporaryFile('w') as xyz:
                for poscar in poscars:
                    atoms = ase.io.read(poscar, format='vasp')
                    atoms.info['tag'] = atoms.get_chemical_symbols()
                    ase.io.write(xyz.name, atoms, format='xyz', append=True)
                dataset = SymmetryFunctionDataset()
                dataset.load(Path(xyz.name), verbose=False)
            self._preproc.decompose(dataset)
            self._datasets.append(dataset)
            elements.update(dataset.elements)
        self._elements = sorted(elements)

        return tag_poscars_map.values()

    def all(self):
        return self._datasets, self._elements

    def foreach(self):
        for dataset in self._datasets:
            yield dataset, self._elements

    def holdout(self, ratio):
        split = []
        while self._datasets:
            dataset = self._datasets.pop(0)
            s = int(len(dataset) * ratio)
            train = dataset.take(slice(None, s, None))
            test = dataset.take(slice(s, None, None))
            split.append((train, test))
        return split, self._elements

    def kfold(self, kfold):
        kf = KFold(n_splits=kfold)
        kfold_indices = [kf.split(range(len(dataset))) for dataset in self._datasets]

        for indices in zip(*kfold_indices):
            split = []
            for dataset, (train_idx, test_idx) in zip(self._datasets, indices):
                train = dataset.take(train_idx)
                test = dataset.take(test_idx)
                split.append((train, test))
            yield split, self._elements


def scatter_dataset(dataset, root=0, max_buf_len=256 * 1024 * 1024):
    """Scatter the given dataset to the workers in the communicator.
    
    refer to chainermn.scatter_dataset()
    change:
        broadcast dataset by split size, NOT whole size, to the workers.
        omit shuffling dataset
        use raw mpi4py method to send/recv dataset
    """
    assert 0 <= root and root < stg.mpi.size
    stg.mpi.comm.Barrier()

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
