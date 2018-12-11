# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import ase.neighborlist


def memorize(f):
    cache = defaultdict(list)
    identifier = ['']

    def helper(atoms, Rc):
        if identifier[0] != atoms.info['tag'] + str(id(atoms)):
            cache.clear()
            identifier[0] = atoms.info['tag'] + str(id(atoms))

        if Rc not in cache:
            for ret in f(atoms, Rc):
                cache[Rc].append(ret)
                yield cache[Rc][-1]
        else:
            for ret in cache[Rc]:
                yield ret

    return helper


@memorize
def neighbour_info(atoms, Rc):
    i_list, j_list, R_list, r_list = ase.neighborlist.neighbor_list('ijdD', atoms, Rc)
    for i in range(len(atoms)):
        i_ind, = np.where(i_list == i)
        if i_ind.size == 0:
            continue

        indices = defaultdict(list)
        for j_ind, j in enumerate(j_list[i_ind]):
            indices[j].append(j_ind)
            indices[atoms[j].symbol].append(j_ind)

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
