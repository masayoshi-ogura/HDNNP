# -*- coding: utf-8 -*-

from itertools import product, combinations
import numpy as np

from .atomic_structure import neighbour_info


def type1(ifeat, atoms, elements, Rc):
    natom = len(atoms)
    size = len(elements)
    G = np.zeros((natom, size))
    dG = np.zeros((natom, size, natom, 3))
    for i, indices, R, tanh, dR, _, _ in neighbour_info(atoms, Rc):
        g = tanh ** 3
        dg = -3. / Rc * ((1. - tanh ** 2) * tanh ** 2)[:, None] * dR
        # G
        for jelem in elements:
            G[i, ifeat[elements[0]][jelem]] = g[indices[jelem]].sum()
        # dG
        for j, jelem in enumerate(atoms.get_chemical_symbols()):
            dG[i, ifeat[elements[0]][jelem], j] = dg.take(indices[j], 0).sum(0)
    return G, dG


def type2(ifeat, atoms, elements, Rc, eta, Rs):
    natom = len(atoms)
    size = len(elements)
    G = np.zeros((natom, size))
    dG = np.zeros((natom, size, natom, 3))
    for i, indices, R, tanh, dR, _, _ in neighbour_info(atoms, Rc):
        g = np.exp(- eta * (R - Rs) ** 2) * tanh ** 3
        dg = (np.exp(- eta * (R - Rs) ** 2) * tanh ** 2 * (
                -2. * eta * (R - Rs) * tanh + 3. / Rc * (tanh ** 2 - 1.0)))[:, None] * dR
        # G
        for jelem in elements:
            G[i, ifeat[elements[0]][jelem]] = g[indices[jelem]].sum()
        # dG
        for j, jelem in enumerate(atoms.get_chemical_symbols()):
            dG[i, ifeat[elements[0]][jelem], j] = dg.take(indices[j], 0).sum(0)
    return G, dG


def type4(ifeat, atoms, elements, Rc, eta, lambda_, zeta):
    natom = len(atoms)
    size = len(elements) * (1 + len(elements)) // 2
    G = np.zeros((natom, size))
    dG = np.zeros((natom, size, natom, 3))
    for i, indices, R, tanh, dR, cos, dcos in neighbour_info(atoms, Rc):
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
        for jelem in elements:
            G[i, ifeat[jelem][jelem]] = g.take(indices[jelem], 0).take(indices[jelem], 1).sum() / 2.0
        for jelem, kelem in combinations(elements, 2):
            G[i, ifeat[jelem][kelem]] = g.take(indices[jelem], 0).take(indices[kelem], 1).sum()
        # dG
        for (j, jelem), kelem in product(enumerate(atoms.get_chemical_symbols()), elements):
            dG[i, ifeat[jelem][kelem], j] = dg.take(indices[j], 0).take(indices[kelem], 1).sum((0, 1))
    return G, dG
