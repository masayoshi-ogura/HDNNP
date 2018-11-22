# -*- coding: utf-8 -*-

import numpy as np
from phonopy.units import VaspToCm

dimensions = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
options = {
    'factor': VaspToCm,
    'symprec': 1.0e-4,
}
distance = 0.01


def callback(phonopy):
    # variables
    points = 101
    point_symmetry = [
        [0.0, 0.0, 0.0],  # Gamma
        [1.0 / 3, 1.0 / 3, 0.0],  # K
        [0.5, 0.0, 0.0],  # M
        [0.0, 0.0, 0.0],  # Gamma
        [0.0, 0.0, 0.5],  # A
        [1.0 / 3, 1.0 / 3, 0.5],  # H
        [0.5, 0.0, 0.5],  # L
        [0.0, 0.0, 0.5],  # A
    ]
    mesh = [8, 8, 8]
    labels = ['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A']

    bands = [np.concatenate([np.linspace(si, ei, points).reshape(-1, 1) for si, ei in zip(s, e)], axis=1)
             for s, e in zip(point_symmetry[:-1], point_symmetry[1:])]

    # phonopy API
    phonopy.set_mesh(mesh)
    phonopy.set_total_DOS()
    phonopy.set_band_structure(bands, is_band_connection=True)
    plt = phonopy.plot_band_structure_and_dos(labels=labels)
    postprocess_axes(plt.gcf().axes[0])
    return plt


def postprocess_axes(ax):
    ax.grid(axis='x')
    xticks = ax.get_xticks()
    ax.set_ylabel('Frequency [cm$^{-1}$]')

    # experimental result measured by IXS and Raman is obtained from
    # Phonon Dispersion Curves in Wurtzite-Structure GaN Determined by Inelastic X-Ray Scattering
    # PRL vol.86 #5 2001/1/29
    # @Gamma, Raman
    ax.scatter([xticks[0]] * 6 + [xticks[3]] * 6,
               [144.2, 533.5, 560.0, 569.2, 739.3, 746.6] * 2,
               marker='o', s=50, facecolors='none', edgecolors='blue')
    # @Gamma, IXS
    ax.scatter([xticks[0]] * 3 + [xticks[3]] * 3,
               [329, 692, 729] * 2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @A, IXS
    ax.scatter([xticks[4]] * 2 + [xticks[7]] * 2,
               [231, 711] * 2,
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @M, IXS
    ax.scatter([xticks[2]] * 5,
               [137, 184, 193, 238, 576],
               marker='o', s=50, facecolors='none', edgecolors='red')
    # @K, IXS
    ax.scatter([xticks[1]] * 2,
               [215, 614],
               marker='o', s=50, facecolors='none', edgecolors='red')
