# -*- coding: utf-8 -*-

import numpy as np

# calculate interatomic vectors and distances
### input
# extend: Atoms Object
### output
# r: numpy array (NATOM x 8*NATOM x 3)
# R: numpy array (NATOM x 8*NATOM)
def vector_ij(extend, NATOM):
    r = np.zeros((NATOM, len(extend), 3))
    R = np.zeros((NATOM, len(extend)))
    for i in range(NATOM):
        r[i] = extend.get_distances(i, range(len(extend)), mic=True, vector=True)
        R[i] = extend.get_distances(i, range(len(extend)), mic=True)
    return r, R

# calculate the symmetry functions and derivatives of it
### input
# atoms_objs: list of Atoms Object
# Rcs,Rss,etas: list of float
### output
# Gs: numpy array (NSAMPLE x NATOM x NINPUT)
# dGs: numpy array (NSAMPLE x NATOM x 3*NATOM * NINPUT)
def symmetric_func(atoms_objs, NATOM, NSAMPLE, NINPUT, Rcs, Rss, etas):
    Gs = np.empty((NSAMPLE, NATOM, NINPUT)) # NSAMPLE x NATOM x NINPUT 個の配列
    dGs = np.empty((NSAMPLE, NATOM, 3*NATOM, NINPUT)) # NSAMPLE x NATOM x 3*NATOM x NINPUT 個の配列
    for m in range(NSAMPLE):
        extend = atoms_objs[m].repeat(2)
        r, R = vector_ij(extend, NATOM)
        # append transposed G and dG to Gs and dGs later
        G = np.empty((NINPUT, NATOM)) # NINPUT x NATOM
        dG = np.empty((NINPUT, 3*NATOM, NATOM)) # NINPUT x 3*NATOM x NATOM
        k = 0
        for Rc in Rcs:
            for Rs in Rss:
                for eta in etas:
                    filter1 = R > Rc
                    filter2 = R == 0.0
                    filter = np.logical_or(filter1,filter2)
                    tanh = np.tanh(1 - R / Rc)
                    f = np.exp(-eta * (R - Rs) ** 2)
                    
                    # G1
                    #gij = tanh ** 3
                    ###############
                    
                    # G2
                    gij = f * (tanh **3)
                    ###########
                    
                    gij[filter] = 0
                    G[k] = np.dot(gij, np.ones(len(extend)))
                    
                    # calculate derivatives
                    dG_k = np.zeros((NATOM,3*NATOM))
                    for i in range(NATOM):
                        for j in range(len(extend)):
                            # if Rij is longer than cut-off, leave at 0
                            if R[i][j] > Rc or i == j:
                                pass
                            else:
                                # dRij: ∂Rij / ∂rkα
                                dRij = np.zeros((NATOM, 3))
                                for l in range(NATOM):
                                    if l == i:
                                        dRij[l] = - r[i][j] / R[i][j]
                                    elif l == j:
                                        dRij[l] = r[i][j] / R[i][j]
                                dRij = dRij.reshape(3*NATOM)
                                
                                # G1
                                #dgij =  (-3 / Rc) * (tanh[i][j] ** 2) * (1 - tanh[i][j] ** 2) * dRij
                                ###################
                                
                                # G2
                                dgij = - f[i][j] * (tanh[i][j] ** 2) * ((2 * tanh[i][j] * eta * (R[i][j] - Rs)) + (3 * (1 - (tanh[i][j] ** 2)) / Rc)) * dRij
                                ###########
                                
                                dG_k[i] += dgij
                    dG[k] = dG_k.T
                    k+=1
        Gs[m] = G.T
        dGs[m] = dG.T
    return Gs, dGs
