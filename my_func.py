# -*- coding: utf-8 -*-

import numpy as np

# calculate interatomic vectors and distances
### input
# extend: Atoms Object
### output
# r: numpy array (natom x 8*natom x 3)
# R: numpy array (natom x 8*ntaom)
def vector_ij(extend, natom):
    r = np.zeros((natom, len(extend), 3))
    R = np.zeros((natom, len(extend)))
    for i in range(natom):
        r[i] = extend.get_distances(i, range(len(extend)), mic=True, vector=True)
        R[i] = extend.get_distances(i, range(len(extend)), mic=True)
    return r, R

# calculate the symmetry functions and derivatives of it
### input
# atoms_objs: list of Atoms Object
# Rcs,Rss,etas: list of float
### output
# Gs: numpy array (nsample x natom x gnum)
# dGs: numpy array (nsample x natom x 3*natom * gnum)
def symmetric_func(atoms_objs, natom, nsample, gnum, Rcs, Rss, etas):
    Gs = np.empty((nsample, natom, gnum)) # nsample x natom x gnum 個の配列
    dGs = np.empty((nsample, natom, 3*natom, gnum)) # nsample x natom x 3*natom x gnum 個の配列
    for m in range(nsample):
        extend = atoms_objs[m].repeat(2)
        r, R = vector_ij(extend, natom)
        # append transposed G and dG to Gs and dGs later
        G = np.empty((gnum, natom)) # gnum x natom
        dG = np.empty((gnum, 3*natom, natom)) # gnum x 3*natom x natom
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
                    dG_k = np.zeros((natom,3*natom))
                    for i in range(natom):
                        for j in range(len(extend)):
                            # if Rij is longer than cut-off, leave at 0
                            if R[i][j] > Rc or i == j:
                                pass
                            else:
                                # dRij: ∂Rij / ∂rkα
                                dRij = np.zeros((natom, 3))
                                for l in range(natom):
                                    if l == i:
                                        dRij[l] = - r[i][j] / R[i][j]
                                    elif l == j:
                                        dRij[l] = r[i][j] / R[i][j]
                                dRij = dRij.reshape(3*natom)
                                
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
