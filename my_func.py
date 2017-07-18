# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np

import hdnnp

# 原子間ベクトルとその長さ
def vector_ij(extend, natom):
    r = np.zeros((natom, len(extend), 3))
    R = np.zeros((natom, len(extend)))
    for i in range(natom):
        r[i] = extend.get_distances(i, range(len(extend)), mic=True, vector=True)
        R[i] = extend.get_distances(i, range(len(extend)), mic=True)
    return r, R

# 対称関数と全方向に関する微分
def symmetric_func(atoms_obj, extend, natom, Rc, Rs, eta):
    r, R = vector_ij(extend, natom)
    filter = R > Rc
    tanh = np.tanh(1 - R / Rc)
    f = np.exp(-eta * (R - Rs) ** 2)
    
    # 一番簡単な対称関数
    #gij = tanh ** 3
    ###############
    
    # 次に難しいやつ
    gij = f * (tanh **3)
    ###########
    
    gij[filter] = 0
    G = np.dot(gij, np.ones(len(extend)))
    
    # 微分を求める
    dG = []
    for i in range(natom):
        dGi = np.zeros((natom, 3))
        for j in range(len(extend)):
            # RijがRc以上、またはiとjが同原子なら、ゼロ行列のまま
            if R[i][j] > Rc or i == j:
                pass
            else:
                # dRij: ∂Rij / ∂rkα
                # n×3行列
                # iとjが変わるたびに計算し直す必要がある？
                dRij = np.zeros((natom, 3))
                for k in range(natom):
                    if k == i:
                        dRij[k] = - r[i][j] / R[i][j]
                    elif k == j:
                        dRij[k] = r[i][j] / R[i][j]
                        
                # 対称関数は一番簡単やなつ
                #dgij =  (-3 / Rc) * (tanh[i][j] ** 2) * (1 - tanh[i][j] ** 2) * dRij
                ###################

                # 次に難しいやつ
                dgij = - f[i][j] * (tanh[i][j] ** 2) * ((2 * tanh[i][j] * eta * (R[i][j] - Rs)) + (3 * (1 - (tanh[i][j] ** 2)) / Rc)) * dRij
                ###########
                dGi += dgij
        dG.append(dGi)
    
    return G, dG

# RMSEを計算する
def calc_RMSE(comm, rank, nnp, natom, nsample, dataset):
    E_RMSE = 0.0
    F_RMSE = np.zeros((natom,3))
    for n in range(nsample):
        Et = dataset[n][0]
        Frt = dataset[n][1]
        G = dataset[n][2]
        dG = dataset[n][3]
        E_out = hdnnp.query_E(comm, nnp, G[rank], natom)
        E_RMSE += (Et - E_out[0]) ** 2
        for k in range(natom):
            for l in range(3):
                  F_rout = hdnnp.query_F(comm, nnp, G[rank], dG[rank], natom)
                  F_RMSE[k][l] += (Frt[k][l] - F_rout[k][l]) ** 2
    E_RMSE /= nsample
    F_RMSE /= (nsample * natom * 3)
    
    return E_RMSE, F_RMSE
