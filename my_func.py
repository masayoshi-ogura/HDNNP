# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
#from quippy import *

import hdnnp

# 原子間ベクトルとその長さ
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

# 対称関数と全方向に関する微分
### input
# atoms_objs: list of Atoms Object
# Rcs,Rss,etas: list of float
### output
# Gs: numpy array (nsample x natom x gnum)
# dGs: numpy array (nsample x natom x 3*natom * gnum)
def symmetric_func(atoms_objs, natom, nsample, gnum, Rcs, Rss, etas):
    # Rc, Rs, etaを配列で受け取り、その相乗数gnumのG,dGを配列で返す
    Gs = np.empty((nsample, natom, gnum)) # nsample x natom x gnum 個の配列
    dGs = np.empty((nsample, natom, 3*natom, gnum)) # nsample x natom x 3*natom x gnum 個の配列
    for m in range(nsample):
        extend = atoms_objs[m].repeat(2)
        r, R = vector_ij(extend, natom)
        # 後で転置してGs,dGsに加える
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
                    
                    # 一番簡単な対称関数
                    #gij = tanh ** 3
                    ###############
                    
                    # 次に難しいやつ
                    gij = f * (tanh **3)
                    ###########
                    
                    gij[filter] = 0
                    G[k] = np.dot(gij, np.ones(len(extend)))
                    
                    # 微分を求める
                    dG_k = np.zeros((natom,3*natom))
                    for i in range(natom):
                        for j in range(len(extend)):
                            # RijがRc以上、またはiとjが同原子なら、ゼロ行列のまま
                            if R[i][j] > Rc or i == j:
                                pass
                            else:
                                # dRij: ∂Rij / ∂rkα
                                # n×3行列
                                # iとjが変わるたびに計算し直す必要がある？
                                dRij = np.zeros((natom, 3))
                                for l in range(natom):
                                    if l == i:
                                        dRij[l] = - r[i][j] / R[i][j]
                                    elif l == j:
                                        dRij[l] = r[i][j] / R[i][j]
                                dRij = dRij.reshape(3*natom)
                                        
                                # 対称関数は一番簡単やなつ
                                #dgij =  (-3 / Rc) * (tanh[i][j] ** 2) * (1 - tanh[i][j] ** 2) * dRij
                                ###################

                                # 次に難しいやつ
                                dgij = - f[i][j] * (tanh[i][j] ** 2) * ((2 * tanh[i][j] * eta * (R[i][j] - Rs)) + (3 * (1 - (tanh[i][j] ** 2)) / Rc)) * dRij
                                ###########
                                dG_k[i] += dgij
                    dG[k] = dG_k.T
                    k+=1
        Gs[m] = G.T
        dGs[m] = dG.T
    return Gs, dGs

# RMSEを計算する
### input
# comm, rank: MPI communicator, rank of the processor
# nnp: hdnnp.single_nnp instance
# dataset: list of following 4 objects
#          energy: float
#          forces: numpy array (3*natom)
#          G: numpy array (natom x gnum)
#          dG: numpy array (natom x 3*natom x gnum)
### output
# E_RMSE: float
# F_RMSE: float
def calc_RMSE(comm, rank, nnp, natom, nsample, dataset):
    E_MSE = 0.0
    F_MSE = 0.0
    for n in range(nsample):
        Et = dataset[n][0]
        Frt = dataset[n][1]
        G = dataset[n][2]
        dG = dataset[n][3]
        E_out = hdnnp.query_E(comm, nnp, G[rank], natom)
        F_rout = hdnnp.query_F(comm, nnp, G[rank], dG[rank], natom)
        E_MSE += (Et - E_out) ** 2
        F_MSE += np.sum((Frt - F_rout)**2)
    E_RMSE = math.sqrt(E_MSE / nsample)
    F_RMSE = math.sqrt(F_MSE / (nsample * natom * 3))
    
    return E_RMSE, F_RMSE

# デバッグ用にリストをファイルに書き出す関数を定義
### input
# file: file object
# list: list object
def output_list(file, list):
    for line in list:
        file.writelines(map(str, line))
        file.write('\n')
