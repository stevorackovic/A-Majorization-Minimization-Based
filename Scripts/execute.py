# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:21:34 2022

@author: Stevo

Executes our method with different choices of regularization parameter and using a pseudoinverse initialization.
"""

import numpy as np
from MM_back import minimization
import time

###############################################################################
# Load data:
deltas    = np.load('a1_deltas.npy')
meshes    = np.load('a1_meshes.npy')
bs1       = np.load('a1_bs1.npy')
keys1     = np.load('a1_keys1.npy')
eig_max_D = np.load('a1_eigen_max.npy')
eig_min_D = np.load('a1_eigen_min.npy')
sigma_D   = np.load('a1_sigma_max.npy')

n,m = deltas.shape
N,n = meshes.shape

###############################################################################
# Regularization parameter values:
lmbds = [0.001,0.01,0.5,2.5,5,7.5,10,20,50]

num_iter = 200
tolerance = 0.0001
for i in range(len(lmbds)):
    lmbd = lmbds[i]
    print('Solving for lambda = ',lmbd)
    Bpsd = (np.linalg.inv(deltas.T.dot(deltas) + lmbd*np.eye(m))).dot(deltas.T)
    Predictions = []
    start_time = time.time()
    for frame in range(N):
        target_mesh = meshes[frame]
        C = Bpsd.dot(target_mesh)
        C[C<0] = 0
        C[C>1] = 1
        C, itr = minimization(num_iter,C,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,n,m,tolerance,lmbd,bs1,keys1)
        Predictions.append(C)
    end_time = time.time()
    vrm = end_time-start_time
    print('Average time - ', vrm/N)
    np.save('a1_MMpsd_pred_lmbd_'+str(lmbd)+'.npy',Predictions)
