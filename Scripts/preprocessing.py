# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:50:15 2022

@author: Stevo
"""

import os
import numpy as np

# FIrst part - subsampling, since we do not want to work with so many vertices (over 20k)

# Load data:
os.chdir(r'C:\Users\User\Data')    
neutral = np.load('neutral.npy')
weights = np.load('weights.npy')
meshes  = np.load('meshes.npy')-neutral
deltas  = np.load('deltas.npy').T
bs1     = np.load('bs1.npy')
keys1   = np.load('keys1.npy')
N,n     = meshes.shape

# split by the coordinates x,y,z and take only every fifth vertex:
xs,ys,zs = neutral[::3],neutral[1::3],neutral[2::3]
xs,ys,zs = xs[::5],ys[::5],zs[::5]
d1,d2,d3 = deltas[::3], deltas[1::3],deltas[2::3]
d1,d2,d3 = d1[::5],d2[::5],d3[::5]
offs = np.mean(d1**2+ d2**2+ d3**2,1)
# now the threshold is used to remove the vertices of the neck, and it depends on the character:
threshold = 145 # for Omar
threshold = 135 # for Danielle
threshold = 145 # for Myles
indices1 = np.where(zs>threshold)[0]
xs_n,ys_n,zs_n = xs[indices1], ys[indices1], zs[indices1]
offs2 = offs[indices1]
# take only 4000 most active vertices:
indices2 = np.argsort(offs2)[-4000:]
xs_n,ys_n,zs_n = xs_n[indices2],ys_n[indices2],zs_n[indices2]

indices = indices1[indices2]*5
np.save('a1_indices.npy',indices)
indx = []
for i in indices:
    indx.append(i*3)
    indx.append(i*3+1)
    indx.append(i*3+2)
indx = np.array(sorted(indx))
neutral = neutral[indx]
deltas = deltas[indx]
meshes = meshes[:,indx]
bs1 = bs1[:,indx]
np.save('a1_neutral.npy',neutral)
np.save('a1_deltas.npy',deltas)
np.save('a1_meshes.npy',meshes)
np.save('a1_weights.npy',weights)
np.save('a1_bs1.npy',bs1)
np.save('a1_keys1.npy',keys1)

# Second part - transfer corrective terms into quadratic matrices D

n,m = deltas.shape
def Quadratic_term(vtx,m,keys1,bs1):
    ''' 
    Function that returns D_chs - i.e. tensor that contains matrices D s.t. we
    can express the rig function in a vertex in a standard quadratic form:
        f_i(c) = b_0[i] + B[i].dot(c) + c^T.dot(D).dot(c) 
    Parameters:
        vtx - the index of a vertex
          m - number of controllers
      keys1 - tuples of indices of the blendshapes that have a corrective term
        bs1 - corrective blendshapes for corresponding pairs from keys1
    '''
    D = np.zeros((m,m))
    for i in range(len(keys1)):
        tpl = keys1[i]
        D[tpl[0],tpl[1]] = bs1[i][vtx]/2
        D[tpl[1],tpl[0]] = bs1[i][vtx]/2
    return D

D_chs = np.array([Quadratic_term(vtx,m,keys1,bs1) for vtx in range(n)])
eig_min,eig_max = [],[]
sigma = []
for i in range(n):
    values = np.linalg.eigh(D_chs[i])[0]
    eig_min.append(np.min(values))
    eig_max.append(np.max(values))
    sigma.append(np.max(np.linalg.svd(D_chs[i])[1]))

np.save('a1_eigen_max.npy',np.array(eig_max))
np.save('a1_eigen_min.npy',np.array(eig_min))
np.save('a1_sigma_max.npy',np.array(sigma))
