# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:27:11 2022

@author: Stevo

We have an option for sequential and for parallel implementation.
"""

import numpy as np
from joblib import Parallel, delayed
from CubicEquationSolver import single_cubic_one

def quadratic_rig(C,deltas,bs1,keys1):
    ''' Computes a quadratic rig (approximation) given a weight vector C.
    Parameters:
               C - vector of activation weights.
          deltas - nxm matrix; delta offset values.
             bs1 - m1xn matrix, where m1 is number of rig corrections of first 
                   order; corrective meshes 
           keys1 - m1*2 matrix; tuples of indices that invoke any of the 
                   corrective meshes from bs1
    '''
    return deltas.dot(C) + bs1.T.dot(C[keys1[:,0]]*C[keys1[:,1]])

def objective_function(rig_mesh,target_mesh,lmbd,C):
    ''' Gives a value of the objective cost. Function doesn't consider any 
    regularization term, only the squared norm of the offset.
    Parameters:
           rig_mesh - vector; mesh obtained using a rig function.
        target_mesh - vector; ground-truth mesh.
               lmbd - scalar; reglarization parameter.
                  C - vector of the controller activation weights.
    '''
    return np.linalg.norm(rig_mesh-target_mesh)**2 + lmbd*np.sum(C)

def h_single_ctr(ctr,C,keys1,bs1,h):
    indices1 = np.where(keys1[:,1]==ctr)[0]
    indices2 = np.where(keys1[:,0]==ctr)[0]
    h[:,ctr] += (C[keys1[indices1,0]]).dot(bs1[indices1])
    h[:,ctr] += (C[keys1[indices2,1]]).dot(bs1[indices2])

def coefficients_sequential(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,bs1,keys1,lmbd):
    ''' Computes coefficients for the upper bound polinomial to be minimized.
    Parametri:
     target_mesh - vector; ground-truth mesh.
               C - vector of the controller activation weights.       
       eig_max_D - vector; the largest eigenvalues for each D form D_sparse.
       eig_min_D - vector; the smallest eigenvalues for each D form D_sparse.
         sigma_D - vector; the largest singular values for each D from D_sparse.
    '''   
    g = np.dot(deltas, C) + np.dot(bs1.T,(C[keys1[:,0]]*C[keys1[:,1]])) - target_mesh
    h = 0. + deltas
    for ctr in range(m):
        h_single_ctr(ctr,C,keys1,bs1,h)
    coef1 = 2*g.dot(h) + lmbd
    coef2 = np.zeros(n)
    coef2[g>0] += eig_max_D[g>0]
    coef2[g<0] += eig_min_D[g<0]
    coef2 = 2*(g.dot(coef2) + np.sum(h**2))
    coef4 = 2*m*np.sum(sigma_D**2)
    return coef1,coef2,coef4

def coefficients_parallel(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,bs1,keys1,lmbd,n_jobs):
    ''' Computes coefficients for the upper bound polinomial to be minimized.
    Parametri:
     target_mesh - vector; ground-truth mesh.
               C - vector of the controller activation weights.       
       eig_max_D - vector; the largest eigenvalues for each D form D_sparse.
       eig_min_D - vector; the smallest eigenvalues for each D form D_sparse.
         sigma_D - vector; the largest singular values for each D from D_sparse.
          n_jobs - int; the maximum number of concurrently running jobs.
    '''   
    g = np.dot(deltas, C) + np.dot(bs1.T,(C[keys1[:,0]]*C[keys1[:,1]])) - target_mesh
    h = 0. + deltas
    Parallel(n_jobs=n_jobs,backend="threading")(
              delayed(h_single_ctr)
              (ctr,C,keys1,bs1,h)
              for ctr in range(m))
    coef1 = 2*g.dot(h) + lmbd
    coef2 = np.zeros(n)
    coef2[g>0] += eig_max_D[g>0]
    coef2[g<0] += eig_min_D[g<0]
    coef2 = 2*(g.dot(coef2) + np.sum(h**2))
    coef4 = 2*m*np.sum(sigma_D**2)
    return coef1,coef2,coef4

def increment_single_ctr(ctr,C,coef1,coef2,coef4,increment):
    # first check the borders
    C_ctr, coef1_ctr = C[ctr], coef1[ctr]
    root = 0. - C_ctr
    root_value = coef1_ctr*root + coef2*(root**2) + coef4*(root**4)
    root1 = 1 - C_ctr
    root_value1 = coef1_ctr*root1 + coef2*(root1**2) + coef4*(root1**4)
    if root_value1 < root_value:
        root = root1
        root_value = root_value1
    # then extreme value(s)
    root2 = single_cubic_one(4*coef4, 0, 2*coef2, coef1_ctr)
    root_value2 = coef1_ctr*root2 + coef2*(root2**2) + coef4*(root2**4)
    if root_value2 < root_value:
        root = root2
        root_value = root_value2                                          
    increment[ctr] += root    
    
def increment_sequential(C,m,coef1,coef2,coef4):
    ''' I take coefficients for the polinomial, and then visit one controller 
    at a time, to find an increment that minimizes the upper bound - this is an
    unconstrained minimization problem.
    Parameters:
              coef1,coef2,coef4 - coefficients for the polinomial 
                  obtained from the function 'terms_and_coefficients'. Scalars,
                  except for the coef1, which is a vector.
    '''
    increment = np.zeros(m)
    for ctr in range(m):
        increment_single_ctr(ctr,C,coef1,coef2,coef4,increment)
    return increment

def increment_parallel(C,m,coef1,coef2,coef4,n_jobs):
    ''' I take coefficients for the polinomial, and then visit one controller 
    at a time, to find an increment that minimizes the upper bound - this is an
    unconstrained minimization problem.
    Parameters:
              coef1,coef2,coef4 - coefficients for the polinomial 
                  obtained from the function 'terms_and_coefficients'. Scalars,
                  except for the coef1, which is a vector.
    '''
    increment = np.zeros(m)
    Parallel(n_jobs=n_jobs,backend="threading")(
              delayed(increment_single_ctr)
              (ctr,C,coef1,coef2,coef4,increment)
              for ctr in range(m))
    return increment

def minimization(num_iter,C,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,n,m,tolerance,lmbd,bs1,keys1):
    ''' We use previously define functions to minimize the upper bound function.
    In each iteration we collect the values of the objective and the bound 
    function. Algorithm terminates if we exceed maximum number of iterations or
    the change in objective is under some predefined tolerance threshold.
    Parameters:
           num_iter - integer; number of iterations before we terminate the 
                      algorithm.
                  C - vector of the controller activation weights.
             deltas - nxm matrix; delta offset values.
        target_mesh - vector; ground-truth mesh.
          eig_max_D - vector; the largest eigenvalues for each D form D_sparse.
          eig_min_D - vector; the smallest eigenvalues for each D form D_sparse.
            sigma_D - vector; the largest sing. values for each D form D_sparse.
                  n - scalar; number of coordinates in the face.
                  m - scalar; number of controllers for the model.
          tolerance - scalar; threshold value. If objective cost or the value 
                      of the upper bound between two consecutive values change 
                      less than tolerance, we terminate the algorithm.
               lmbd - scalar; reglarization parameter.
                bs1 - m1xn matrix, where m1 is number of rig corrections of first 
                      order; corrective meshes 
              keys1 - m1*2 matrix; tuples of indices that invoke any of the 
                      corrective meshes from bs1
    '''
    bound_values = []
    for i in range(num_iter):
        coef1,coef2,coef4 = coefficients_sequential(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,bs1,keys1,lmbd)
        increment = increment_sequential(C,m,coef1,coef2,coef4)        
        bound_val = coef1.dot(increment) + coef2*(np.linalg.norm(increment)**2) + coef4*(np.sum(increment**4))
        bound_values.append(bound_val)
        C += increment
        if i>1 and (np.abs(bound_values[-1]-bound_values[-2])<tolerance):
            break
    return C, i

def minimization_parallel(num_iter,C,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,n,m,tolerance,lmbd,bs1,keys1,n_jobs):
    ''' We use previously define functions to minimize the upper bound function.
    In each iteration we collect the values of the objective and the bound 
    function. Algorithm terminates if we exceed maximum number of iterations or
    the change in objective is under some predefined tolerance threshold.
    Parameters:
           num_iter - integer; number of iterations before we terminate the 
                      algorithm.
                  C - vector of the controller activation weights.
             deltas - nxm matrix; delta offset values.
        target_mesh - vector; ground-truth mesh.
          eig_max_D - vector; the largest eigenvalues for each D form D_sparse.
          eig_min_D - vector; the smallest eigenvalues for each D form D_sparse.
            sigma_D - vector; the largest sing. values for each D form D_sparse.
                  n - scalar; number of coordinates in the face.
                  m - scalar; number of controllers for the model.
          tolerance - scalar; threshold value. If objective cost or the value 
                      of the upper bound between two consecutive values change 
                      less than tolerance, we terminate the algorithm.
               lmbd - scalar; reglarization parameter.
                bs1 - m1xn matrix, where m1 is number of rig corrections of first 
                      order; corrective meshes 
              keys1 - m1*2 matrix; tuples of indices that invoke any of the 
                      corrective meshes from bs1
             n_jobs - int; the maximum number of concurrently running jobs.
    '''
    bound_values = []
    for i in range(num_iter):
        coef1,coef2,coef4 = coefficients_parallel(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,bs1,keys1,lmbd,n_jobs)
        increment = increment_parallel(C,m,coef1,coef2,coef4,n_jobs)        
        bound_val = coef1.dot(increment) + coef2*(np.linalg.norm(increment)**2) + coef4*(np.sum(increment**4))
        bound_values.append(bound_val)
        C += increment
        if i>1 and (np.abs(bound_values[-1]-bound_values[-2])<tolerance):
            break
    return C, i

    
    
