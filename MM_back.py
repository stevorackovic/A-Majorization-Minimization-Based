# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:54:53 2022

@author: Stevo

Contains functions for our model that are later invoked in the script for the execution
"""


import numpy as np


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

def terms_and_coefficients(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,lmbd,bs1,keys1):
    ''' Computes coefficients for the upper bound polinomial to be minimized.
    Parametri:
        target_mesh - vector; ground-truth mesh.
                  C - vector of the controller activation weights.       
          eig_max_D - vector; the largest eigenvalues for each D form D_sparse.
          eig_min_D - vector; the smallest eigenvalues for each D form D_sparse.
            sigma_D - vector; the largest singular values for each D from 
                      D_sparse
               lmbd - scalar; reglarization parameter.
    '''
    term_p = quadratic_rig(C,deltas,bs1,keys1) - target_mesh
    coef0 = np.linalg.norm(term_p)**2 + lmbd*np.sum(C)
    term_q = 0. + deltas
    for ctr in range(m):
        indices1 = np.where(keys1[:,0]==ctr)
        indices2 = keys1[indices1,1]
        term_q[:,ctr] += (C[indices2].dot(bs1[indices1])/2)[0,:]
    coef1 = 2*term_p.dot(term_q) + lmbd
    term_r = np.zeros(n)
    term_r[term_p>0] += eig_max_D[term_p>0]
    term_r[term_p<0] += eig_min_D[term_p<0]
    coef2 = 2*(np.sum(term_p*term_r) + np.sum(term_q**2))
    coef4 = 2*m*np.sum(sigma_D**2)
    return coef0,coef1,coef2,coef4

def compute_increment(C,m,coef0,coef1,coef2,coef4):
    ''' It takes coefficients for the polinomial, and then visit one controller 
    at a time, to find an increment that minimizes the upper bound.
    Parameters:
        coef0,coef1,coef2,coef4 - coefficients for the polinomial 
                 obtained from the function 'terms_and_coefficients'. Scalars,
                 except for the coef1, which is a vector.
    '''
    increment = np.zeros(m)
    for ctr in range(m):
        min_x = -C[ctr] # check the borders (of the feasible set) first
        min_y = coef4*(min_x**4) + coef2*(min_x**2) + coef1[ctr]*min_x + coef0
        candidate_x = 1-C[ctr]
        candidate_y = coef4*(candidate_x**4) + coef2*(candidate_x**2) + coef1[ctr]*candidate_x + coef0
        if candidate_y < min_y:
            min_y = candidate_y
            min_x = candidate_x
        # then we check potential extreme values.
        # If it is within the feasible set (-C, 1-C), I check if the value is lower than at the border.
        # Use a closed form solution
        term0 = 27*((4*coef4)**2)*coef1[ctr]
        term1 = (term0)**2 - 4*((-3*(4*coef4)*(2*coef2))**3)
        if term1 >= 0: # if this is negative, a root is complex, so we dismiss it 
            candidate_x = -1/(3*(4*coef4))*np.cbrt(.5*(term0+np.sqrt(term1))) - 1/(3*(4*coef4))*np.cbrt(.5*(term0-np.sqrt(term1)))
            if candidate_x > -C[ctr] and candidate_x < 1-C[ctr]:
                candidate_y = coef4*(candidate_x**4) + coef2*(candidate_x**2) + coef1[ctr]*candidate_x + coef0
                if candidate_y < min_y:
                    min_y = candidate_y
                    min_x = candidate_x               
        increment[ctr] += min_x
    return increment

def minimization(num_iter,C,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,n,m,tolerance,lmbd,bs1,keys1,return_residuals=False):
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
   return_residuals - boolean; specifies if the function will additionally 
                      return the values of the objective (and the bound) for
                      each iteration. Deafult False.
    '''
    if return_residuals==False:
        bound_values = []
        for i in range(num_iter):
            coef0,coef1,coef2,coef4 = terms_and_coefficients(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,lmbd,bs1,keys1)
            increment = compute_increment(C,m,coef0,coef1,coef2,coef4)        
            bound_val = coef0 + coef1.dot(increment) + coef2*(np.linalg.norm(increment)**2) + coef4*(np.sum(increment**4))
            bound_values.append(bound_val)
            C += increment
            if i>1 and (np.abs(bound_values[-1]-bound_values[-2])<tolerance):
                break
        return C, i
    # this is the case when we want also to return the values of the objective function and the bound for each iteration:
    else:
        bound_values, objective_values = [],[]
        fidelity = [np.linalg.norm(deltas.dot(C) + bs1.T.dot(C[keys1[:,0]]*C[keys1[:,1]])-target_mesh)**2]
        regularization = [np.sum(C)]
        for i in range(num_iter):
            coef0,coef1,coef2,coef4 = terms_and_coefficients(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,lmbd,bs1,keys1)
            increment = compute_increment(C,m,coef0,coef1,coef2,coef4)        
            bound_val = coef0 + coef1.dot(increment) + coef2*(np.linalg.norm(increment)**2) + coef4*(np.sum(increment**4))
            bound_values.append(bound_val)
            C += increment
            rig_mesh = quadratic_rig(C,deltas,bs1,keys1)
            objective_val = objective_function(rig_mesh,target_mesh,lmbd,C)
            objective_values.append(objective_val)
            fidelity.append(np.linalg.norm(rig_mesh-target_mesh)**2)
            regularization.append(np.sum(C))
            if i>1 and (np.abs(bound_values[-1]-bound_values[-2])<tolerance):
                break
        return C, bound_values, objective_values, fidelity, regularization
    
    
