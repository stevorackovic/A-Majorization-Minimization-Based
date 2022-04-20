# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:57:57 2022

@author: Stevo

Borrowed from https://github.com/NKrvavica/fqs/blob/master/fqs.py
"""

import math
from numba import jit

@jit(nopython=True)
def single_cubic_one(a0, b0, c0, d0):
    ''' Analytical closed-form solver for a single cubic equation
    (3rd order polynomial), gives only one real root.
    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::
            a0*x^3 + b0*x^2 + c0*x + d0 = 0
    Returns
    -------
    roots: float
        Output data is a real root of a given polynomial.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        if x.real >= 0:
            return x**third
        else:
            return -(-x)**third

    if f == g == h == 0:
        return -cubic_root(c)

    elif h <= 0:
        j = math.sqrt(-f)
        k = math.acos(-0.5*g / (j*j*j))
        m = math.cos(third*k)
        return 2*j*m - a13

    else:
        sqrt_h = math.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        return S_plus_U - a13