# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:15:46 2021

@author: Administrator
"""
import numpy as np

def MatrixMultiply(a, b):
    c = np.zeros((a.shape[0], b.shape[1]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(a.shape[1]):
                c[i][j]=c[i][j]+a[i][k]*b[k][j]
    return c
                
def matrix_multiplications_for_big_diagnoal_matrix(a, b, num_block_order, num_order):
    '''
    Parameters
    ----------
    a : 2-D array
    b : 2-D array
    num_block_order : int
        size for big block eg: number of block will be (num_block_order * num_block_order).
    num_order : int
        size for square matrix eg: size of matrix will be (num_order * num_order).
    Returns
    -------
    c : 2-D array.
    '''
    c = np.zeros((a.shape[0], b.shape[1]))
    
    if a.shape[0] == a.shape[1]:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                for k in range(num_block_order):
                    c[i][j]=c[i][j]+a[i][(k-1)* num_order + i%num_order]*b[(k-1)* num_order + i%num_order][j]
    elif b.shape[0] == b.shape[1]:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                for k in range(num_block_order):
                    c[i][j]=c[i][j]+a[i][(k-1)* num_order + j%num_order]*b[(k-1)* num_order + j%num_order][j]
    
    return c

# def matrix_inverse_for_big_diagnoal_matrix():