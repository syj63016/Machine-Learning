# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:23:24 2021

@author: Administrator
"""
import numpy as np

def diagonal_matrix(num_order, diagonal_elements):
    '''
    Parameters
    ----------
    num_order : int
        size for square matrix eg: size of matrix will be (num_order * num_order).
    diagonal_elements : 1-D array
        one_dimention_array represents diagonal_elements in 1-D form. length of diagonal_elements equals num_order.
    Returns
    -------
    matrix: 2-D array
        num_order * num_order digonal_matrix
    '''
    #initialize matrix
    matrix = np.zeros((num_order, num_order))
    # put elements in diagnoal
    for i in range(num_order):
        matrix[i][i] = diagonal_elements[i]
    return matrix

#construct diagonal_matrix_block

def diagonal_matrix_blocks(num_block_order, diagonal_elements):
    '''
    Parameters
    ----------
    num_block_order : int
        size for big block eg: number of block will be (num_block_order * num_block_order).
    diagonal_elements : 2-D array
        2-D array represents diagonal_elements in 2-D form.
        each row represents diagonal_elements for 1 diagonal_matrix.
    Returns
    -------
    diagonal_matrix_block : 2-D array
        diagonal_matrix_blocks consists of num_block_order * num_block_order diagonal_matrix.
        size of diagonal_matrix depends on the length of each row of diagonal_elements.
    '''
    # get size of each diagonal_matrix
    num_order = diagonal_elements[0, :].shape[0]
    #initialize diagonal_matrix_blocks
    matrix_blocks = np.zeros((num_order * num_block_order, num_order * num_block_order))
    # fill out diagnoal for each diagonal_matrix in order
    for i in range(num_block_order):
        for j in range(num_block_order):
            for k in range(num_order):
                matrix_blocks[i * num_order + k][j * num_order + k] = diagonal_matrix(
                                                                                    num_order, 
                                                                                    diagonal_elements[i*num_block_order + j,:])[k,k]
    return matrix_blocks