# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:03:10 2021

@author: syj
test for big_diagnoal_matrix function and operations on big_diagnoal_matrix
"""

import numpy as np
import big_diagnoal_matrix
import operations_on_big_diagnoal_matrix as operations

#creat diagonal_elements for 2*2 matrix blocks, each block is a 3*3 diagnoal matrix
diagonal_elements = np.random.randint(0,10,(4,3))
print(diagonal_elements)

#creat big_diagnoal_matrix
matrix_blocks = big_diagnoal_matrix.diagonal_matrix_blocks(2, diagonal_elements)
print(matrix_blocks)

#creat random matrix
b = np.random.randint(0,10,(6,4))
bb = np.random.randint(0,10,(4,6))

#perform matrix multiplication
result = operations.matrix_multiplications_for_big_diagnoal_matrix(matrix_blocks, b, 2, 3)
result2 = operations.matrix_multiplications_for_big_diagnoal_matrix(bb, matrix_blocks, 2, 3)

#perform matrix inverse
