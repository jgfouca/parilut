#! /usr/bin/env python3

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from scipy.sparse import bsr_matrix, csr_matrix

def pprint(mtx):
    dim = mtx.shape[0]
    for row in mtx:
        for item in row:
            print("{:.5f}".format(item), end=" ")

        print()

A = scipy.array([
    [10.00, 0.00, 0.30, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00],
    [0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.00, 0.00],
    [0.00, 0.00, 12.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [5.00, 0.00, 0.00, 13.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    [4.00, 0.00, 0.00, 0.00, 14.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 3.00, 0.00, 0.00, 0.00, 15.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 7.00, 0.00, 0.00, 0.00, 16.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 6.00, 5.00, 0.00, 0.00, 17.00, 0.00],
    [0.00, 0.00, 0.00, 2.00, 2.50, 0.00, 0.00, 0.00, 18.00]
])
P, L, U = scipy.linalg.lu(A)

print("A:")
pprint(A)

print("P:")
pprint(P)

print("L:")
pprint(L)

print("U:")
pprint(U)

A_csr = csr_matrix(A)
A_bsr = bsr_matrix(A, blocksize=(3, 3))

slu_csr = scipy.sparse.linalg.spilu(A_csr, fill_factor=2)
print("L_csr:")
pprint(slu_csr.L.toarray())
print("U_csr:")
pprint(slu_csr.U.toarray())

slu_bsr = scipy.sparse.linalg.spilu(A_bsr, fill_factor=2)
print("L_bsr:")
pprint(slu_bsr.L.toarray())
print("U_bsr:")
pprint(slu_bsr.U.toarray())
