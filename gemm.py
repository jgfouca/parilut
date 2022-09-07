#!/usr/bin/env python3

"""
Run general matrix matrix multiply. Time performance
"""

import argparse, sys, pathlib, time, random

from common import Matrix2DR, require

###############################################################################
class RMMatrix(object):
###############################################################################
    """
    A 2D uncompressed sparse matrix, row major, implemented as a list of lists

    Slow, only used for initialization and debugging
    """

    ###########################################################################
    def __init__(self, rows, cols, pct_nz=0, diag_lambda=None):
    ###########################################################################
        self._matrix = [0]*(rows*cols)
        self._rows   = rows
        self._cols   = cols

        for row in range(rows):
            for col in range(cols): # each row has num-cols items
                roll = random.randint(0, 100)
                nz = roll < pct_nz
                nz_val = random.uniform(0.0, 1.0)
                self[row,col] = nz_val if nz else 0.0 # Slow but that's OK for this class

        # Non-zero matrix must have at least 1 value in each row and col
        # so make unit diagonal
        if pct_nz > 0 or diag_lambda is not None:
            if rows == cols:
                if diag_lambda:
                    for i in range(rows):
                        self[i,i] = diag_lambda(i)
                else:
                    for i in range(rows):
                        self[i,i] = 1.0
            else:
                self[0,0] = random.uniform(0.0, 1.0)

    ###########################################################################
    def __str__(self):
    ###########################################################################
        result = ""
        for row in range(self.nrows()):
            for col in range(self.ncols()):
                result += "{:.2f} ".format(self[row,col])

            result += "\n"

        return result

    ###########################################################################
    def nrows(self): return self._rows
    ###########################################################################

    ###########################################################################
    def ncols(self): return self._cols
    ###########################################################################

    ###########################################################################
    def __getitem__(self, tup): return self._matrix[self.ncols()*tup[0] + tup[1]]
    ###########################################################################

    ###########################################################################
    def __setitem__(self, tup, val): self._matrix[self.ncols()*tup[0] + tup[1]] = val
    ###########################################################################

    ###########################################################################
    def __eq__(self, rhs):
    ###########################################################################
        if self.nrows() != rhs.nrows() or self.ncols() != rhs.ncols():
            return False

        for row in range(self.nrows()):
            for col in range(self.ncols()):
                if not near(self[row,col], rhs[row,col]):
                    return False

        return True

    ###########################################################################
    def __add__(self, rhs):
    ###########################################################################
        require(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        require(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

        result = RMMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[i,j] = self[i,j] + rhs[i,j]

        return result

    ###########################################################################
    def __sub__(self, rhs):
    ###########################################################################
        require(self.nrows() == rhs.nrows(), "Cannot subtract matrix, incompatible dims")
        require(self.ncols() == rhs.ncols(), "Cannot subtract matrix, incompatible dims")

        result = RMMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[i,j] = self[i,j] - rhs[i,j]

        return result

    ###########################################################################
    def __mul__(self, rhs):
    ###########################################################################
        if isinstance(rhs, float):
            result = RMMatrix(self.nrows(), self.ncols())

            for i in range(self.nrows()):
                for j in range(self.ncols()):
                    result[i,j] = self[i,j]*rhs
        else:

            require(self.ncols() == rhs.nrows(),
                    f"Cannot multiply matrix, incompatible dims. LHS cols={self.ncols()}, RHS rows={rhs.nrows()}")

            result = RMMatrix(self.nrows(), rhs.ncols())

            for i in range(self.nrows()):
                for j in range(rhs.ncols()):
                    curr_sum = 0.0
                    for k in range(self.ncols()):
                        curr_sum += (self[i,k] * rhs[k,j])

                    result[i,j] = curr_sum

        return result

###############################################################################
class CMMatrix(RMMatrix):
###############################################################################
    """
    A 2D uncompressed sparse matrix, row major, implemented as a list of lists

    Slow, only used for initialization and debugging
    """
    ###########################################################################
    def __init__(self, rows, cols, pct_nz=0, diag_lambda=None):
    ###########################################################################
        super(CMMatrix, self).__init__(rows, cols, pct_nz=pct_nz, diag_lambda=diag_lambda)

    ###########################################################################
    def __getitem__(self, tup): return self._matrix[self.nrows()*tup[1] + tup[0]]
    ###########################################################################

    ###########################################################################
    def __setitem__(self, tup, val): self._matrix[self.nrows()*tup[1] + tup[0]] = val
    ###########################################################################


###########################################################################
class Submatrix(RMMatrix):
###########################################################################

    ###########################################################################
    def __init__(self, parent_matrix, row_offset, col_offset, srows, scols):
    ###########################################################################
        self._parent_matrix = parent_matrix
        self._row_offset    = row_offset
        self._col_offset    = col_offset
        self._srows         = srows
        self._scols         = scols

    ###########################################################################
    def nrows(self): return self._srows
    ###########################################################################

    ###########################################################################
    def ncols(self): return self._scols
    ###########################################################################

    ###########################################################################
    def __getitem__(self, tup):
    ###########################################################################
        return self._parent_matrix[tup[0] + self._row_offset, tup[1] + self._col_offset]

    ###########################################################################
    def __setitem__(self, tup, val):
    ###########################################################################
        self._parent_matrix[tup[0] + self._row_offset, tup[1] + self._col_offset] = val

###############################################################################
def gemm(rows, cols, pct_nz, seed, repeat):
###############################################################################
    total = 0
    for _ in range(repeat):
        a = Matrix2DR(rows, cols, pct_nz=pct_nz)
        b = Matrix2DR(rows, cols, pct_nz=pct_nz)
        t1 = time.time()
        c = a * b
        total += (time.time() - t1)

    print(f"Did GEMM using Matrix2DR for {rows}x{cols} {repeat} times in {total} seconds")

    # total = 0
    # for _ in range(repeat):
    #     a = RMMatrix(rows, cols, pct_nz=pct_nz)
    #     b = RMMatrix(rows, cols, pct_nz=pct_nz)
    #     t1 = time.time()
    #     c = a * b
    #     total += (time.time() - t1)

    # print(f"Did GEMM using RMMatrix  for {rows}x{cols} {repeat} times in {total} seconds")

    # total = 0
    # for _ in range(repeat):
    #     a = RMMatrix(rows, cols, pct_nz=pct_nz)
    #     b = CMMatrix(rows, cols, pct_nz=pct_nz)
    #     t1 = time.time()
    #     c = a * b
    #     total += (time.time() - t1)

    # print(f"Did GEMM using Mixed     for {rows}x{cols} {repeat} times in {total} seconds")

###############################################################################
def parse_command_line(args, description):
###############################################################################
    parser = argparse.ArgumentParser(
        usage="""\n{0} <rows> <cols> <non-zero-pct> [--verbose]
OR
{0} --help

\033[1mEXAMPLES:\033[0m
    \033[1;32m# Run gmres on a 30x40 matrix with 20%% non-zero entries \033[0m
    > {0} 30 40 20

    \033[1;32m# Run with hardcoded matrix 0 \033[0m
    > {0} 10 10 10 -c 0  # The 10's are ignored. the dims of the harcoded mtx are followed
""".format(pathlib.Path(args[0]).name),
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("rows", type=int, help="The number of rows")

    parser.add_argument("cols", type=int, help="The number of cols")

    parser.add_argument("-p", "--pct_nz", default=100, type=int, help="The percent of non-zero entries as an integer 0-100")

    parser.add_argument("-s", "--seed", type=int, help="The random number generator seed.")

    parser.add_argument("-r", "--repeat", type=int, default=1, help="How many times to repeat")

    return parser.parse_args(args[1:])

###############################################################################
def _main_func(description):
###############################################################################
    gemm(**vars(parse_command_line(sys.argv, description)))

###############################################################################

if (__name__ == "__main__"):
    _main_func(__doc__)
