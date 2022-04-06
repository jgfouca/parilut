
import random

###############################################################################
def expect(condition, error_msg, exc_type=SystemExit, error_prefix="ERROR:"):
###############################################################################
    """
    Similar to assert except doesn't generate an ugly stacktrace. Useful for
    checking user error, not programming error.
    """
    if not condition:
        msg = error_prefix + " " + error_msg
        raise exc_type(msg)

###############################################################################
class SparseMatrix(object):
###############################################################################
    """
    A 2D uncompressed sparse matrix
    """

    ###########################################################################
    def __init__(self, rows, cols, pct_nz=0):
    ###########################################################################
        # row-major
        self._matrix = [list() for _ in range(rows)]

        for row in self:
            for _ in range(cols): # each row has num-cols items
                roll = random.randint(0, 100)
                nz = roll < pct_nz
                nz_val = random.uniform(0.0, 1.0)
                row.append(nz_val if nz else 0.0)

    ###########################################################################
    def __str__(self):
    ###########################################################################
        result = ""
        for row in self:
            for val in row:
                result += "{:.2f} ".format(val)

            result += "\n"

        result += "\n"

        return result

    ###########################################################################
    def nrows(self): return len(self._matrix)
    ###########################################################################

    ###########################################################################
    def ncols(self): return len(self._matrix[0])
    ###########################################################################

    ###########################################################################
    def __iter__(self): return iter(self._matrix)
    ###########################################################################

    ###########################################################################
    def __getitem__(self, idx): return self._matrix[idx]
    ###########################################################################

    ###########################################################################
    def __eq__(self, rhs):
    ###########################################################################
        if self.nrows() != rhs.nrows() or self.ncols() != rhs.ncols():
            return False

        for lhsrow, rhsrow in zip(self, rhs):
            for lhsval, rhsval in zip(lhsrow, rhsrow):
                if lhsval != rhsval:
                    return False

        return True

###############################################################################
class CSR(object):
###############################################################################

    ###########################################################################
    def __init__(self, src_matrix):
    ###########################################################################
        self._nrows = src_matrix.nrows()
        self._ncols = src_matrix.ncols()

        self._values   = []
        self._csr_cols = []
        self._csr_rows = []

        nnz = 0
        for row in src_matrix:
            for col_idx, val in enumerate(row):
                if val != 0.0:
                    self._csr_cols.append(col_idx)
                    self._values.append(val)
                    nnz += 1

            self._csr_rows.append(nnz)

    ###########################################################################
    def __str__(self):
    ###########################################################################
        return str(self.uncompress())

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        prev = 0
        for row_idx, val in enumerate(self._csr_rows):
            active_cols = self._csr_cols[prev:val]
            idx = prev
            for col in range(self._ncols):
                if col in active_cols:
                    result[row_idx][col] = self._values[idx]
                    idx += 1

            prev = val

        return result

###############################################################################
def parilut_init(A):
###############################################################################
    rows = A.nrows()
    cols = A.ncols()

    L = SparseMatrix(rows, cols)
    U = SparseMatrix(rows, cols)

    for row in range(rows):
        for col in range(cols):
            aval = A[row][col]
            if aval > 0.0:
                if row >= col:
                    L[row][col] = aval
                if row <= col:
                    U[row][col] = aval

    return CSR(A), CSR(L), CSR(U)

###############################################################################
def parilut_main(A, L_csr, U_csr):
###############################################################################
    # Identify candidate locations
    pass

###############################################################################
def parilut(rows, cols, pct_nz):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"{rows} != {cols}. Only square matrices allowed")
    expect(pct_nz > 0 and pct_nz <= 100, f"Bad pct_nz {pct_nz}")

    A = SparseMatrix(rows, cols, pct_nz)
    print(A)
    A_csr, L_csr, U_csr  = parilut_init(A)
    print(A_csr)
    expect(A == A_csr.uncompress(), "Something is wrong")

    parilut_main(A, L_csr, U_csr)
