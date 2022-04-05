
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
def print_matrix(matrix):
###############################################################################
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(rows):
        for col in range(cols):
            print("{:.2f} ".format(matrix[row][col]), end="")

        print()

    print()

###############################################################################
class CSR(object):
###############################################################################

    ###########################################################################
    def __init__(self, src_matrix):
    ###########################################################################
        self._rows = len(src_matrix)
        self._cols = len(src_matrix[0])

        self._values   = []
        self._csr_cols = []
        self._csr_rows = []

        nnz = 0
        for row in range(self._rows):
            for col in range(self._cols):
                if src_matrix[row][col] != 0.0:
                    self._csr_cols.append(col)
                    self._values.append(src_matrix[row][col])
                    nnz += 1

            self._csr_rows.append(nnz)

    ###########################################################################
    def print(self):
    ###########################################################################
        prev = 0
        for val in self._csr_rows:
            active_cols = self._csr_cols[prev:val]
            idx = prev
            for col in range(self._cols):
                if col in active_cols:
                    print("{:.2f} ".format(self._values[idx]), end="")
                    idx += 1
                else:
                    print("{:.2f} ".format(0.0), end="")

            prev = val
            print()

###############################################################################
def parilut_init(A):
###############################################################################
    rows = len(A)
    cols = len(A[0])

    L = [list() for _ in range(rows)]
    U = [list() for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            L[row].append(0.0)
            U[row].append(0.0)

            aval = A[row][col]
            if aval > 0.0:
                if row >= col:
                    L[row][col] = aval
                if row <= col:
                    U[row][col] = aval

    return CSR(L), CSR(U)

###############################################################################
def parilut_main(A, L_csr, U_csr):
###############################################################################
    pass

###############################################################################
def parilut(rows, cols, pct_nzero):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"Only square matrices allowed")
    expect(pct_nzero > 0 and pct_nzero <= 100, f"Bad pct_nzero {pct_nzero}")

    # row-major matrix A
    A = [list() for _ in range(rows)]
    for row in A:
        for _ in range(cols): # each row has num-cols items
            roll = random.randint(0, 100)
            nz = roll < pct_nzero
            nz_val = random.uniform(0.0, 1.0)
            row.append(nz_val if nz else 0.0)

    A[0][4] = .42

    print_matrix(A)
    L_csr, U_csr  = parilut_init(A)

    parilut_main(A, L_csr, U_csr)
    
