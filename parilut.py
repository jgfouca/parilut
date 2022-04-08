
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
    A 2D uncompressed sparse matrix, row major, implemented as a list of lists
    """

    ###########################################################################
    def __init__(self, rows, cols, pct_nz=0):
    ###########################################################################
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
    def ncols(self): return 0 if self.nrows() == 0 else len(self._matrix[0])
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

    ###########################################################################
    def __mul__(self, rhs):
    ###########################################################################
        expect(self.ncols() == rhs.nrows(), "Cannot multiply matrix, incompatible dims")

        result = SparseMatrix(self.nrows(), rhs.ncols())

        for i in range(self.nrows()):
            for j in range(rhs.ncols()):
                curr_sum = 0.0
                for k in range(self.ncols()):
                    curr_sum += (self[i][k] * rhs[k][j])

                result[i][j] = curr_sum

        return result

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
        self._csr_rows = [0]

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
    def nrows(self): return self._nrows
    ###########################################################################

    ###########################################################################
    def ncols(self): return self._ncols
    ###########################################################################

    ###########################################################################
    def _spgemm_insert_row2(self, cols, a, b, row_idx):
    ###########################################################################
        """
        """
        for a_nz in range(a._csr_rows[row_idx], a._csr_rows[row_idx+1]):
            b_row = a._csr_cols[a_nz]
            cols.update(b._csr_cols[b._csr_rows[b_row]:b._csr_rows[b_row+1]])

    ###########################################################################
    def _spgemm_accumulate_row2(self, cols, a, b, scale, row_idx):
    ###########################################################################
        """
        """
        for a_nz in range(a._csr_rows[row_idx], a._csr_rows[row_idx+1]):
            b_row = a._csr_cols[a_nz]
            a_val = a._values[a_nz]
            for b_nz in range(b._csr_rows[b_row], b._csr_rows[b_row+1]):
                b_col = b._csr_cols[b_nz]
                b_val = b._values[b_nz]
                if b_col in cols:
                    cols[b_col] += scale * a_val * b_val
                else:
                    cols[b_col] = scale * a_val * b_val

    ###########################################################################
    def __mul__(self, rhs):
    ###########################################################################
        """
        Sparse general matrix-matrix multiplication (spgemm)
        """
        expect(self.ncols() == rhs.nrows(), "Cannot multiply matrix, incompatible dims")

        # Cheesy way to create an empty CSR
        result = CSR(SparseMatrix(0, 0))
        result._csr_rows = []

        result._nrows = self.nrows()
        result._ncols = rhs.ncols()

        # first sweep: count nnz for each row
        for lhs_row in range(self.nrows()):
            local_col_idxs = set()
            self._spgemm_insert_row2(local_col_idxs, self, rhs, lhs_row)
            result._csr_rows.append(len(local_col_idxs))

        # build row pointers
        curr_sum = 0
        for idx, nnz in enumerate(result._csr_rows):
            result._csr_rows[idx] = curr_sum
            curr_sum += nnz

        result._csr_rows.append(curr_sum)
        expect(len(result._csr_rows) == self.nrows() + 1, "Bad csr_rows")

        # second sweep: accumulate non-zeros
        result._csr_cols = [0] * curr_sum
        result._values   = [0] * curr_sum

        local_row_nzs = {}
        for lhs_row in range(self.nrows()):
            local_row_nzs.clear()
            self._spgemm_accumulate_row2(local_row_nzs, self, rhs, 1.0, lhs_row)
            # store result
            c_nz = result._csr_rows[lhs_row]
            for k, v in local_row_nzs.items():
                result._csr_cols[c_nz] = k
                result._values[c_nz] = v
                c_nz += 1

        # Debug checks
        uself, urhs = self.uncompress(), rhs.uncompress()
        uresult = uself * urhs
        curesult = CSR(uresult)
        #expect(uresult == result.uncompress(), "CSR dot prod does not work")
        if uresult != result.uncompress():
            print("lhs:")
            print(self)
            print("rhs:")
            print(rhs)
            print("uresult:")
            print(uresult)
            print("result:")
            print(result)
            print(f"curesult csr_rows = {curesult._csr_rows}")
            print(f"  result csr_rows = {result._csr_rows}")
            expect(False, "end")

        return result

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        prev = 0
        for row_idx, val in enumerate(self._csr_rows[1:]):
            active_cols = self._csr_cols[prev:val]
            idx = prev
            for col in range(self._ncols):
                if col in active_cols:
                    result[row_idx][col] = self._values[idx]
                    idx += 1

            prev = val

        return result

###############################################################################
class PARILUT(object):
###############################################################################

    ###########################################################################
    def __init__(self, A):
    ###########################################################################
        self._A = A

        rows = A.nrows()
        cols = A.ncols()

        expect(rows == cols, "PARILUT matrix must be square")

        L = SparseMatrix(rows, cols)
        U = SparseMatrix(rows, cols)

        # A bit inefficient. It would be more inefficient to use A_csr to make the
        # L and U csrs, but we don't care about efficiency here.
        for row in range(rows):
            for col in range(cols):
                aval = A[row][col]
                if aval > 0.0:
                    if row >= col:
                        L[row][col] = aval
                    if row <= col:
                        U[row][col] = aval

        self._A_csr, self._L_csr, self._U_csr = CSR(A), CSR(L), CSR(U)

        expect(self._A == self._A_csr.uncompress(), "CSR compression does not work")

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        while not converged:
            LU = self._L_csr * self._U_csr
            converged = True

###############################################################################
def parilut(rows, cols, pct_nz):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"{rows} != {cols}. Only square matrices allowed")
    expect(pct_nz > 0 and pct_nz <= 100, f"Bad pct_nz {pct_nz}")

    A = SparseMatrix(rows, cols, pct_nz)

    pilut = PARILUT(A)

    pilut.main()