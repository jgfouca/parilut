
import random, sys
from dataclasses import dataclass

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
def convert_counts_to_sum(the_list):
###############################################################################
    curr_sum = 0
    for idx, count in enumerate(the_list):
        the_list[idx] = curr_sum
        curr_sum += count

    return curr_sum

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
    def __add__(self, rhs):
    ###########################################################################
        expect(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        expect(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

        result = SparseMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[i][j] = self[i][j] + rhs[i][j]

        return result

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

    ###########################################################################
    def make_triangular(self, lower=True):
    ###########################################################################
        result = SparseMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                if (lower and j <= i) or \
                   (not lower and j >= i):
                    result[i][j] = self[i][j]

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
    def iter_cols_in_row(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._csr_cols[self._csr_rows[row_idx]:self._csr_rows[row_idx+1]])

    ###########################################################################
    def iter_vals_in_row(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._values[self._csr_rows[row_idx]:self._csr_rows[row_idx+1]])

    ###########################################################################
    def __eq__(self, rhs):
    ###########################################################################
        if self.nrows() != rhs.nrows() or self.ncols() != rhs.ncols():
            return False

        if (self._csr_rows != rhs._csr_rows) or \
           (self._csr_cols != rhs._csr_cols) or \
           (self._values   != rhs._values):
            return False

        return True

    ###########################################################################
    def make_triangular(self, lower=True):
    ###########################################################################
        result = CSR(SparseMatrix(0, 0))
        result._nrows = self.nrows()
        result._ncols = self.ncols()

        nnz_count = 0
        nnz_skip = 0
        for row_idx in range(self.nrows()):
            for col_idx in self.iter_cols_in_row(row_idx):
                if (lower and col_idx <= row_idx) or \
                    (not lower and col_idx >= row_idx):
                    result._csr_cols.append(col_idx)
                    result._values.append(self._values[nnz_count + nnz_skip])
                    nnz_count += 1
                else:
                    nnz_skip += 1

            result._csr_rows.append(nnz_count)

        return result

    ###########################################################################
    def abstract_spgeam(self, rhs, begin_row_cb, entry_cb):
    ###########################################################################
        expect(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        expect(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

        for row_idx in range(self.nrows()):
            a_begin = self._csr_rows[row_idx]
            a_end   = self._csr_rows[row_idx+1]
            b_begin = rhs._csr_rows[row_idx]
            b_end   = rhs._csr_rows[row_idx+1]
            tot     = (a_end - a_begin) + (b_end - b_begin)

            local_data = begin_row_cb(row_idx)

            skip = False
            for _ in range(tot):
                if skip:
                    skip = False
                    continue

                a_col = self._csr_cols[a_begin] if a_begin < a_end else sys.maxsize
                a_val = self._values[a_begin]   if a_begin < a_end else 0.0
                b_col = rhs._csr_cols[b_begin]  if b_begin < b_end else sys.maxsize
                b_val = rhs._values[b_begin]    if b_begin < b_end else 0.0

                col_idx = min(a_col, b_col)
                skip = col_idx == a_col and col_idx == b_col
                a_begin += col_idx == a_col
                b_begin += col_idx == b_col
                entry_cb(row_idx, col_idx,
                         (a_val if col_idx == a_col else 0.0), (b_val if col_idx == b_col else 0.0),
                         local_data)

    ###########################################################################
    def __add__(self, rhs):
    ###########################################################################
        """
        Sparse general matrix-matrix addition (spgeam)
        """
        # Cheesy way to create an empty CSR
        result = CSR(SparseMatrix(0, 0))
        result._csr_rows = []

        result._nrows = self.nrows()
        result._ncols = 0

        begin_row_cb = lambda row_idx : result._csr_rows.append(result._ncols)
        def entry_cb(row_idx, col_idx, a_val, b_val, _):
            result._ncols += 1
            result._csr_cols.append(col_idx)
            result._values.append(a_val + b_val)

        self.abstract_spgeam(rhs, begin_row_cb, entry_cb)

        result._csr_rows.append(result._ncols)
        result._ncols = self.ncols()

        # Debug checks
        uself, urhs = self.uncompress(), rhs.uncompress()
        uresult = uself + urhs
        expect(uresult == result.uncompress(), "CSR addition does not work")

        return result

    ###########################################################################
    def _spgemm_insert_row2(self, cols, a, b, row_idx):
    ###########################################################################
        for a_nz in range(a._csr_rows[row_idx], a._csr_rows[row_idx+1]):
            b_row = a._csr_cols[a_nz]
            cols.update(b.iter_cols_in_row(b_row))

    ###########################################################################
    def _spgemm_accumulate_row2(self, cols, a, b, scale, row_idx):
    ###########################################################################
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
        nnz = convert_counts_to_sum(result._csr_rows)

        result._csr_rows.append(nnz)
        expect(len(result._csr_rows) == self.nrows() + 1, "Bad csr_rows")

        # second sweep: accumulate non-zeros
        result._csr_cols = [0] * nnz
        result._values   = [0] * nnz

        local_row_nzs = {}
        for lhs_row in range(self.nrows()):
            local_row_nzs.clear()
            self._spgemm_accumulate_row2(local_row_nzs, self, rhs, 1.0, lhs_row)
            # store result
            c_nz = result._csr_rows[lhs_row]
            for k, v in sorted(local_row_nzs.items()):
                result._csr_cols[c_nz] = k
                result._values[c_nz] = v
                c_nz += 1

        # Debug checks
        uself, urhs = self.uncompress(), rhs.uncompress()
        uresult = uself * urhs
        expect(uresult == result.uncompress(), "CSR dot prod does not work")

        return result

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        idx = 0
        for row_idx in range(self._nrows):
            for col_idx in self.iter_cols_in_row(row_idx):
                result[row_idx][col_idx] = self._values[idx]
                idx += 1

        return result

###############################################################################
class CSC(object):
###############################################################################

    ###########################################################################
    def __init__(self, csr_matrix):
    ###########################################################################
        self._nrows = csr_matrix.nrows()
        self._ncols = csr_matrix.ncols()

        csr_nnz = len(csr_matrix._values)
        self._values = [0.0] * csr_nnz
        self._csc_cols = [0] * (self._ncols + 1)
        self._csc_rows = [0] * csr_nnz

        nnz = 0
        for col_idx in csr_matrix._csr_cols:
            self._csc_cols[col_idx] += 1
            nnz += 1

        expect(nnz == csr_nnz, f"Bad nnz in CSC init, {nnz} != {csr_nnz}")

        convert_counts_to_sum(self._csc_cols)

        counts = [0] * self._ncols # [col]->num occurences found for col
        # Find rows that have these cols
        for row_idx in range(csr_matrix.nrows()):
            for csr_col_idx, val in zip(csr_matrix.iter_cols_in_row(row_idx), csr_matrix.iter_vals_in_row(row_idx)):
                col_count = counts[csr_col_idx]
                counts[csr_col_idx] += 1
                rows_offset = self._csc_cols[csr_col_idx] + col_count
                self._csc_rows[rows_offset] = row_idx
                self._values[rows_offset]   = val

        # Debug check
        expect(csr_matrix.uncompress() == self.uncompress(), "CSC contructor broken")

    ###########################################################################
    def nrows(self): return self._nrows
    ###########################################################################

    ###########################################################################
    def ncols(self): return self._ncols
    ###########################################################################

    ###########################################################################
    def __str__(self):
    ###########################################################################
        return str(self.uncompress())

    ###########################################################################
    def iter_rows_in_col(self, col_idx):
    ###########################################################################
        expect(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return iter(self._csc_rows[self._csc_cols[col_idx]:self._csc_cols[col_idx+1]])

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        idx = 0
        for col_idx in range(self._ncols):
            for row_idx in self.iter_rows_in_col(col_idx):
                result[row_idx][col_idx] = self._values[idx]
                idx += 1

        return result

###############################################################################
class PARILUT(object):
###############################################################################

    ###########################################################################
    def __init__(self, A):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "PARILUT matrix must be square")

        self._A     = A
        self._A_csr = CSR(A)
        self._L_csr = self._A_csr.make_triangular()
        self._U_csr = self._A_csr.make_triangular(lower=False)

        expect(self._A == self._A_csr.uncompress(), "CSR compression does not work")
        expect(self._L_csr.uncompress() == self._A.make_triangular(), "CSR lower tri does not work")
        expect(self._U_csr.uncompress() == self._A.make_triangular(lower=False), "CSR lower tri does not work")

    ###########################################################################
    def _add_candidates(self, lu):
    ###########################################################################
        """
        Adds new entries from the sparsity pattern of A - L * U
        to L and U, where new values are chosen based on the residual
        value divided by the corresponding diagonal entry.
        """
        # Very inefficient
        l_new = CSR(SparseMatrix(self._A_csr.nrows(), self._A_csr.ncols()))
        u_new = CSR(SparseMatrix(self._A_csr.nrows(), self._A_csr.ncols()))

        def begin_row_cb_sizing(row_idx):
            l_new._csr_rows[row_idx] = l_new._csr_rows[-1]
            u_new._csr_rows[row_idx] = u_new._csr_rows[-1]

        def entry_cb_sizing(row_idx, col_idx, _, __, ___):
            l_new._csr_rows[-1] += col_idx <= row_idx
            u_new._csr_rows[-1] += col_idx >= row_idx

        self._A_csr.abstract_spgeam(lu, begin_row_cb_sizing, entry_cb_sizing)

        l_new._csr_cols = [0]   * l_new._csr_rows[-1]
        l_new._values   = [0.0] * l_new._csr_rows[-1]
        u_new._csr_cols = [0]   * u_new._csr_rows[-1]
        u_new._values   = [0.0] * u_new._csr_rows[-1]

        @dataclass
        class RowState:
            l_new_nz:    int
            u_new_nz:    int
            l_old_begin: int
            l_old_end:   int
            u_old_begin: int
            u_old_end:   int
            finished_l:  bool

        def begin_row_cb(row_idx):
            state = RowState(
                l_new._csr_rows[row_idx],
                u_new._csr_rows[row_idx],
                self._L_csr._csr_rows[row_idx],
                self._L_csr._csr_rows[row_idx + 1] - 1,  # skip diagonal. JGF: what if there isn't one?
                self._U_csr._csr_rows[row_idx],
                self._U_csr._csr_rows[row_idx + 1],
                (self._L_csr._csr_rows[row_idx] >= self._L_csr._csr_rows[row_idx + 1] - 1)) # JGF Modify!
            return state

        def entry_cb(row_idx, col_idx, a_val, lu_val, state):
            r_val = a_val - lu_val
            # load matching entry of L + U
            lpu_col = (self._U_csr._csr_cols[state.u_old_begin] if state.u_old_begin < state.u_old_end else sys.maxsize) \
                      if state.finished_l else self._L_csr._csr_cols[state.l_old_begin]
            lpu_val = (self._U_csr._values[state.u_old_begin] if state.u_old_begin < state.u_old_end else 0.0) \
                      if state.finished_l else self._L_csr._values[state.l_old_begin]

            # load diagonal entry of U for lower diagonal entries
            idx = self._U_csr._csr_rows[col_idx]
            diag = self._U_csr._values[idx] if col_idx < row_idx and idx < len(self._U_csr._values) else 1.0 # JGF Modify!
            # if there is already an entry present, use that instead.
            out_val = lpu_val if lpu_col == col_idx else r_val / diag
            # store output entries
            if row_idx >= col_idx:
                l_new._csr_cols[state.l_new_nz] = col_idx
                l_new._values[state.l_new_nz] = 1.0 if row_idx == col_idx else out_val
                state.l_new_nz += 1
            if row_idx <= col_idx:
                u_new._csr_cols[state.u_new_nz] = col_idx
                u_new._values[state.u_new_nz] = out_val
                state.u_new_nz += 1

            # advance entry of L + U if we used it
            if state.finished_l:
                state.u_old_begin += (lpu_col == col_idx)
            else:
                state.l_old_begin += (lpu_col == col_idx)
                state.finished_l = (state.l_old_begin == state.l_old_end)

        self._A_csr.abstract_spgeam(lu, begin_row_cb, entry_cb)

        return l_new, u_new

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        while not converged:
            LU_csr = self._L_csr * self._U_csr

            l_new_csr, u_new_csr = self._add_candidates(LU_csr)

            # Convert u_new to CSC
            u_new_csc = CSC(u_new_csr)

            # Convert u_new and l_new to COO
            #u_

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
