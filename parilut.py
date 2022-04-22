
import random, sys, bisect, math
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

    Slow, only used for initialization and debugging
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
                row.append(nz_val if nz else 0.0) # Slow but that's OK for this class

        # Non-zero matrix must have at least 1 value in each row and col
        # so make unit diagonal
        if pct_nz > 0:
            for i in range(rows):
                self[i][i] = 1.0

    ###########################################################################
    @staticmethod
    def get_hardcode(matrix_id):
    ###########################################################################
        if matrix_id == 0:
            hardcoded_vals = [
                [1.,   6.,   4., 7.],
                [2.,  -5.,   0., 8.],
                [0.5, -3.,   6., 0.],
                [0.2, -0.5, -9., 0.,],
            ]
            result = SparseMatrix(4, 4)
            for row_idx in range(4):
                for col_idx in range(4):
                    result[row_idx][col_idx] = hardcoded_vals[row_idx][col_idx]

            return result
        else:
            expect(False, f"Unknown hardcoded matrix id {matrix_id}")

        return None

    ###########################################################################
    def __str__(self):
    ###########################################################################
        result = ""
        for row in self:
            for val in row:
                result += "{:.2f} ".format(val)

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
    def make_lu(self):
    ###########################################################################
        l = SparseMatrix(self.nrows(), self.ncols())
        u = SparseMatrix(self.nrows(), self.ncols())

        for row_idx in range(self.nrows()):
            diag_val = 1.
            for col_idx in range(self.ncols()):
                val = self[row_idx][col_idx]
                if col_idx < row_idx:
                    l[row_idx][col_idx] = val
                elif col_idx == row_idx:
                    l[row_idx][col_idx] = 1.
                    if val != 0.0:
                        diag_val = val
                else:
                    u[row_idx][col_idx] = val

            u[row_idx][row_idx] = diag_val

        return l, u

    ###########################################################################
    def nnz(self):
    ###########################################################################
        result = 0
        for row in self:
            for col in row:
                result += col != 0.0

        return result

    ###########################################################################
    def filter_pred(self, pred):
    ###########################################################################
        result = SparseMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                val = self[i][j]
                if pred(i, j, val):
                    result[i][j] = val

        return result

    ###########################################################################
    def transpose(self):
    ###########################################################################
        result = SparseMatrix(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[j][i] = self[i][j]

        return result

###############################################################################
class CompressedMatrix(object):
###############################################################################

    def __init__(self, nrows, ncols, nnz):
        self._nrows, self._ncols = nrows, ncols
        self._values = [0.0] * nnz

    def __str__(self):
        return str(self.uncompress())

    def nrows(self): return self._nrows

    def ncols(self): return self._ncols

    def nnz(self): return len(self._values)

    def uncompress(self): expect(False, "Subclass should implement")

###############################################################################
class CSR(CompressedMatrix):
###############################################################################

    ###########################################################################
    def __init__(self, nrows=0, ncols=0, nnz=0, src_matrix=None):
    ###########################################################################
        if src_matrix is not None:
            expect(nrows == 0 and ncols == 0 and nnz == 0, "Do not use with src_matrix")
            nrows, ncols, nnz = src_matrix.nrows(), src_matrix.ncols(), src_matrix.nnz()

        super().__init__(nrows, ncols, nnz)

        self._csr_cols = [0] * self.nnz()
        self._csr_rows = [0] * (self.nrows()+1)

        if src_matrix is not None:
            nnz = 0
            for row_idx, row in enumerate(src_matrix):
                for col_idx, val in enumerate(row):
                    if val != 0.0:
                        self._csr_cols[nnz] = col_idx
                        self._values[nnz] = val
                        nnz += 1

                self._csr_rows[row_idx+1] = nnz

            expect(self.uncompress() == src_matrix, "CSR compression failed")

    ###########################################################################
    def get_nnz_range(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return self._csr_rows[row_idx], self._csr_rows[row_idx+1]

    ###########################################################################
    def iter_cols_in_row(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._csr_cols[slice(*self.get_nnz_range(row_idx))])

    ###########################################################################
    def iter_vals_in_row(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._values[slice(*self.get_nnz_range(row_idx))])

    ###########################################################################
    def iter_row(self, row_idx):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return zip(self.iter_cols_in_row(row_idx), self.iter_vals_in_row(row_idx))

    ###########################################################################
    def has(self, row_idx, col_idx_arg):
    ###########################################################################
        expect(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        # Slow but that's OK for now
        for col_idx in self.iter_cols_in_row(row_idx):
            if col_idx == col_idx_arg:
                return True
            if col_idx > col_idx_arg: # We passed it
                return False

        return False

    ###########################################################################
    def resize(self, nnz):
    ###########################################################################
        self._csr_cols = [0]*nnz
        self._values   = [0.]*nnz

    ###########################################################################
    def __eq__(self, rhs):
    ###########################################################################
        result = True
        if self.nrows() != rhs.nrows() or self.ncols() != rhs.ncols():
            result = False

        if (self._csr_rows != rhs._csr_rows) or \
           (self._csr_cols != rhs._csr_cols) or \
           (self._values   != rhs._values):
            result = False

        # Debug check
        expect(result == (self.uncompress() == rhs.uncompress()), "csr eq failed")

        return result

    ###########################################################################
    def make_lu(self):
    ###########################################################################
        l = CSR(self.nrows(), self.ncols())
        u = CSR(self.nrows(), self.ncols())

        # Sizing
        l_nnz = 0
        u_nnz = 0
        for row_idx in range(self.nrows()):
            for col_idx in self.iter_cols_in_row(row_idx):
                # don't count diagonal
                l_nnz += col_idx < row_idx
                u_nnz += col_idx > row_idx

            # add diagonal again
            l_nnz += 1
            u_nnz += 1
            l._csr_rows[row_idx + 1] = l_nnz
            u._csr_rows[row_idx + 1] = u_nnz

        l.resize(l_nnz)
        u.resize(u_nnz)

        for row_idx in range(self.nrows()):
            current_index_l = l._csr_rows[row_idx]
            current_index_u = u._csr_rows[row_idx] + 1 # we treat the diagonal separately
            # if there is no diagonal value, set it to 1 by default
            diag_val = 1.
            for col_idx, val in self.iter_row(row_idx):
                if col_idx < row_idx:
                    l._csr_cols[current_index_l] = col_idx
                    l._values[current_index_l] = val
                    current_index_l += 1
                elif col_idx == row_idx:
                    # save diagonal value
                    diag_val = val
                else: # col > row
                    u._csr_cols[current_index_u] = col_idx
                    u._values[current_index_u] = val
                    current_index_u += 1

            # store diagonal values separately
            l_diag_idx = l._csr_rows[row_idx + 1] - 1
            u_diag_idx = u._csr_rows[row_idx]
            l._csr_cols[l_diag_idx] = row_idx
            u._csr_cols[u_diag_idx] = row_idx
            l._values[l_diag_idx] = 1.
            u._values[u_diag_idx] = diag_val

        # Debug
        uself = self.uncompress()
        ul, uu = uself.make_lu()
        expect(l.uncompress() == ul, "make_lu failed for u")
        expect(u.uncompress() == uu, "make_lu failed for u")

        return l, u

    ###########################################################################
    def abstract_spgeam(self, rhs, begin_row_cb, entry_cb):
    ###########################################################################
        expect(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        expect(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

        for row_idx in range(self.nrows()):
            a_begin, a_end = self.get_nnz_range(row_idx)
            b_begin, b_end = rhs.get_nnz_range(row_idx)
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
    def abstract_spgeam_lu_size(self, rhs, l_new, u_new):
    ###########################################################################

        def begin_row_cb_sizing(row_idx):
            l_new[row_idx] = l_new[-1]
            u_new[row_idx] = u_new[-1]

        def entry_cb_sizing(row_idx, col_idx, _, __, ___):
            l_new[-1] += col_idx <= row_idx
            u_new[-1] += col_idx >= row_idx

        self.abstract_spgeam(rhs, begin_row_cb_sizing, entry_cb_sizing)

        return l_new[-1], u_new[-1]

    ###########################################################################
    def abstract_spgeam_size(self, rhs, csr_rows):
    ###########################################################################

        def begin_row_cb_sizing(row_idx):
            csr_rows[row_idx] = csr_rows[-1]

        def entry_cb_sizing(_, __, ___, ____, _____):
            csr_rows[-1] += 1

        self.abstract_spgeam(rhs, begin_row_cb_sizing, entry_cb_sizing)

        return csr_rows[-1]

    ###########################################################################
    def __add__(self, rhs):
    ###########################################################################
        """
        Sparse general matrix-matrix addition (spgeam)
        """
        result = CSR(self.nrows(), self.nrows())

        nnz = self.abstract_spgeam_size(rhs, result._csr_rows)
        result._csr_cols = [0] * nnz
        result._values = [0.0] * nnz
        result._ncols = 0 # Temporarily use as scratch space for cur_nnz count

        begin_row_cb = lambda x : x
        def entry_cb(row_idx, col_idx, a_val, b_val, _):
            result._csr_cols[result._ncols] = col_idx
            result._values[result._ncols] = a_val + b_val
            result._ncols += 1

        self.abstract_spgeam(rhs, begin_row_cb, entry_cb)

        result._ncols = self.ncols() # Stop using as scratch space

        # Debug checks
        uself, urhs = self.uncompress(), rhs.uncompress()
        uresult = uself + urhs
        expect(uresult == result.uncompress(), "CSR addition does not work")

        return result

    ###########################################################################
    def _spgemm_insert_row2(self, cols, a, b, row_idx):
    ###########################################################################
        for a_col_idx in a.iter_cols_in_row(row_idx):
            cols.update(b.iter_cols_in_row(a_col_idx))

    ###########################################################################
    def _spgemm_accumulate_row2(self, cols, a, b, scale, row_idx):
    ###########################################################################
        for b_row, a_val in a.iter_row(row_idx):
            for b_col, b_val in b.iter_row(b_row):
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

        result = CSR(self.nrows(), rhs.ncols())

        # first sweep: count nnz for each row
        for lhs_row in range(self.nrows()):
            local_col_idxs = set()
            self._spgemm_insert_row2(local_col_idxs, self, rhs, lhs_row)
            result._csr_rows[lhs_row] = len(local_col_idxs)

        # build row pointers
        nnz = convert_counts_to_sum(result._csr_rows)

        expect(len(result._csr_rows) == self.nrows() + 1, "Bad csr_rows")

        # second sweep: accumulate non-zeros
        result._csr_cols = [0] * nnz
        result._values   = [0.0] * nnz

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

        for row_idx in range(self._nrows):
            for col_idx, val in self.iter_row(row_idx):
                result[row_idx][col_idx] = val

        return result

    ###########################################################################
    def filter_pred(self, pred):
    ###########################################################################
        result = CSR(self.nrows(), self.ncols())

        # Sizes
        for row_idx in range(self.nrows()):
            for col_idx, val in self.iter_row(row_idx):
                if pred(row_idx, col_idx, val):
                    result._csr_rows[row_idx] += 1

        convert_counts_to_sum(result._csr_rows)
        nnz = result._csr_rows[-1]

        result._values = [0.0] * nnz
        result._csr_cols = [0] * nnz

        # Vals
        nnz = 0
        for row_idx in range(self.nrows()):
            for col_idx, val in self.iter_row(row_idx):
                if pred(row_idx, col_idx, val):
                    result._csr_cols[nnz] = col_idx
                    result._values[nnz] = val
                    nnz += 1

        # Debug check
        expect(result.uncompress() == self.uncompress().filter_pred(pred),
               "csr filter_pred failed")

        return result

    ###########################################################################
    def sort(self):
    ###########################################################################
        """
        What does sorting a matrix even mean?
        """
        expect(False, "No yet implemented")


    ###########################################################################
    def transpose(self):
    ###########################################################################
        result = CSR(self.nrows(), self.ncols(), self.nnz())

        # Sizes
        for row_idx in range(self.nrows()):
            for col_idx in self.iter_cols_in_row(row_idx):
                result._csr_rows[col_idx] += 1

        convert_counts_to_sum(result._csr_rows)

        counts = [0] * self.ncols() # [col]->num occurences found for col
        for row_idx in range(self.nrows()):
            for col_idx, val in self.iter_row(row_idx):
                col_count = counts[col_idx]
                counts[col_idx] += 1
                rows_offset = result._csr_rows[col_idx] + col_count
                result._csr_cols[rows_offset] = row_idx
                result._values[rows_offset]   = val

        # Debug
        expect(result.uncompress() == self.uncompress().transpose(), "Tranpose failed")

        return result

###############################################################################
class CSC(CompressedMatrix):
###############################################################################

    ###########################################################################
    def __init__(self, csr_matrix):
    ###########################################################################
        super().__init__(csr_matrix.nrows(), csr_matrix.ncols(), csr_matrix.nnz())

        self._csc_cols = [0] * (self._ncols + 1)
        self._csc_rows = [0] * self.nnz()

        nnz = 0
        for col_idx in csr_matrix._csr_cols:
            self._csc_cols[col_idx] += 1
            nnz += 1

        expect(nnz == self.nnz(), f"Bad nnz in CSC init, {nnz} != {self.nnz()}")

        convert_counts_to_sum(self._csc_cols)

        counts = [0] * self._ncols # [col]->num occurences found for col
        # Find rows that have these cols
        for row_idx in range(csr_matrix.nrows()):
            for col_idx, val in csr_matrix.iter_row(row_idx):
                col_count = counts[col_idx]
                counts[col_idx] += 1
                rows_offset = self._csc_cols[col_idx] + col_count
                self._csc_rows[rows_offset] = row_idx
                self._values[rows_offset]   = val

        # Debug check
        expect(csr_matrix.uncompress() == self.uncompress(), "CSC contructor broken")

    ###########################################################################
    def get_nnz_range(self, col_idx):
    ###########################################################################
        expect(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return self._csc_cols[col_idx], self._csc_cols[col_idx+1]

    ###########################################################################
    def iter_rows_in_col(self, col_idx):
    ###########################################################################
        expect(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return iter(self._csc_rows[slice(*self.get_nnz_range(col_idx))])

    ###########################################################################
    def iter_vals_in_col(self, col_idx):
    ###########################################################################
        expect(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return iter(self._values[slice(*self.get_nnz_range(col_idx))])

    ###########################################################################
    def iter_col(self, col_idx):
    ###########################################################################
        expect(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return zip(self.iter_rows_in_col(col_idx), self.iter_vals_in_col(col_idx))

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        for col_idx in range(self._ncols):
            for row_idx, val in self.iter_col(col_idx):
                result[row_idx][col_idx] = val

        return result

###############################################################################
class COO(CompressedMatrix):
###############################################################################

    ###########################################################################
    def __init__(self, csr_matrix):
    ###########################################################################
        super().__init__(csr_matrix.nrows(), csr_matrix.ncols(), csr_matrix.nnz())

        self._coo_rows = [0] * self.nnz()
        self._coo_cols = [0] * self.nnz()

        idx = 0
        for row_idx in range(self._nrows):
            for col_idx, val in csr_matrix.iter_row(row_idx):
                self._coo_rows[idx] = row_idx
                self._coo_cols[idx] = col_idx
                self._values[idx]   = val
                idx += 1

        # Debug check
        expect(csr_matrix.uncompress() == self.uncompress(), "COO constructor broken")

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = SparseMatrix(self._nrows, self._ncols)

        for row_idx, col_idx, val in zip(self._coo_rows, self._coo_cols, self._values):
            result[row_idx][col_idx] = val

        return result

###############################################################################
class PARILUT(object):
###############################################################################

    ###########################################################################
    def __init__(self, A):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "PARILUT matrix must be square")

        self._A     = A
        self._A_csr = CSR(src_matrix=A)
        self._L_csr, self._U_csr = self._A_csr.make_lu()

        # params
        self._fill_in_limit = 0.75

        self._l_nnz_limit = int(math.floor(self._fill_in_limit * self._L_csr.nnz()))
        self._u_nnz_limit = int(math.floor(self._fill_in_limit * self._U_csr.nnz()))

        print(f"With fill in limit {self._fill_in_limit}, got l_nnz_limit={self._l_nnz_limit} and u_nnz_limit={self._u_nnz_limit}")

    ###########################################################################
    def _add_candidates(self, lu):
    ###########################################################################
        """
        Adds new entries from the sparsity pattern of A - L * U
        to L and U, where new values are chosen based on the residual
        value divided by the corresponding diagonal entry.
        """
        nrows = self._A.nrows()

        l_new_rows = [0] * (nrows+1)
        u_new_rows = [0] * (nrows+1)

        l_new_nnz, u_new_nnz = \
            self._A_csr.abstract_spgeam_lu_size(lu, l_new_rows, u_new_rows)

        l_new = CSR(nrows, nrows, l_new_nnz)
        u_new = CSR(nrows, nrows, u_new_nnz)

        l_new._csr_rows = l_new_rows
        u_new._csr_rows = u_new_rows

        # Debug, expect diagonal non-zero
        for row_idx in range(nrows):
            expect(self._L_csr.has(row_idx, row_idx),
                   f"No diagonal in L_csr[{row_idx}][{row_idx}]")

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
                self._L_csr._csr_rows[row_idx + 1] - 1,  # skip diagonal.
                self._U_csr._csr_rows[row_idx],
                self._U_csr._csr_rows[row_idx + 1],
                False)
            state.finished_l = state.l_old_begin == state.l_old_end
            return state

        def entry_cb(row_idx, col_idx, a_val, lu_val, state):
            r_val = a_val - lu_val
            # load matching entry of L + U
            lpu_col = (self._U_csr._csr_cols[state.u_old_begin] if state.u_old_begin < state.u_old_end else sys.maxsize) \
                      if state.finished_l else self._L_csr._csr_cols[state.l_old_begin]
            lpu_val = (self._U_csr._values[state.u_old_begin] if state.u_old_begin < state.u_old_end else 0.0) \
                      if state.finished_l else self._L_csr._values[state.l_old_begin]

            # load diagonal entry of U for lower diagonal entries
            diag = self._U_csr._values[self._U_csr._csr_rows[col_idx]] if col_idx < row_idx else 1.0

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
    def _compute_l_u_factors(self, l_csr, u_csr, u_csc):
    ###########################################################################
        """
        JGF: why are COO's needed?
        """
        def compute_sum(row, col):
            # find value from A
            a_begin, a_end = self._A_csr.get_nnz_range(row)
            a_nz = bisect.bisect_left(self._A_csr._csr_cols, col, a_begin, a_end)
            has_a = a_nz < a_end and self._A_csr._csr_cols[a_nz] == col
            a_val = self._A_csr._values[a_nz] if has_a else 0.0
            # accumulate l(row,:) * u(:,col) without the last entry (row, col)
            sum_ = 0.0
            ut_nz = 0
            l_begin, l_end = l_csr.get_nnz_range(row)
            u_begin, u_end = u_csc.get_nnz_range(col)
            last_entry = min(row, col)
            while (l_begin < l_end and u_begin < u_end):
                l_col = l_csr._csr_cols[l_begin]
                u_row = u_csc._csc_rows[u_begin]
                if (l_col == u_row and l_col < last_entry):
                    sum_ += l_csr._values[l_begin] * u_csc._values[u_begin]

                if u_row == row:
                    ut_nz = u_begin

                l_begin += (l_col <= u_row)
                u_begin += (u_row <= l_col)

            return (a_val - sum_, ut_nz)

        num_rows = self._A_csr.nrows()
        for row in range(num_rows):
            nnz_range = l_csr.get_nnz_range(row)
            for l_nz in range(*(nnz_range[0], nnz_range[1]-1)):
                col    = l_csr._csr_cols[l_nz]
                u_diag = u_csc._values[u_csc._csc_cols[col + 1] - 1]
                if u_diag != 0.0:
                    new_val = compute_sum(row, col)[0] / u_diag
                    l_csr._values[l_nz] = new_val

            for u_nz in range(*u_csr.get_nnz_range(row)):
                col = u_csr._csr_cols[u_nz]
                new_val, ut_nz = compute_sum(row, col)
                u_csr._values[u_nz] = new_val
                u_csc._values[ut_nz] = new_val

        # Debug
        expect(u_csr.uncompress() == u_csc.uncompress(), "Bad compute_l_u_factors")

    ###########################################################################
    def _threshold_select(self, m, idx):
    ###########################################################################
        values_cp = list(m._values)
        values_cp.sort(key=abs)
        return abs(values_cp[idx])

    ###########################################################################
    def _is_converged(self):
    ###########################################################################
        nrows = self._A.nrows()
        return self._L_csr.nnz() == nrows  or self._U_csr.nnz() == nrows

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        it = 0
        while not converged:
            LU_csr = self._L_csr * self._U_csr
            print("LU")
            print(LU_csr)

            l_new_csr, u_new_csr = self._add_candidates(LU_csr)

            print("candidates")
            print(l_new_csr)
            print(u_new_csr)

            u_new_csc = CSC(u_new_csr)
            print("u_new_csc")
            print(u_new_csc)

            self._compute_l_u_factors(l_new_csr, u_new_csr, u_new_csc)

            print("LU factors")
            print(l_new_csr)
            print(u_new_csr)

            l_nnz = l_new_csr.nnz()
            u_nnz = u_new_csr.nnz()
            l_filter_rank = max(0, l_nnz - self._l_nnz_limit - 1)
            u_filter_rank = max(0, u_nnz - self._u_nnz_limit - 1)

            # select threshold to remove smallest candidates
            l_threshold = self._threshold_select(l_new_csr, l_filter_rank)
            u_threshold = self._threshold_select(u_new_csr, u_filter_rank)

            # remove smallest candidates from L' and U', storing the
            # results in the original objects
            self._L_csr = l_new_csr.filter_pred(lambda row, col, val: abs(val) >= l_threshold or row == col)
            self._U_csr = u_new_csr.filter_pred(lambda row, col, val: abs(val) >= u_threshold or row == col)
            # expect(self._L_csr.nnz() == l_nnz-l_filter_rank, "Bad filter")
            # expect(self._U_csr.nnz() == u_nnz-u_filter_rank, "Bad filter")

            print(f"After ranks {l_filter_rank}, {u_filter_rank} and threshholds {l_threshold}, {u_threshold}")
            print(self._L_csr)
            print(self._U_csr)

            self._compute_l_u_factors(self._L_csr, self._U_csr, CSC(self._U_csr))

            print("LU factors 2")
            print(self._L_csr)
            print(self._U_csr)

            converged = self._is_converged()

            it += 1
            expect(it < 5, "Not converging")

        print(f"Converged in {it} iterations")

###############################################################################
def parilut(rows, cols, pct_nz, seed, hardcoded):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"{rows} != {cols}. Only square matrices allowed")
    expect(pct_nz > 0 and pct_nz <= 100, f"Bad pct_nz {pct_nz}")

    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randrange(sys.maxsize)
        random.seed(seed)

    try:
        if hardcoded is not None:
            A = SparseMatrix.get_hardcode(hardcoded)
        else:
            A = SparseMatrix(rows, cols, pct_nz)

        print("Original matrix")
        print(A)

        pilut = PARILUT(A)

        pilut.main()
    except SystemExit as e:
        print(f"Encounter error with seed {seed}", file=sys.stderr)
        raise e
