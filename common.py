
import random, sys, math

_DEBUG = False
###############################################################################
def enable_debug():
###############################################################################
    """
    Drop into pdb if requires fail
    """
    global _DEBUG
    _DEBUG = True

_GLOBAL_TOL = 1e-10
###############################################################################
def set_global_tol(tol):
###############################################################################
    global _GLOBAL_TOL
    _GLOBAL_TOL = tol

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
def require(condition, error_msg, exc_type=RuntimeError, error_prefix="ERROR:"):
###############################################################################
    """
    Similar to assert except uses proper function syntax. Useful for
    checking programming error.
    """
    if _DEBUG:
        if not condition:
            import pdb
            pdb.set_trace()
    else:
        expect(condition, error_msg, exc_type=exc_type, error_prefix=error_prefix)

###############################################################################
def near(val1, val2, abs_tol=_GLOBAL_TOL):
###############################################################################
    return math.isclose(val1, val2, abs_tol=abs_tol)

###############################################################################
def convert_counts_to_sum(the_list):
###############################################################################
    """
    Convert a list of ints into a running sum and return the total sum

    >>> l = [1, 0, 3, 2, 0]
    >>> convert_counts_to_sum(l)
    6
    >>> l
    [0, 1, 1, 4, 6]
    """
    require(the_list[-1] == 0, "Last entry should be empty")
    curr_sum = 0
    for idx, count in enumerate(the_list):
        the_list[idx] = curr_sum
        curr_sum += count

    return curr_sum

###############################################################################
def get_basis_vector(nrows, col_idx):
###############################################################################
    result = Matrix2DR(nrows, 1)
    result[col_idx,0] = 1
    return result

###############################################################################
def get_identity_matrix(nrows):
###############################################################################
    idm = Matrix2DR(nrows, nrows)
    for i in range(nrows):
        idm[i,i] = 1

    return idm

###############################################################################
class Matrix2DR(object):
###############################################################################
    """
    A 2D uncompressed sparse matrix, row major, implemented as a list of lists

    Slow, only used for initialization and debugging
    """

    ###########################################################################
    def __init__(self, rows, cols, pct_nz=0, diag_lambda=None):
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
    @staticmethod
    def get_hardcode(matrix_id):
    ###########################################################################
        """
        Hardcoded matrices for testing
        """
        if matrix_id == 0:
            hardcoded_vals = [
                [1.,   6.,   4., 7.],
                [2.,  -5.,   0., 8.],
                [0.5, -3.,   6., 0.],
                [0.2, -0.5, -9., 0.,],
            ]
            result = Matrix2DR(4, 4)
            for row_idx in range(4):
                for col_idx in range(4):
                    result[row_idx,col_idx] = hardcoded_vals[row_idx][col_idx]

            return result
        else:
            expect(False, f"Unknown hardcoded matrix id {matrix_id}")

        return None

    ###########################################################################
    @staticmethod
    def get_hardcode_gmres(matrix_id):
    ###########################################################################
        """
        Hardcoded matrices for testing
        """
        if matrix_id == 0:
            n = 3
            hardcoded_vals = [
                [1., 0., 0.],
                [0., 2., 0.],
                [0., 0., 3.],
            ]
            hardcoded_vect = [ 1., 1., 1]

        elif matrix_id == 1:
            n = 3
            hardcoded_vals = [
                [4., 0., 0.],
                [0., 4., 0.],
                [0., 0., 4.],
            ]
            hardcoded_vect = [ 1., 1., 1]

        elif matrix_id == 2:
            def diag_func(r):
                if r == 0:
                    return .1
                elif r == 1:
                    return .5
                else:
                    return r-1

            n = 100
            result = Matrix2DR(n, n, pct_nz=0, diag_lambda=diag_func)
            result_v = Matrix2DR(n, 1,    pct_nz=100)
            for i in range(n):
                result_v[i,0] = 1.

            return result, result_v

        else:
            expect(False, f"Unknown hardcoded matrix id {matrix_id}")

        result = Matrix2DR(n, n)
        for row_idx in range(n):
            for col_idx in range(n):
                result[row_idx,col_idx] = hardcoded_vals[row_idx][col_idx]

        result_v = Matrix2DR(n, 1)
        for row_idx in range(n):
            result_v[row_idx,0] = hardcoded_vect[row_idx]

        return result, result_v

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
    def __getitem__(self, tup): return self._matrix[tup[0]][tup[1]]
    ###########################################################################

    ###########################################################################
    def __setitem__(self, tup, val): self._matrix[tup[0]][tup[1]] = val
    ###########################################################################

    ###########################################################################
    def __eq__(self, rhs):
    ###########################################################################
        if self.nrows() != rhs.nrows() or self.ncols() != rhs.ncols():
            return False

        for lhsrow, rhsrow in zip(self, rhs):
            for lhsval, rhsval in zip(lhsrow, rhsrow):
                if not near(lhsval, rhsval):
                    return False

        return True

    ###########################################################################
    def __add__(self, rhs):
    ###########################################################################
        require(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        require(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

        result = Matrix2DR(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[i,j] = self[i,j] + rhs[i,j]

        return result

    ###########################################################################
    def __sub__(self, rhs):
    ###########################################################################
        require(self.nrows() == rhs.nrows(), "Cannot subtract matrix, incompatible dims")
        require(self.ncols() == rhs.ncols(), "Cannot subtract matrix, incompatible dims")

        result = Matrix2DR(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[i,j] = self[i,j] - rhs[i,j]

        return result

    ###########################################################################
    def __mul__(self, rhs):
    ###########################################################################
        if isinstance(rhs, float):
            result = Matrix2DR(self.nrows(), self.ncols())

            for i in range(self.nrows()):
                for j in range(self.ncols()):
                    result[i,j] = self[i,j]*rhs
        else:
            require(self.ncols() == rhs.nrows(),
                    f"Cannot multiply matrix, incompatible dims. LHS cols={self.ncols()}, RHS rows={rhs.nrows()}")

            result = Matrix2DR(self.nrows(), rhs.ncols())

            for i in range(self.nrows()):
                for j in range(rhs.ncols()):
                    curr_sum = 0.0
                    for k in range(self.ncols()):
                        curr_sum += (self[i,k] * rhs[k,j])

                    result[i,j] = curr_sum

        return result

    ###########################################################################
    def make_lu(self):
    ###########################################################################
        """
        Split self into lower and upper triangular matrices. The diagonal vals
        for L should be 1. The diagonals for U should match what is in self or
        be 1.0 if self's is zero.
        """
        l = Matrix2DR(self.nrows(), self.ncols())
        u = Matrix2DR(self.nrows(), self.ncols())

        for row_idx in range(self.nrows()):
            diag_val = 1.
            for col_idx in range(self.ncols()):
                val = self[row_idx,col_idx]
                if col_idx < row_idx:
                    l[row_idx,col_idx] = val
                elif col_idx == row_idx:
                    l[row_idx,col_idx] = 1.
                    if val != 0.0:
                        diag_val = val
                else:
                    u[row_idx,col_idx] = val

            u[row_idx,row_idx] = diag_val

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
        result = Matrix2DR(self.nrows(), self.ncols())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                val = self[i,j]
                if pred(i, j, val):
                    result[i,j] = val

        return result

    ###########################################################################
    def transpose(self):
    ###########################################################################
        result = Matrix2DR(self.ncols(), self.nrows())

        for i in range(self.nrows()):
            for j in range(self.ncols()):
                result[j,i] = self[i,j]

        return result

    ###########################################################################
    def normalized(self):
    ###########################################################################
        require(self.ncols() == 1, "Normalize only works on vectors")

        result = Matrix2DR(self.nrows(), 1)

        enorm = self.eucl_norm()
        require(enorm != 0, f"Got zero euclidean norm for {self}")

        for i in range(self.nrows()):
            result[i,0] = self[i,0] / enorm

        require(near(result.length(), 1.0), f"Normalize did not produce unit vector? Length is {self.length()}")

        return result

    ###########################################################################
    def length(self):
    ###########################################################################
        require(self.ncols() == 1, "Length only works on vectors")

        result = 0
        for i in range(self.nrows()):
            result += self[i,0]*self[i,0]

        return math.sqrt(result)

    ###########################################################################
    def eucl_norm(self):
    ###########################################################################
        require(self.ncols() == 1, "Euclidean norm only works on vectors")

        return math.sqrt(self.dot_product(self))

    ###########################################################################
    def dot_product(self, rhs):
    ###########################################################################
        require(self.ncols() == 1, "Dot product only works on vectors")
        require(rhs.ncols()  == 1, "Dot product only works on vectors")
        require(self.nrows() == rhs.nrows(), "Incompatible vectors for dot product")

        result = 0
        for i in range(self.nrows()):
            result += (self[i,0]*rhs[i,0])

        return result

    ###########################################################################
    def set_column(self, col_idx, vector):
    ###########################################################################
        require(self.nrows() == vector.nrows(), "set_column vector incompatible")
        require(vector.ncols() == 1, "set_column takes a vector")
        require(col_idx < self.ncols(), "col_idx is beyond matrix")

        for i in range(self.nrows()):
            self[i,col_idx] = vector[i,0]

    ###########################################################################
    def get_column(self, col_idx):
    ###########################################################################
        require(col_idx < self.ncols(), "col_idx is beyond matrix")

        result = Matrix2DR(self.nrows(), 1)

        for i in range(self.nrows()):
            result[i,0] = self[i,col_idx]

        return result

    ###########################################################################
    def projection(self, u, a):
    ###########################################################################
        num = u.dot_product(a)
        den = u.dot_product(u)

        return u*(num/den)

    ###########################################################################
    def get_QR_fact(self):
    ###########################################################################
        """
        Gram-Schmidt QR factorization
        """

        U = Matrix2DR(self.nrows(), self.ncols())
        Q = Matrix2DR(self.nrows(), self.ncols())
        R = Matrix2DR(self.nrows(), self.ncols())

        for i in range(self.ncols()):
            a_i = self.get_column(i)
            u_i = self.get_column(i)
            for j in range(i):
                u_i -= self.projection(U.get_column(j), a_i)

            U.set_column(i, u_i)

            Q.set_column(i, u_i.normalized())

        R = Q.transpose() * self
        require(R.is_upper(), f"R should be an upper triangular matrix, it is:\n{R}")
        require((Q.transpose() * Q).is_identity(), f"Q matrix not orthogonal:\n{Q}")

        return Q, R

    ###########################################################################
    def is_upper(self):
    ###########################################################################
        for i in range(self.nrows()):
            for j in range(self.ncols()):
                if i > j and not near(self[i,j], 0.0):
                    return False

        return True

    ###########################################################################
    def is_identity(self):
    ###########################################################################
        return self == get_identity_matrix(self.nrows())

    ###########################################################################
    def scale_row(self, row_idx, scale):
    ###########################################################################
        for j in range(self.ncols()):
            self[row_idx,j] *= scale

    ###########################################################################
    def add_rows(self, src_row, tgt_row, scale):
    ###########################################################################
        for j in range(self.ncols()):
            self[tgt_row,j] += (scale * self[src_row,j])

    ###########################################################################
    def inverse(self):
    ###########################################################################
        require(self.nrows() == self.ncols(), "Inverse only works for square matrices")
        require(self.is_upper(), "Inverse is for upper triangular matrices only")

        result = Matrix2DR(self.nrows(), self.ncols())
        orig   = Matrix2DR(self.nrows(), self.ncols())

        # Copy
        for i in range(self.nrows()):
            for j in range(self.ncols()):
                orig[i,j] = self[i,j]

        # Set result to identity matrix
        for i in range(self.nrows()):
            result[i,i] = 1

        # Get 1's in diagonals
        for i in range(self.nrows()):
            orig_diag = orig[i,i]
            if orig_diag != 1.0 and orig_diag != 0:
                scale = 1/orig[i,i]
                orig.scale_row(i, scale)
                result.scale_row(i, scale)

        # Get zeros in non-diagonals
        for i in range(self.nrows()):
            for j in range(i):
                if not near(orig[j,i], 0):
                    scale = -orig[j,i]
                    orig.add_rows(i, j, scale)
                    result.add_rows(i, j, scale)

        require(orig.is_identity(), "Orig did not become identity")
        require((self * result).is_identity(), "Result is not inverse")

        return result

    ###########################################################################
    def submatrix(self, nrows, ncols):
    ###########################################################################
        require(nrows <= self.nrows(), f"Bad nrows for submatrix, can't go beyond {self.nrows()} but got {nrows}")
        require(ncols <= self.ncols(), f"Bad ncols for submatrix, can't go beyond {self.ncols()} but got {ncols}")

        result = Matrix2DR(nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                result[i,j] = self[i,j]

        return result

###############################################################################
class CompressedMatrix(object):
###############################################################################
    """
    Abstract base class for compressed matrix implementations
    """

    def __init__(self, nrows, ncols, nnz):
        self._nrows, self._ncols = nrows, ncols
        self._values = [0.0] * nnz

    def __str__(self):
        return str(self.uncompress())

    def nrows(self): return self._nrows

    def ncols(self): return self._ncols

    def nnz(self): return len(self._values)

    def uncompress(self): require(False, "Subclass should implement")

###############################################################################
class CSR(CompressedMatrix):
###############################################################################
    """
    Compressed sparse row matrix implementation
    """

    ###########################################################################
    def __init__(self, nrows=0, ncols=0, nnz=0, src_matrix=None):
    ###########################################################################
        if src_matrix is not None:
            require(nrows == 0 and ncols == 0 and nnz == 0, "Do not use with src_matrix")
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

            require(self.uncompress() == src_matrix, "CSR compression failed")

    ###########################################################################
    def get_nnz_range(self, row_idx):
    ###########################################################################
        require(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return self._csr_rows[row_idx], self._csr_rows[row_idx+1]

    ###########################################################################
    def iter_cols_in_row(self, row_idx):
    ###########################################################################
        require(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._csr_cols[slice(*self.get_nnz_range(row_idx))])

    ###########################################################################
    def iter_vals_in_row(self, row_idx):
    ###########################################################################
        require(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return iter(self._values[slice(*self.get_nnz_range(row_idx))])

    ###########################################################################
    def iter_row(self, row_idx):
    ###########################################################################
        require(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
        return zip(self.iter_cols_in_row(row_idx), self.iter_vals_in_row(row_idx))

    ###########################################################################
    def __iter__(self):
    ###########################################################################
        for row_idx in range(self.nrows()):
            for col_idx, val in self.iter_row(row_idx):
                yield row_idx, col_idx, val

    ###########################################################################
    def has(self, row_idx, col_idx_arg):
    ###########################################################################
        require(row_idx < self.nrows(), f"Bad row_idx: {row_idx}")
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
        require(result == (self.uncompress() == rhs.uncompress()), "csr eq failed")

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
        require(l.uncompress() == ul, "make_lu failed for u")
        require(u.uncompress() == uu, "make_lu failed for u")

        return l, u

    ###########################################################################
    def abstract_spgeam(self, rhs, begin_row_cb, entry_cb):
    ###########################################################################
        require(self.nrows() == rhs.nrows(), "Cannot add matrix, incompatible dims")
        require(self.ncols() == rhs.ncols(), "Cannot add matrix, incompatible dims")

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
    def _add_impl(self, rhs, scale=1.):
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
            result._values[result._ncols] = a_val + (b_val * scale)
            result._ncols += 1

        self.abstract_spgeam(rhs, begin_row_cb, entry_cb)

        result._ncols = self.ncols() # Stop using as scratch space

        # Debug checks
        uself, urhs = self.uncompress(), rhs.uncompress()
        uresult = uself + (urhs*scale)
        require(uresult == result.uncompress(), f"CSR addition does not work for:\n{self}\n{rhs}")

        return result

    ###########################################################################
    def __add__(self, rhs):
    ###########################################################################
        return self._add_impl(rhs)

    ###########################################################################
    def __sub__(self, rhs):
    ###########################################################################
        return self._add_impl(rhs, scale=-1.)

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
        require(self.ncols() == rhs.nrows(), "Cannot multiply matrix, incompatible dims")

        result = CSR(self.nrows(), rhs.ncols())

        # first sweep: count nnz for each row
        for lhs_row in range(self.nrows()):
            local_col_idxs = set()
            self._spgemm_insert_row2(local_col_idxs, self, rhs, lhs_row)
            result._csr_rows[lhs_row] = len(local_col_idxs)

        # build row pointers
        nnz = convert_counts_to_sum(result._csr_rows)

        require(len(result._csr_rows) == self.nrows() + 1, "Bad csr_rows")

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
        require(uresult == result.uncompress(), "CSR dot prod does not work")

        return result

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = Matrix2DR(self._nrows, self._ncols)

        for row_idx in range(self._nrows):
            for col_idx, val in self.iter_row(row_idx):
                result[row_idx,col_idx] = val

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

        nnz = convert_counts_to_sum(result._csr_rows)

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
        require(result.uncompress() == self.uncompress().filter_pred(pred),
               "csr filter_pred failed")

        return result

    ###########################################################################
    def sort(self):
    ###########################################################################
        """
        What does sorting a matrix even mean?
        """
        require(False, "Not yet implemented")

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
        require(result.uncompress() == self.uncompress().transpose(), "Tranpose failed")

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

        require(nnz == self.nnz(), f"Bad nnz in CSC init, {nnz} != {self.nnz()}")

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
        require(csr_matrix.uncompress() == self.uncompress(), "CSC contructor broken")

    ###########################################################################
    def get_nnz_range(self, col_idx):
    ###########################################################################
        require(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return self._csc_cols[col_idx], self._csc_cols[col_idx+1]

    ###########################################################################
    def iter_rows_in_col(self, col_idx):
    ###########################################################################
        require(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return iter(self._csc_rows[slice(*self.get_nnz_range(col_idx))])

    ###########################################################################
    def iter_vals_in_col(self, col_idx):
    ###########################################################################
        require(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return iter(self._values[slice(*self.get_nnz_range(col_idx))])

    ###########################################################################
    def iter_col(self, col_idx):
    ###########################################################################
        require(col_idx < self.ncols(), f"Bad col_idx: {col_idx}")
        return zip(self.iter_rows_in_col(col_idx), self.iter_vals_in_col(col_idx))

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = Matrix2DR(self._nrows, self._ncols)

        for col_idx in range(self._ncols):
            for row_idx, val in self.iter_col(col_idx):
                result[row_idx,col_idx] = val

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
        require(csr_matrix.uncompress() == self.uncompress(), "COO constructor broken")

    ###########################################################################
    def uncompress(self):
    ###########################################################################
        result = Matrix2DR(self._nrows, self._ncols)

        for row_idx, col_idx, val in zip(self._coo_rows, self._coo_cols, self._values):
            result[row_idx,col_idx] = val

        return result
