
import random, sys, bisect, math
from dataclasses import dataclass

from common import expect, Matrix2DR, CSR, CSC

###############################################################################
class PARILUT(object):
###############################################################################

    ###########################################################################
    def __init__(self, A, fill_in_limit=0.75, it=5, use_approx_select=False):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "PARILUT matrix must be square")

        self._A     = A
        self._A_csr = CSR(src_matrix=A)
        self._L_csr, self._U_csr = self._A_csr.make_lu() # Initial LU approx

        # params
        self._fill_in_limit = fill_in_limit
        self._it = it
        self._use_approx_select = use_approx_select
        expect(not use_approx_select, "use_approx_select is not supported") # JGF do we want/need this

        self._l_nnz_limit = int(math.floor(self._fill_in_limit * self._L_csr.nnz()))
        self._u_nnz_limit = int(math.floor(self._fill_in_limit * self._U_csr.nnz()))

        self._prev_residual_norm = None

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
                   f"No diagonal in L_csr[{row_idx},{row_idx}]")
            expect(self._U_csr.has(row_idx, row_idx),
                   f"No diagonal in U_csr[{row_idx},{row_idx}]")

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
        JGF: why are COO's needed? Looks like maybe they are needed for omp/cuda impls.

        In the paper, this step is called the "fixed-point" iteration
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
    def _compute_residual_norm(self):
    ###########################################################################
        """
        This is SLOW and not done in gingko in every iteration.

        R = A - LU

        There is some repeated calculation here that's also being done in
        add_candidates.
        """
        LU = self._L_csr * self._U_csr
        R = self._A_csr - LU

        print("Residual")
        print(R)

        # Get residual norm
        residual_norm = 0.
        ait = iter(self._A_csr)
        arow_idx, acol_idx, _ = ait.__next__()
        for row_idx, col_idx, val in R:
            if row_idx == arow_idx and col_idx == acol_idx:
                residual_norm += val*val
                try:
                    arow_idx, acol_idx, _ = ait.__next__()
                except StopIteration:
                    arow_idx, acol_idx, _ = None, None, None

        residual_norm = math.sqrt(residual_norm)

        print(f"Residual norm = {residual_norm}")

        return residual_norm

    ###########################################################################
    def _threshold_select(self, m, idx):
    ###########################################################################
        values_cp = list(m._values)
        values_cp.sort(key=abs)
        return abs(values_cp[idx])

    ###########################################################################
    def _is_converged(self):
    ###########################################################################
        curr_residual = self._compute_residual_norm()
        if self._prev_residual_norm is None:
            self._prev_residual_norm = curr_residual
            return False
        else:
            curr_diff = self._prev_residual_norm - curr_residual
            if curr_diff == 0.0:
                return True
            else:
                self._prev_residual_norm = curr_residual
                return False

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        it = 0
        while it < self._it and not converged:
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

            print(f"After ranks {l_filter_rank}, {u_filter_rank} and threshholds {l_threshold}, {u_threshold}")
            print(self._L_csr)
            print(self._U_csr)

            self._compute_l_u_factors(self._L_csr, self._U_csr, CSC(self._U_csr))

            print("LU factors 2")
            print(self._L_csr)
            print(self._U_csr)

            it += 1
            converged = self._is_converged()

        if converged:
            print(f"Converged in {it} iterations")
        else:
            expect(False, f"Did not converge in {it} iterations")

    ###########################################################################
    def check_hc_result(self, matrix_id):
    ###########################################################################
        if matrix_id == 0:
            expected_l = [
                [1.,       0.,       0.,       0.],
                [2.,       1.,       0.,       0.],
                [0.5,      0.352941, 1.,       0.],
                [0.,       0.,       -1.31897, 1.],
            ]
            expected_u = [
                [1.,       6.,       4.,       7.],
                [0.,       -17.,     -8.,     -6.],
                [0.,       0.,       6.82353,  0.],
                [0.,       0.,       0.,       0.],
            ]
        else:
            expect(False, f"Unknown hardcoded matrix id {matrix_id}")

        l = self._L_csr.uncompress()
        u = self._U_csr.uncompress()
        tol = 1./1000

        for row_idx in range(self._A.nrows()):
            for col_idx in range(self._A.nrows()):
                expect(math.isclose(l[row_idx,col_idx], expected_l[row_idx][col_idx], abs_tol=tol),
                       f"L[{row_idx},{col_idx}] did not have expected value.")
                expect(math.isclose(u[row_idx,col_idx], expected_u[row_idx][col_idx], abs_tol=tol),
                       f"U[{row_idx},{col_idx}] did not have expected value.")

        print(f"hardcoded result {matrix_id} check passed.")

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
            A = Matrix2DR.get_hardcode(hardcoded)
        else:
            A = Matrix2DR(rows, cols, pct_nz)

        print("Original matrix")
        print(A)

        pilut = PARILUT(A)

        pilut.main()

        if hardcoded is not None:
            pilut.check_hc_result(hardcoded)

    except SystemExit as e:
        print(f"Encounter error with seed {seed}", file=sys.stderr)
        raise e
