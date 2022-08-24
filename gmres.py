
import random, sys, math

from common import expect, SparseMatrix, CSR

###############################################################################
class GMRES(object):
###############################################################################

    ###########################################################################
    def __init__(self, A, fill_in_limit=0.75, it=5, use_approx_select=False):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "GMRES matrix must be square")

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
    def main(self):
    ###########################################################################
        converged = False
        it = 0
        while it < self._it and not converged:
            continue

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
                expect(math.isclose(l[row_idx][col_idx], expected_l[row_idx][col_idx], abs_tol=tol),
                       f"L[{row_idx}][{col_idx}] did not have expected value.")
                expect(math.isclose(u[row_idx][col_idx], expected_u[row_idx][col_idx], abs_tol=tol),
                       f"U[{row_idx}][{col_idx}] did not have expected value.")

        print(f"hardcoded result {matrix_id} check passed.")

###############################################################################
def gmres(rows, cols, pct_nz, seed, hardcoded):
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

        gmr = GMRES(A)

        gmr.main()

        if hardcoded is not None:
            gmr.check_hc_result(hardcoded)

    except SystemExit as e:
        print(f"Encountered error with seed {seed}", file=sys.stderr)
        raise e
