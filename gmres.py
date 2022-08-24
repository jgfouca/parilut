
import random, sys, math

from common import expect, SparseMatrix, CSR

###############################################################################
class GMRES(object):
###############################################################################

    ###########################################################################
    def __init__(self, A, f, it=5):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "GMRES matrix must be square")

        self._A     = A
        self._A_csr = CSR(src_matrix=A)
        self._f     = f
        self._x     = SparseMatrix(A.nrows(), 1)

        # params
        self._it = it

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        it = 0

        h = SparseMatrix(self._A.nrows(), self._A.ncols())
        v = SparseMatrix(self._A.nrows(), self._A.ncols())

        while it < self._it and not converged:
            # Start
            r = self._f - (self._A * self._x)
            v.set_column(0, r.normalized())

            # Iterate
            for j in range(self._A.ncols()-1): # Not sure about this range

                Avj = self._A * v.get_column(j)

                # Build hessenberg matrix
                for i in range(j+1):
                    h[i][j] = Avj.dot_product(v.get_column(i))

                sum_ =  v.get_column(0) * h[0][j]
                for i in range(1, j+1):
                    sum_ += v.get_column(i) * h[i][j]

                v_next_col = Avj - sum_

                h[j+1][j] = v_next_col.eucl_norm()
                v.set_column(j+1, v_next_col.normalized())

            # Form approximate solution
            print(f"H is:\n{h}")

            Q, R = h.get_QR_fact()
            y = 4

            it += 1
            converged = True

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
            f = SparseMatrix.get_hardcode(hardcoded)
        else:
            A = SparseMatrix(rows, cols, pct_nz=pct_nz)
            f = SparseMatrix(rows, 1,    pct_nz=pct_nz)
            print(f"JGF\n{f} {pct_nz}")
            f = f.normalized()

        print("A")
        print(A)
        print("f")
        print(f)

        gmr = GMRES(A, f)

        gmr.main()

        if hardcoded is not None:
            gmr.check_hc_result(hardcoded)

    except SystemExit as e:
        print(f"Encountered error with seed {seed}", file=sys.stderr)
        raise e
