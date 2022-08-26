
import random, sys, math

from common import expect, SparseMatrix, CSR, enable_debug, get_basis_vector, require, near

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
        restarts = 0

        n = self._A.nrows()
        k = min(n, 50) # Number of arnoldi steps, what to pick for this?
        h = SparseMatrix(k+1, k)
        V = SparseMatrix(n, k+1)

        while restarts < self._it and not converged:
            # Start
            r = self._f - (self._A * self._x)
            V.set_column(0, r.normalized())

            # Iterate
            breakdown = False
            for j in range(k):

                Avj = self._A * V.get_column(j)

                # Build hessenberg matrix
                for i in range(j+1):
                    h[i][j] = Avj.dot_product(V.get_column(i))

                sum_ =  V.get_column(0) * h[0][j]
                for i in range(1, j+1):
                    sum_ += V.get_column(i) * h[i][j]

                v_next_col = Avj - sum_

                norm_v = v_next_col.eucl_norm()
                if near(norm_v, 0):
                    # breakdown
                    breakdown = True
                    break
                else:
                    h[j+1][j] = v_next_col.eucl_norm()
                    V.set_column(j+1, v_next_col.normalized())

            # Form approximate solution
            # vtemp = V.transpose() * V
            # require(vtemp.is_identity(), f"V is not orthogonal\n{V}")
            if not breakdown:
                print(f"H is:\n{h}\nV is:\n{V}")

                Q, R = h.get_QR_fact()
                print(f"Q is:\n{h}\nR is:\n{R}")
                beta = r.eucl_norm()
                y = (R.inverse() * beta) * Q.transpose() * get_basis_vector(k+1, 0)
                print(f"y is:\n{y}")

                self._x = self._x + (V.submatrix(n, n) * y)
                print(f"New x is:\n{self._x}")

            r = self._f - (self._A * self._x)
            print(f"Residual norm is {r.eucl_norm()} with r:\n{r}")
            # How to know if satisfied?

            restarts += 1
            r_norm = r.eucl_norm()
            if near(r_norm, 0):
                converged = True
            else:
                expect(not breakdown, f"Problem broke down but did not converge, residual={r_norm}")

        if converged:
            print(f"Converged in {k} iterations and {restarts} restarts")
        else:
            expect(False, f"Did not converge in {it} iterations")

    ###########################################################################
    def check_result(self):
    ###########################################################################
        f = self._A * self._x
        expect(f == self._f, "Solution was not correct")

###############################################################################
def gmres(rows, cols, pct_nz, seed, hardcoded, debug):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"{rows} != {cols}. Only square matrices allowed")
    expect(pct_nz > 0 and pct_nz <= 100, f"Bad pct_nz {pct_nz}")

    if debug:
        enable_debug()

    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randrange(sys.maxsize)
        random.seed(seed)

    try:
        if hardcoded is not None:
            A, f = SparseMatrix.get_hardcode_gmres(hardcoded)
        else:
            A = SparseMatrix(rows, cols, pct_nz=pct_nz)
            f = SparseMatrix(rows, 1,    pct_nz=100)
            f = f.normalized()

        print("A")
        print(A)
        print("f")
        print(f)

        gmr = GMRES(A, f)

        gmr.main()

        gmr.check_result()

    except BaseException as e:
        print(f"Encountered error with seed {seed}", file=sys.stderr)
        raise e
