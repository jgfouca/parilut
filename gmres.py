
import random, sys, math

from common import expect, Matrix2DR, CSR, enable_debug, get_basis_vector, require, near, set_global_tol

###############################################################################
class GMRES(object):
###############################################################################

    ###########################################################################
    def __init__(self, A, f, max_iter=50, restarts=500, verbose=False):
    ###########################################################################
        expect(A.nrows() == A.ncols(), "GMRES matrix must be square")

        self._A     = A
        self._A_csr = CSR(src_matrix=A)
        self._f     = f
        self._x     = Matrix2DR(A.nrows(), 1)

        # params
        self._max_iter = max_iter
        self._restarts = restarts
        self._verbose = verbose

    ###########################################################################
    def main(self):
    ###########################################################################
        converged = False
        restarts = 0

        n = self._A.nrows()
        k = self._max_iter # min(n, self._max_iter) # Number of arnoldi steps, what to pick for this?
        H = Matrix2DR(k+1, k)
        V = Matrix2DR(n, k+1)

        prev_residual = None
        while restarts < self._restarts and not converged:
            # Start
            r = self._f - (self._A * self._x)
            V.set_column(0, r.normalized())

            # Iterate
            breakdown = False
            for j in range(k):

                Avj = self._A * V.get_column(j)

                # Build hessenberg matrix
                for i in range(j+1):
                    H[i,j] = Avj.dot_product(V.get_column(i))

                sum_ =  V.get_column(0) * H[0,j]
                for i in range(1, j+1):
                    sum_ += V.get_column(i) * H[i,j]

                v_next_col = Avj - sum_

                norm_v = v_next_col.eucl_norm()
                if norm_v == 0:
                    # breakdown
                    breakdown = True
                    break
                else:
                    H[j+1,j] = v_next_col.eucl_norm()
                    V.set_column(j+1, v_next_col.normalized())

            # Form approximate solution
            # vtemp = V.transpose() * V
            # require(vtemp.is_identity(), f"V is not orthogonal\n{V}")
            if not breakdown:
                if self._verbose:
                    print(f"H is:\n{H}\nV is:\n{V}")

                Q, R = H.get_QR_fact()
                if self._verbose:
                    print(f"Q is:\n{H}\nR is:\n{R}")

                beta = r.eucl_norm()
                y = (R.inverse() * beta) * Q.transpose() * get_basis_vector(k+1, 0)
                if self._verbose:
                    print(f"y is:\n{y}")

                self._x = self._x + (V.submatrix(n, k) * y)

                if self._verbose:
                    print(f"New x is:\n{self._x}")

            r = self._f - (self._A * self._x)
            print(f"Residual norm is {r.eucl_norm()}")
            if self._verbose:
                print(f"  with r:\n{r}")

            restarts += 1
            r_norm = r.eucl_norm()
            if near(r_norm, 0):
                converged = True
                prev_residual = r_norm
            else:
                expect(not breakdown, f"Problem broke down but did not converge, residual={r_norm}")
                if prev_residual is not None:
                    improvement = prev_residual - r_norm
                    print(f"  Improvment is {improvement}")
                    if near(improvement, 0, abs_tol=1e-13):
                        converged = True
                        print("    Stagnated!")

                prev_residual = r_norm

        if converged:
            if near(prev_residual, 0):
                print(f"Converged in {k} iterations and {restarts} restarts")
            else:
                print(f"Stagnated in {k} iterations and {restarts} restarts")
        else:
            expect(False, f"Did not converge in {restarts} iterations")

    ###########################################################################
    def check_result(self):
    ###########################################################################
        f = self._A * self._x
        expect(f == self._f, "Solution was not correct")
        print("Solution was correct!")

###############################################################################
def gmres(rows, cols, pct_nz, seed, max_iters, max_restarts, global_tol, hardcoded, debug, verbose):
###############################################################################
    expect(rows > 0, f"Bad rows {rows}")
    expect(cols > 0, f"Bad cols {rows}")
    expect(rows == cols, f"{rows} != {cols}. Only square matrices allowed")
    expect(pct_nz >= 0 and pct_nz <= 100, f"Bad pct_nz {pct_nz}")

    if debug:
        enable_debug()

    if global_tol:
        set_global_tol(global_tol)

    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randrange(sys.maxsize)
        random.seed(seed)

    try:
        if hardcoded is not None:
            A, f = Matrix2DR.get_hardcode_gmres(hardcoded)
        else:
            A = Matrix2DR(rows, cols, pct_nz=pct_nz, diag_lambda=lambda i: pow(rows, 1/3.))
            f = Matrix2DR(rows, 1,    pct_nz=100)
            f = f.normalized()

        if verbose:
            print("A")
            print(A)
            print("f")
            print(f)

        gmr = GMRES(A, f, max_iter=max_iters, restarts=max_restarts, verbose=verbose)

        gmr.main()

        gmr.check_result()

    except BaseException as e:
        print(f"Encountered error with seed {seed}", file=sys.stderr)
        raise e
