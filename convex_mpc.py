import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from dynamics import CartPoleDynamics
from cartpole_env import CartPole
from scipy.linalg import block_diag
import casadi as ca

from utils import plot


class QP_MPC:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        self.Q = np.diag([100, 1, 100, 1])
        self.R = np.array([[0.1]])
        self.Nh = 50  # horizon
        self.Nx = 4  # state num
        self.Nu = 1  # input num

    def mpc1(self, x0, A, B):
        """
        this mpc controller can only balance at [0,0,0,0]
        """
        opti = ca.Opti("nlp")

        A = ca.numpy.array(A)
        B = ca.numpy.array(B)

        Nh = self.Nh
        Nx = self.Nx
        Nu = self.Nu

        X = opti.variable(Nx, Nh)
        U = opti.variable(Nu, Nh)

        cost = 0
        X0 = ca.vec(x0)

        opti.subject_to(X[:, 0] == X0)
        opti.subject_to(X[:, -1] == ca.vec([0, 0, 0, 0]))

        """
        xk+1 = dfdx @ xk  + dfdu @ uk  + noise
        """

        for k in range(Nh - 1):
            cost += X[:, k].T @ self.Q @ X[:, k] + U[:, k].T @ self.R @ U[:, k]

            opti.subject_to(opti.bounded(-100, U[0, k], 100))
            opti.subject_to(opti.bounded(-5, X[:, k], 5))

            opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])

        opti.minimize(cost)
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=1)
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            ret = sol.value(U)[[0]]
        except:
            ret = np.array([0])

        return ret

    def mpc(self, x0, A, B):
        Q = self.Q
        R = self.R

        Nh = self.Nh
        Nx = self.Nx
        Nu = self.Nu

        RQ = block_diag(R, Q)
        P = np.kron(np.eye(Nh - 1), RQ)

        C = np.zeros(((Nx * (Nh - 1)), ((Nx + Nu) * (Nh - 1))))

        # dynamics constraints
        C_cell = np.hstack([A, B, -np.eye(Nx)])
        C[:Nx, 0 : Nx + Nu] = C_cell[:, Nx:]
        for i in range(1, Nh - 1):
            C[
                i * Nx : i * Nx + Nx,
                (i - 1) * (Nx + Nu) + Nu : (i - 1) * (Nx + Nu) + Nu + (Nx + Nu + Nx),
            ] = C_cell

        d = np.zeros((Nx * (Nh - 1), 1))
        d[:Nx, 0] = -A @ x0

        # inputs constraints
        U = np.zeros((Nh - 1, (Nu + Nx) * (Nh - 1)))

        for i in range(Nh - 1):
            U[i, i * (Nx + Nu)] = 1

        G1 = np.vstack([U, -U])
        u_rhs = np.ones((Nh - 1, 1)) * 50
        u_rhs = np.vstack([u_rhs, u_rhs])

        # position constraints
        X = np.zeros((Nh - 1, (Nu + Nx) * (Nh - 1)))

        for i in range(Nh - 1):
            X[i, i * (Nx + Nu) + 1] = 1

        G2 = np.vstack([X, -X])
        x_rhs = np.ones((Nh - 1, 1)) * 5
        x_rhs = np.vstack([x_rhs, x_rhs])

        z = cp.Variable(((Nh - 1) * (Nu + Nx), 1))

        obj = cp.Minimize(0.5 * cp.quad_form(z, P))

        constraints = [C @ z == d, G1 @ z <= u_rhs, G2 @ z <= x_rhs]

        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            ret = z.value[0]
            # print(z.value[0])
        except:
            print("can not find the result, current theta is ", x0[2])
            ret = np.array([0.0])

        return ret

    def balance(self):
        x, _ = self.env.reset(options={"angle": 0.2})
        x_linearize = np.zeros(4)
        u_linearize = np.zeros(1)
        A, B = self.dynamics.get_AB_matrix(x_linearize, u_linearize)

        x_traj, u_traj = [], []
        while True:
            u = self.mpc1(x, A, B).astype("float32")

            u += np.random.normal(0, 1)

            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")

            x_traj.append(x)
            u_traj.append(u)

            if t or d:
                break

        X = np.array(x_traj).T
        U = np.array(u_traj).T
        plot(X, U)
        plt.show()

    def swing_up(self):
        """
        Description:
        ---
        xk^ , uk^: linearize point

        xk+1 = f(xk, uk)

        xk+1 - xk+1^ = dfdx @ (xk - xk^) + dfdu @ (uk - uk^)

        xk+1 = dfdx @ (xk - xk^) + dfdu @ (uk - uk^) + xk+1^

        xk+1 = dfdx @ xk  + dfdu @ uk - (dfdx @ xk^ + dfdu @  uk^ - xk+1^)

        xk+1 = A @ xk  + B @ uk - δ(noise)
        """

        x, _ = self.env.reset(options={"angle": np.pi})

        u = np.zeros(1)

        x_traj, u_traj = [], []
        while True:
            A, B = self.dynamics.get_AB_matrix(x, u)
            u = self.mpc(x, A, B).astype("float32")

            # u += np.random.normal(0, 0.1)

            x, r, t, d, _ = self.env.step(u)

            x = x.reshape(4).astype("float32")

            x_traj.append(x)
            u_traj.append(u)

            if t or d:
                break

        X = np.array(x_traj).T
        U = np.array(u_traj).T
        plot(X, U)
        plt.show()


if __name__ == "__main__":
    qp_mpc = QP_MPC()
    # qp_mpc.swing_up()
    qp_mpc.balance()
