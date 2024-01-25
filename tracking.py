import cvxpy as cp
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from dynamics import CartPoleDynamics
from cartpole_env import CartPole
from utils import plot


class Tracking:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Parameters.
        """

        self.Q = np.diag([100, 1, 100, 1])
        self.R = np.array([[0.1]])
        self.Nh = 50
        self.Nx = 4
        self.Nu = 1

        """
        Load reference trajectory and expand it at tail.
        """
        Xref = np.loadtxt("x_traj_ref.txt")
        Uref = np.loadtxt("u_traj_ref.txt")

        Xtail = np.hstack([Xref[:, [-1]] for _ in range(self.Nh)])
        Utail = np.hstack([0 for _ in range(self.Nh)])

        Uref = np.hstack([Uref, Utail])
        self.Xref = np.hstack([Xref, Xtail])
        self.Uref = Uref.reshape((self.Nu, len(Uref)))

        _, self.trajlen = self.Xref.shape

    def mpc(self, Xcur, Ucur, Xref, Uref):
        Nx = self.Nx
        Nh = self.Nh
        Nu = self.Nu
        Q = self.Q
        R = self.R
        X = cp.Variable((Nx, Nh))
        U = cp.Variable((Nu, Nh - 1))

        cost = 0.0
        for i in range(Nh):
            cost += cp.quad_form(X[:, i] - Xref[:, i], Q)

        for i in range(Nh - 1):
            cost += cp.quad_form(U[:, i] - Uref[:, i], R)

        # dynamics constraints
        constraints = []
        constraints.append(X[:, 0] == Xcur)

        for i in range(Nh - 1):
            A, B = self.dynamics.get_AB_matrix(Xref[:, i], Uref[:, i])

            constraints.append(
                X[:, i + 1] - Xref[:, i + 1]
                == A @ (X[:, i] - Xref[:, i]) + B @ (U[:, i] - Uref[:, i])
            )

        constraints.append(X[:, Nh - 1] == Xref[:, Nh - 1])

        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        return U.value[:, 0]

    def lqr(self, Xref, Uref):
        Q = self.Q
        R = self.R

        P = Q
        Klist = []
        for k in range(self.trajlen - 1, -1, -1):
            A, B = self.dynamics.get_AB_matrix(Xref[:, k], Uref[:, k])
            K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
            Klist.append(K)

        Klist.reverse()

        return Klist

    def mpc_tracking(self):
        init_a = self.Xref[2, 0]
        x, _ = self.env.reset(options={"angle": init_a})
        u = np.ones((1,)) * 0.00
        count = 0

        x_traj = []
        u_traj = []
        for k in range(self.trajlen - self.Nh):
            Xref = self.Xref[:, k : k + self.Nh]
            Uref = self.Uref[:, k : k + self.Nh]
            u = self.mpc(Xcur=x, Ucur=u, Xref=Xref, Uref=Uref).astype("float32")

            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")

            x_traj.append(x)
            u_traj.append(u)

            if t or d or la.norm(x) < 1e-2 or count > self.trajlen:
                break

        X = np.array(x_traj).T
        U = np.array(u_traj).T

        plot(self.Xref, self.Uref)
        plot(X, U)
        plt.show()

    def lqr_tracking(self):
        init_a = self.Xref[2, 0]
        x, _ = self.env.reset(options={"angle": init_a})

        Klist = self.lqr(self.Xref, self.Uref)

        x_traj = []
        u_traj = []

        for k in range(self.trajlen - self.Nh):
            u = self.Uref[:, k] - Klist[k] @ (x - self.Xref[:, k])

            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")

            x_traj.append(x)
            u_traj.append(u)

            if t or d or la.norm(x) < 1e-2:
                break

        X = np.array(x_traj).T
        U = np.array(u_traj).T

        plot(self.Xref, self.Uref)
        plot(X, U)
        plt.show()


if __name__ == "__main__":
    tracking = Tracking()
    tracking.mpc_tracking()
    # tracking.lqr_tracking()
