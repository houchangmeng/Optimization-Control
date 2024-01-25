import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from dynamics import CartPoleDynamics
from cartpole_env import CartPole
from copy import deepcopy

from utils import plot


class ILQR:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Parameters.
        """
        self.Q = np.diag([1000, 10, 100, 10])
        self.R = np.array([[0.1]])
        self.Nh = 50
        self.Nx = 4
        self.Nu = 1

        """
        Goal.
        """
        self.xgoal = np.array([0, 0, 0, 0])
        self.ugoal = np.array([0])

    def stage_cost(self, x, u):
        return (x - self.xgoal).T @ self.Q @ (x - self.xgoal) + (
            u - self.ugoal
        ).T @ self.R @ (u - self.ugoal)

    def cost(self, X, U):
        totalcost = 0

        for x, u in zip(X.T, U.T):
            totalcost += 0.5 * self.stage_cost(x, u)

        totalcost += 0.5 * self.stage_cost(X[:, -1], 0.0)
        return totalcost

    def backward(self, X, U):
        s = self.Q @ (X[:, -1] - self.xgoal)
        S = self.Q

        deltaV = 0.0

        dList = []
        KList = []

        for k in range(self.Nh - 2, -1, -1):
            A, B = self.dynamics.get_AB_matrix(X[:, k], U[:, k])

            lx = self.Q @ (X[:, k] - self.xgoal)
            lu = self.R @ (U[:, k] - self.ugoal)

            lxx = self.Q
            luu = self.R
            lux = 0

            Qx = lx + A.T @ s
            Qu = lu + B.T @ s

            Qxx = lxx + A.T @ S @ A
            Quu = luu + B.T @ S @ B
            Qux = lux + B.T @ S @ A
            Qxu = np.transpose(Qux)

            d = -np.linalg.inv(Quu) @ Qu
            K = -np.linalg.inv(Quu) @ Qux

            dList.append(d)
            KList.append(K)

            s = Qx + K.T @ Quu @ d + K.T @ Qu + Qux.T @ d
            S = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

            deltaV += 0.5 * d.T @ Quu @ d + d.T @ Qu

        dList.reverse()
        KList.reverse()

        return dList, KList, deltaV

    def forward(self, X, Xinit, U, Uinit, dList, KList, alpha=1.0):
        for k in range(self.Nh - 1):
            deltaX = X[:, k] - Xinit[:, k]
            U[:, k] = Uinit[:, k] + alpha * dList[k] + KList[k] @ deltaX
            X[:, k + 1] = self.dynamics.forward(X[:, k], U[:, k])

        return X, U

    def search(self, x0):
        U = np.random.random((self.Nu, self.Nh))
        X = np.zeros((self.Nx, self.Nh))
        X[:, 0] = x0
        for k in range(self.Nh - 1):
            X[:, k + 1] = self.dynamics.forward(X[:, k], U[:, k])

        d = 1000
        nor = 10
        J = self.cost(X, U)

        while d > 0.1:
            alpha = 0.1

            Xinit = deepcopy(X)
            Uinit = deepcopy(U)

            J_linesearch_start = self.cost(Xinit, Uinit)
            dList, KList, deltaV = self.backward(Xinit, Uinit)

            X, U = self.forward(X, Xinit, U, Uinit, dList, KList, alpha)

            J = self.cost(X, U)

            count = 0
            while J >= J_linesearch_start and count < 10:
                alpha = 0.5 * alpha
                X, U = self.forward(X, Xinit, U, Uinit, dList, KList, alpha)
                J = self.cost(X, U)
                count += 1

            d = abs(np.sum(dList))
            nor = la.norm(self.xgoal - Xinit[:, -1])

            print(
                "J start: %.2f, J final %.2f, d %.2f norm %.2f"
                % (J, J_linesearch_start, d, nor)
            )

        return X, U

    def swing_up(self):
        x, _ = self.env.reset(options={"angle": np.pi})

        x0 = np.reshape(x, (4,))
        X, U = self.search(x0)

        """
        Save for tracking.
        """
        np.savetxt("x_traj_ref.txt", X, fmt="%.4f")
        np.savetxt("u_traj_ref.txt", U, fmt="%.4f")

        """
        Open loop control.
        """
        _, traj_len = X.shape

        x_traj = []
        u_traj = []

        for i in range(traj_len):
            u = U[:, i]
            x, r, t, d, _ = self.env.step(u)

            x = x.reshape(4).astype("float32")

            x_traj.append(x)
            u_traj.append(u)

        x_traj = np.array(x_traj).T
        u_traj = np.array(u_traj).T
        plot(x_traj, u_traj)

        print("--- ILQR Finish ---")

    def playback(self):
        import pygame

        x, _ = self.env.reset(options={"angle": np.pi})

        Xref = np.loadtxt("x_traj_ref.txt")
        Uref = np.loadtxt("u_traj_ref.txt")

        _, traj_len = Xref.shape

        X = Xref
        U = Uref.reshape((1, traj_len))
        for i in range(traj_len):
            self.env.render(X[:, i])
            pygame.time.delay(20)  # 0.02 sec = 20 milliseconds

        plot(X, U)



if __name__ == "__main__":
    ilqr = ILQR()
    ilqr.swing_up()
    # ilqr.playback()
