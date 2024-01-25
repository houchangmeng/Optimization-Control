import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from dynamics import CartPoleDynamics
from cartpole_env import CartPole
from copy import deepcopy

from utils import plot


class Constraint_ILQR:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Parameters.
        """
        self.Q = np.diag([1000, 10, 1000, 10])
        self.R = np.diag([0.1])
        self.Nh = 50  # horizion
        self.Nx = 4  # num of states
        self.Nu = 1  # num of inputs

        """
        Goal.
        """

        self.xgoal = np.array([0, 0, 0, 0])
        self.ugoal = np.array([0])

        """
        Constraint.
        """
        self.xbound = np.array([2.4, 100, 2 * np.pi, 100])
        self.ubound = np.array([150])
        self.costQ = np.diag([100, 1, 100, 1])  # state constraint cost weight
        self.costR = np.diag([10])  # # input constraint cost weight

        """
        Stop criterion and iteration parameters.
        """
        self.costMidTol = 1
        self.lineSearchMaxIter = 10
        self.AlILQRIterMax = 10
        self.alpha = 0.1  # linesearch step
        self.dThres = 5.0  # feedforward gain thres

    def x_cons_cost(self, x):
        x = abs(x) - self.xbound
        x = np.max(np.vstack([x, np.zeros_like(x)]), axis=0, keepdims=False)
        return x.T @ self.costQ @ x

    def u_cons_cost(self, u):
        u = abs(u) - self.ubound
        u = np.max(np.vstack([u, np.zeros_like(u)]), axis=0, keepdims=False)
        return u.T @ self.costR @ u

    def dcdxfun(self, x):
        return self.costQ @ (x - self.xbound) * (1 - self.x_in_cons(x))

    def dcdufun(self, u):
        return self.costR @ (u - self.ubound) * (1 - self.u_in_cons(u))

    def u_in_cons(self, u):
        return np.array(u < self.ubound, dtype=np.int32)

    def x_in_cons(self, x):
        return np.array(x < self.xbound, dtype=np.int32)

    def cost(self, X, U):
        totalcost = 0

        def stage_cost(x, u):
            return (
                (x - self.xgoal).T @ self.Q @ (x - self.xgoal)
                + (u - self.ugoal).T @ self.R @ (u - self.ugoal)
                + self.x_cons_cost(x)
                + self.u_cons_cost(u)
            )

        for x, u in zip(X.T, U.T):
            totalcost += 0.5 * stage_cost(x, u)

        totalcost += 0.5 * stage_cost(X[:, -1], 0.0)
        return totalcost

    def constraintCost(self, X, U):
        total_violation = 0.0

        def stage_violation(x, u):
            return self.x_cons_cost(x) + self.u_cons_cost(u)

        for x, u in zip(X.T, U.T):
            total_violation += stage_violation(x, u)

        total_violation += stage_violation(X[:, -1], 0.0)
        return total_violation

    def backward(self, X, U, lambX, lambU):
        dcdx = self.dcdxfun(X[:, -1])
        Ix = self.x_in_cons(X[:, -1])
        cx = self.x_cons_cost(X[:, -1])

        s = self.Q @ (X[:, -1] - self.xgoal) + dcdx.T @ (lambX[:, -1] + Ix * cx)
        S = self.Q + (dcdx * Ix).T @ (dcdx * Ix)

        deltaV = 0.0

        dList = []
        KList = []

        for k in range(self.Nh - 2, -1, -1):
            A, B = self.dynamics.get_AB_matrix(X[:, k], U[:, k])

            dcdx = self.dcdxfun(X[:, k])
            dcdu = self.dcdufun(U[:, k])
            Ix = self.x_in_cons(X[:, k])
            Iu = self.u_in_cons(U[:, k])
            cx = self.x_cons_cost(X[:, k])
            cu = self.u_cons_cost(U[:, k])

            lx = self.Q @ (X[:, k] - self.xgoal)
            lu = self.R @ (U[:, k] - self.ugoal)

            lxx = self.Q
            luu = self.R
            lux = 0

            Qx = lx + A.T @ s + (dcdx * Ix).T * (cx + lambX[:, k])
            Qu = lu + B.T @ s + (dcdu * Iu).T * (cu + lambU[:, k])

            Qxx = lxx + A.T @ S @ A + (dcdx * Ix).T @ (dcdx * Ix)
            Quu = luu + B.T @ S @ B + (dcdu * Iu).T @ (dcdu * Iu)
            Qux = lux + B.T @ S @ A + (dcdu * Iu) * (dcdx * Ix)

            Qxu = np.transpose(Qux)

            while Quu.item() < 0:
                Quu + 0.01

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

    def updateLamb(self, Xstar, Ustar, lambX, lambU):
        for k in range(self.Nh):
            lambX[:, k] = lambX[:, k] + 1.1 * self.x_cons_cost(Xstar[:, k])
            lambU[:, k] = lambU[:, k] + 1.1 * self.u_cons_cost(Ustar[:, k])
            for i in range(self.Nx):
                lambX[i, k] = max(lambX[i, k], 0)

            for i in range(self.Nu):
                lambU[i, k] = max(lambU[i, k], 0)

        return lambX, lambU

    def search(self, x0):
        U = np.random.random((self.Nu, self.Nh))
        X = np.zeros((self.Nx, self.Nh))
        X[:, 0] = x0

        lambX = np.ones((self.Nx, self.Nh))
        lambU = np.ones((self.Nu, self.Nh))

        al_ilqr_iter = 0
        while al_ilqr_iter < self.AlILQRIterMax:
            for k in range(self.Nh - 1):
                X[:, k + 1] = self.dynamics.forward(X[:, k], U[:, k])

            d = 1000
            nor = 10
            J = self.cost(X, U)

            while True:
                alpha = deepcopy(self.alpha)

                Xinit = deepcopy(X)
                Uinit = deepcopy(U)

                J_linesearch_start = self.cost(Xinit, Uinit)
                dList, KList, deltaV = self.backward(Xinit, Uinit, lambX, lambU)

                X, U = self.forward(X, Xinit, U, Uinit, dList, KList, alpha)

                J = self.cost(X, U)

                # forward with line search

                line_search_iter = 0
                while J >= J_linesearch_start:
                    alpha = 0.5 * alpha
                    X, U = self.forward(X, Xinit, U, Uinit, dList, KList, alpha)
                    J = self.cost(X, U)
                    line_search_iter += 1

                    if line_search_iter > self.lineSearchMaxIter:
                        X = Xinit
                        U = Uinit
                        J = J_linesearch_start
                        break

                d = np.average(np.abs(dList))
                nor = la.norm(self.xgoal - Xinit[:, -1])

                cons_cost = self.constraintCost(X, U)

                print(
                    "iter: %d, cost start: %.2f, cost final %.2f, d %.2f terminal dis %.2f, constraint cost %.2f"
                    % (al_ilqr_iter, J, J_linesearch_start, d, nor, cons_cost)
                )

                if J_linesearch_start >= J and J_linesearch_start - J < self.costMidTol:
                    self.costMidTol = self.costMidTol / 10.0
                    break

                if d < self.dThres:
                    self.dThres = self.dThres / 2.0
                    break

            if cons_cost < 1e-3:
                break

            if al_ilqr_iter > 10:
                break

            al_ilqr_iter += 1

            self.costQ = self.costQ * 1.628
            self.costR = self.costR * 1.628
            lambX, lambU = self.updateLamb(X, U, lambX, lambU)

        return X, U

    def swing_up(self):
        x, _ = self.env.reset(options={"angle": np.pi})

        x0 = np.reshape(x, (4,))
        X, U = self.search(x0)

        # save for tracking
        np.savetxt("x_traj_ref.txt", X, fmt="%.4f")
        np.savetxt("u_traj_ref.txt", U, fmt="%.4f")

        # open loop control
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
        plt.show()
        print("--- Constraint ILQR Finish ---")

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
        plt.show()


if __name__ == "__main__":
    cilqr = Constraint_ILQR()

    cilqr.swing_up()
    # cilqr.playback()
