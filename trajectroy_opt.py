import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from dynamics import CartPoleDynamics
from cartpole_env import CartPole
import casadi as ca
from utils import plot


class TrajectoryOptimization:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Paramaters.
        """
        self.Q = np.diag([100, 1, 100, 1])
        self.R = np.array([[0.1]])
        self.Nh = 50  # horizon
        self.Nx = 4  # state num
        self.Nu = 1  # input num

        self.dT = self.dynamics.dt

    def cafunc(self):
        """
        Rewrite dynamics function and discrete it in casadi api.
        """
        x = ca.MX.sym("x", 4)
        u = ca.MX.sym("u", 1)

        _, x_dot, theta, theta_dot = x[0], x[1], x[2], x[3]
        force = u
        costheta = ca.cos(theta)
        sintheta = ca.sin(theta)

        """
        Here is real dynamics.
        """
        length = 0.5
        masspole = 0.1
        polemass_length = masspole * length
        gravity = 9.8
        total_mass = 1.1

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        xdot = ca.vertcat(x_dot, xacc, theta_dot, thetaacc)

        """
        Get dynamic function.
        """
        dyn = ca.Function("dyn", [x, u], [xdot], ["x", "u"], ["xdot"])

        xk = ca.MX.sym("xk", 4)
        uk = ca.MX.sym("uk", 1)
        dT = ca.MX.sym("dt", 1)

        """
        Discrete dynamics function.
        """
        k1 = dyn(x=xk, u=uk)["xdot"]
        k2 = dyn(x=xk + 0.5 * dT * k1, u=uk)["xdot"]
        k3 = dyn(x=xk + 0.5 * dT * k2, u=uk)["xdot"]
        k4 = dyn(x=xk + dT * k3, u=uk)["xdot"]
        xk1 = xk + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        rk4 = ca.Function("rk4", [xk, uk, dT], [xk1], ["xk", "uk", "dt"], ["xk1"])

        return dyn, rk4

    def transcription(self, state):
        """
        Direct Transcription/ Multiple Shooting
        """
        dyn, rk4 = self.cafunc()

        opti = ca.Opti()

        N = self.Nh
        X = opti.variable(4, N)
        U = opti.variable(1, N)  # this dim should be N - 1,N for draw plot

        opti.set_initial(X, ca.DM_rand(4, N))
        opti.set_initial(U, ca.DM_rand(1, N))

        dT = self.dT
        costFunction = 0
        opti.subject_to(X[:, 0] == ca.vec(state))
        opti.subject_to(X[:, -1] == ca.vec([0, 0, 0, 0]))
        opti.subject_to(U[0, 0] == 0)
        opti.subject_to(U[0, -2] == 0)

        for k in range(N - 1):
            Xk1 = rk4(xk=X[:, k], uk=U[:, k], dt=dT)["xk1"]
            opti.subject_to(Xk1 == X[:, k + 1])
            costFunction += U[:, k].T @ self.R @ U[:, k]

            opti.subject_to(opti.bounded(-200, U[0, k], 200))
            opti.subject_to(opti.bounded(-4, X[0, k], 4))

        opti.minimize(costFunction)
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=False)
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        print("solve problem. ")

        return sol.value(X), sol.value(U)

    def shooting_time(self, state):
        """
        This function add time as a decision variable.
        The objective is to minimize the total time / or input.
        """
        dyn, rk4 = self.cafunc()

        opti = ca.Opti()

        N = self.Nh
        X = opti.variable(4, N)
        U = opti.variable(1, N)  # this dim should be N - 1,N for draw plot

        opti.set_initial(X, ca.DM_rand(4, N))
        opti.set_initial(U, ca.DM_rand(1, N))

        dT = opti.variable(1, N)
        # dT = 0.02

        costFunction = 0
        opti.subject_to(X[:, 0] == ca.vec(state))
        opti.subject_to(X[:, -1] == ca.vec([0, 0, 0, 0]))

        opti.subject_to(U[0, 0] == 0)
        opti.subject_to(U[0, -2] == 0)

        for k in range(N - 1):
            # single_shooting method
            Xk1 = rk4(xk=X[:, k], uk=U[:, k], dt=dT[:, k])["xk1"]
            opti.subject_to(Xk1 == X[:, k + 1])

            costFunction += dT[:, k].T @ dT[:, k]

            # costFunction += U[:, k].T @ self.R @ U[:, k]

            opti.subject_to(opti.bounded(-200, U[0, k], 200))
            opti.subject_to(opti.bounded(-4, X[0, k], 4))

            opti.subject_to(opti.bounded(0, dT[0, k], ca.inf))

        opti.minimize(costFunction)

        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=1)
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        print("total time", np.sum(sol.value(dT)))

        return sol.value(X), sol.value(U), sol.value(dT)

    def collection(self, state):
        """
        this function use collection method to optimize full trajectory.
        """
        dyn, rk4 = self.cafunc()

        opti = ca.Opti()

        N = self.Nh
        dT = self.dT
        X = opti.variable(4, N)
        U = opti.variable(1, N)  # this dim should be N - 1,N for draw plot

        costFunction = 0
        opti.subject_to(X[:, 0] == ca.vec(state))
        opti.subject_to(X[:, -1] == ca.vec([0, 0, 0, 0]))

        opti.subject_to(U[0, 0] == 0)
        opti.subject_to(U[0, -2] == 0)

        fnext = dyn(x=X[:, 0], u=U[:, 0])["xdot"]
        for k in range(N - 1):
            # cubic hermite polynomial
            fcurr = fnext
            fnext = dyn(x=X[:, k + 1], u=U[:, k + 1])["xdot"]
            Xmid = 0.5 * (X[:, k] + X[:, k + 1]) + (dT / 8) * (fcurr - fnext)
            Umid = 0.5 * (U[:, k] + U[:, k + 1])
            fmid = dyn(x=Xmid, u=Umid)["xdot"]
            opti.subject_to(
                dT / 6.0 * (fcurr + 4 * fmid + fnext) == X[:, k + 1] - X[:, k]
            )
            costFunction += (
                U[:, k].T @ self.R @ U[:, k] + 4 * U[:, k + 1].T @ self.R @ U[:, k + 1]
            )
            # sometimes add this constraint can not find solution.
            # opti.subject_to(opti.bounded(-4, X[0, k], 4))
            # opti.subject_to(opti.bounded(-200, U[0, k], 200))

        opti.minimize(costFunction)

        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)

        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        print("solve problem. ")

        return sol.value(X), sol.value(U)

    def playback(self, X, T=None):
        import pygame

        _, traj_len = X.shape

        if T is None:
            T = np.ones((traj_len)) * 0.02

        for i in range(traj_len):
            self.env.render(X[:, i])

            pygame.time.delay(int(T[i] * 1000))  # 0.02 sec = 20 milliseconds

    def swing_up(self):
        """
        give an initial state, final state is [0.0, 0.0, 0.0, 0.0], use above functions to find best tarjectory.
        """

        import pygame

        x, _ = self.env.reset(options={"angle": np.pi})

        """
        trajectory optimize.
        """
        # X, U = self.collection(x)

        X, U = self.transcription(x)

        T = None

        # X, U, T = self.shooting_time(x)

        self.playback(X, T)

        """
        Save for tracking
        """
        U = np.expand_dims(U, 0)
        x_traj = np.array(X)
        u_traj = np.array(U)
        np.savetxt("x_traj_nominal.txt", x_traj, fmt="%.4f")
        np.savetxt("u_traj_nominal.txt", u_traj, fmt="%.4f")

        plot(X, U)
        plt.show()


if __name__ == "__main__":
    trajopt = TrajectoryOptimization()
    trajopt.swing_up()
