import numpy as np
import numpy.linalg as la
from cartpole_env import CartPole
from dynamics import CartPoleDynamics
from utils import plot


class LQR:
    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Parameters.
        """
        self.Q = np.diag([100, 1, 100, 1])
        self.R = np.array([[0.1]])

    @staticmethod
    def dlqr(A, B, Q, R):
        P = Q

        K = 0
        K_last = K + 1.0
        while la.norm(K - K_last) > 1e-2:
            K_last = K
            K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A

            # Formula from QP
            # P = Q + A.T @ P @ (A - B @ K)

            # Formula from DP
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)

        return K

    def balance(self):
        """
        ### Brief:

        xk+1 = f(xk, uk)

        △xk+1 = dfdx @ (△xk) + dfdu @ (△uk)
        """
        x, _ = self.env.reset(options={"angle": 0.2})
        u = np.ones((1,)) * 0.00
        A, B = self.dynamics.get_AB_matrix(x, u)

        k = self.dlqr(A, B, self.Q, self.R)

        while True:
            u = (-k @ x).reshape(1).astype("float32")
            u += np.random.normal(0, 1)
            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")

            if t or d:
                break

    def swing_up(self):
        """
        ### Brief:

        xk^ , uk^: linearize point

        xk+1 = f(xk, uk)

        xk+1 - xk+1^ = dfdx @ (xk - xk^) + dfdu @ (uk - uk^)

        xk+1 = dfdx @ (xk - xk^) + dfdu @ (uk - uk^) + xk+1^

        xk+1 = dfdx @ xk  + dfdu @ uk - (dfdx @ xk^ - dfdu @  uk^ + xk+1^)

        Note: here we think the tail terms at a noise

        xk+1 = A @ xk  + B @ uk - δ(noise)
        """
        x, _ = self.env.reset(options={"angle": -np.pi})
        u = np.ones((1,)) * 0.00
        A, B = self.dynamics.get_AB_matrix(x, u)

        k = self.dlqr(A, B, self.Q, self.R)

        x_traj = []
        u_traj = []

        step = 0
        while True:
            u = (-k @ x).reshape(1).astype("float32")
            u = np.clip(u, -100, 100)
            u += np.random.normal(0, 0.1)
            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")

            A, B = self.dynamics.get_AB_matrix(x, u)
            k = self.dlqr(A, B, self.Q, self.R)

            step += 1

            x_traj.append(x)
            u_traj.append(u)

            if t or d or step > 200:
                break

        x_traj = np.array(x_traj).T
        u_traj = np.array(u_traj).T
        np.savetxt("x_traj_ref.txt", x_traj, fmt="%.4f")
        np.savetxt("u_traj_ref.txt", u_traj, fmt="%.4f")

        plot(x_traj, u_traj)

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
    lqr = LQR()
    # lqr.balance()
    lqr.swing_up()
    # lqr.playback()
