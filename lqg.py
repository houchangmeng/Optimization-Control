import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from cartpole_env import CartPole
from dynamics import CartPoleDynamics
from utils import plot


class LQG:
    """
    ### Description

    LQR + EKF

    x_k+1 = f(xk, uk) + Normal(mu, sigma)

    y_k+1 = C @ x_k+1 + Normal(mu, sigma)
    """

    def __init__(self) -> None:
        self.env = CartPole(render_mode="human")
        self.dynamics = CartPoleDynamics()

        """
        Parameters.
        """
        self.Q = np.diag([100, 1, 100, 1])
        self.R = np.array([[0.1]])
        self.Nx = 4
        self.Nu = 1
        self.Nh = 50

        self.W = np.diag([1.0, 1.0, 1.0, 1.0])
        self.V = np.diag([0.2, 0.2])
        """
        Only observation position and angle
        """

        self.ObservationMatrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    @staticmethod
    def dlqr(A, B, Q, R):
        P = Q

        K = 0
        K_last = K + 1.0
        while la.norm(K - K_last) > 1e-2:
            K_last = K
            K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A
            # Formula from DP
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)

        return K

    def ekf(self, A, x, u, P, C, y, W, V):
        x_hat = self.dynamics.forward(x, u)
        P = A @ P @ A.T + W

        z = y - C @ x_hat
        S = C @ P @ C.T + V
        L = P @ C.T @ la.inv(S)

        x_filter = x_hat + L @ z
        P_new = (np.eye(4) - L @ C) @ P @ (np.eye(4) - L @ C).T + L @ V @ L.T

        return x_filter, P_new

    def balance(self):
        
        x_linearize = np.zeros(4)
        u_linearize = np.zeros(1)
        A, B = self.dynamics.get_AB_matrix(x_linearize, u_linearize)

        C = self.ObservationMatrix

        W = self.W
        V = self.V

        k = self.dlqr(A, B, self.Q, self.R)

        x, _ = self.env.reset(options={"angle": 0.2})
        u = (-k @ x).reshape(1).astype("float32")
        y = C @ x
        P = self.W

        while True:
            x_kf, P = self.ekf(A, x, u, P, C, y, W, V)
            u = (-k @ x_kf).reshape(1).astype("float32")
            u += np.random.normal(0, 1)

            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")
            x += np.random.normal(0, 0.1, (4,))
            y = C @ x
            print(x_kf - x)
            if t or d:
                break

    def swing_up(self):
        """
        Brief:
        ---
        xk^ , uk^: linearize point

        xk+1 = f(xk, uk)

        xk+1 - xk+1^ = dfdx @ (xk - xk^) + dfdu @ (uk - uk^)

        xk+1 = dfdx @ (xk - xk^) + dfdu @ (uk - uk^) + xk+1^

        xk+1 = dfdx @ xk  + dfdu @ uk - (dfdx @ xk^ - dfdu @  uk^ + xk+1^)

        Note: here we think the tail terms at a noise

        xk+1 = A @ xk  + B @ uk - δ(noise)
        """

        C = self.ObservationMatrix

        W = self.W
        V = self.V
        Q = self.Q
        R = self.R
        x, _ = self.env.reset(options={"angle": -np.pi})
        u = np.zeros(1)
        y = C @ x
        P = self.W

        x_traj, u_traj, x_filter = [], [], []
        while True:
            A, B = self.dynamics.get_AB_matrix(x, u)
            k = self.dlqr(A, B, Q, R)
            x_kf, P = self.ekf(A, x, u, P, C, y, W, V)
            u = (-k @ x_kf).reshape(1).astype("float32")
            u = np.clip(u, -100, 100)
            u += np.random.normal(0, 1)

            x, r, t, d, _ = self.env.step(u)
            x = x.reshape(4).astype("float32")
            x += np.random.normal(0, 0.1, (4,))
            y = C @ x

            x_filter.append(x_kf)
            x_traj.append(x)
            u_traj.append(u)

            if t or d:
                break

        x_traj = np.array(x_traj).T
        x_filter = np.array(x_filter).T
        u_traj = np.array(u_traj).T

        np.savetxt("x_traj_ref.txt", x_traj, fmt="%.4f")
        np.savetxt("u_traj_ref.txt", u_traj, fmt="%.4f")

        plot(x_traj, u_traj)
        plot(x_filter, u_traj)
        plt.show()

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
    lqg = LQG()
    # lqg.balance()
    lqg.swing_up()
    # lqg.playback()
