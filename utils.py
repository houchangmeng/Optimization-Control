import numpy as np
import matplotlib.pyplot as plt


def plot(X, U):
    _, Nh = X.shape

    pos = X[0, :]
    vel = X[1, :]
    ang = X[2, :]
    ang_dot = X[3, :]

    u = U[0, :]

    t = np.arange(0, Nh, 1)
    plt.subplot(5, 1, 1)
    plt.plot(t, pos, label="x")
    plt.title("x")
    plt.subplot(5, 1, 2)
    plt.plot(t, vel, label="dx")
    plt.subplot(5, 1, 3)
    plt.plot(t, ang, label="θ")
    plt.subplot(5, 1, 4)
    plt.plot(t, ang_dot, label="dθ")
    plt.subplot(5, 1, 5)
    plt.title("u")
    plt.plot(t, u, label="u")

    plt.draw()


def playback(env, Xfile: str, Ufile: str):
    import pygame

    x, _ = env.reset(options={"angle": np.pi})

    Xref = np.loadtxt(Xfile)
    Uref = np.loadtxt(Ufile)

    _, traj_len = Xref.shape

    X = Xref
    U = Uref.reshape((1, traj_len))
    for i in range(traj_len):
        env.render(X[:, i])
        pygame.time.delay(20)  # 0.02 sec = 20 milliseconds

    plot(X, U)
    plt.show()
