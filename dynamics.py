from functools import partial
from jax import jit, jacobian
import jax.numpy as jnp
import numpy as np


class CartPoleDynamics:

    """
    ### Description

    This dynamics class is created for auto calculate jacbian.
    """

    def __init__(self, masscart=1.0, masspole=0.1, lengthpole=0.5, dt=0.02):
        """
        Paramater copied from cartpole environmnet.
        """
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = lengthpole  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.dt = dt  # seconds between state updates
        self.dfdx = jacobian(self.rk4, 0)
        self.dfdu = jacobian(self.rk4, 1)

        self.state = np.zeros(4)

    def forward(self, x, u):
        """
        This function is faster than direct call rk4.
        """
        h = self.dt

        def func(x, u):
            x, x_dot, theta, theta_dot = x

            force = u[0]

            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (
                force + self.polemass_length * theta_dot**2 * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            return np.array([x_dot, xacc, theta_dot, thetaacc])

        f1 = func(x, u)
        f2 = func(x + 0.5 * h * f1, u)
        f3 = func(x + 0.5 * h * f2, u)
        f4 = func(x + h * f3, u)
        x_ret = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

        return x_ret

    @partial(jit, static_argnums=(0,))
    def rk4(self, x, u):
        """
        Discrete system function. get x_k+1 = f(xk, uk)
        """
        h = self.dt

        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + 0.5 * h * f1, u)
        f3 = self.dynamics(x + 0.5 * h * f2, u)
        f4 = self.dynamics(x + h * f3, u)

        return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    @partial(jit, static_argnums=(0,))
    def dynamics(self, x, u):
        """
        Continuous system function. x_dot = f(x, u)
        """

        x, x_dot, theta, theta_dot = x

        force = u[0]
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        return jnp.array([x_dot, xacc, theta_dot, thetaacc])

    @partial(jit, static_argnums=(0,))
    def get_AB_matrix(self, x, u):
        A = self.dfdx(x, u)
        B = self.dfdu(x, u)

        return A, B
