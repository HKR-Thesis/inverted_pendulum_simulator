import numpy as np
from typing import List
from scipy.integrate import solve_ivp


class InvertedPendulum:
    """
    # Inverted Pendulum
    A class for simulating a pendulum and cart system with additional mass.

    Args:
        None

    Attributes:
    | Attribute                              | Description                                                                                                                        |
    | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
    | g (float)                              | Gravitational acceleration in m/s^2.                                                                                               |
    | m (float)                              | Mass of the pendulum in kg.                                                                                                        |
    | l (float)                              | Length of the pendulum rod in meters.                                                                                              |
    | I (float)                              | Moment of inertia for the pendulum in kg*m^2.                                                                                      |
    | dt (float)                             | Time step for the simulation in seconds.                                                                                           |
    | additional_mass (float)                | Mass of the additional weight in kg.                                                                                               |
    | additional_moment_of_inertia (float)   | Moment of inertia for the additional mass in kg*m^2, using mass formula I = m * r^2.                                               |
    | max_voltage (float)                    | Maximum voltage for the motor in volts.                                                                                            |
    | max_force (float)                      | Maximum force that can be applied by the motor in Newtons. Calculated using F = τ/r. TODO: Update with real life measurements!     |
    | voltage_to_force_scale (float)         | Scale factor for converting voltage to force.                                                                                      |
    | max_theta (float)                      | Maximum deviation from the vertical in radians.                                                                                    |
    | track_length (float)                   | Length of the track in meters.                                                                                                     |
    | cart_position (float)                  | Initial position of the cart on the track in meters.                                                                               |
    | friction_coefficient (float)           | Coefficient of friction.                                                                                                           |
    | air_resistance_coefficient (float)     | Coefficient of air resistance.                                                                                                     |
    | friction_exponent (float)              | Exponent for friction calculations.                                                                                                |
    | state (list)                           | Initial state of the system [theta, omega, cart_position, cart_velocity].\n*theta* (float): Pendulum angle from the vertical (downward) position in radians.\n*omega* (float): Angular velocity of the pendulum.\n*cart_position* (float): Position of the cart on the track.\n*cart_velocity* (float): Velocity of the cart.|

    Usage:
        pendulum_simulator = PendulumSimulator()
    """

    def __init__(self) -> None:
        self.g = 9.81
        self.m = 0.17
        self.l = 0.305
        self.I = self.m * self.l**2
        self.dt = 0.02

        self.additional_mass = 0.09

        attachment_position = self.l * 7 / 8

        self.additional_moment_of_inertia = (
            self.additional_mass * attachment_position**2
        )

        # Adjust the total moment of inertia to include the additional mass
        self.I += self.additional_moment_of_inertia

        self.max_voltage = 60
        self.max_force = 1.2
        self.voltage_to_force_scale = self.max_force / self.max_voltage
        self.max_theta = np.radians(25)

        self.track_length = 0.5
        self.cart_position = 0.25

        # Friction and air resistance constants
        self.friction_coefficient = 0.3
        self.air_resistance_coefficient = 0.1
        self.friction_exponent = 1.5

        # Starts upright with a small push
        self.state = [np.pi, 0.1, self.cart_position, 0]

    def apply_voltage(self, voltage: float) -> float:
        """
        ### apply_voltage/2
        Apply a voltage to the motor.
        Returns the force applied by the motor.

        Args:
            voltage (float): Voltage

        Returns:
            float: Force, Convert the applied voltage to force
        """

        force = voltage * self.voltage_to_force_scale
        return force

    def lagrangian(self, state: List[float]) -> float:
        """
        ### lagrangian/2
        Lagrangian for the system.

        Args:
            state (List[float]): [theta, omega, cart_position, cart_velocity]

        Returns:
            float: (L = Kinetic Energy (T) - Potential Energy (V))
        """
        theta, omega, cart_position, cart_velocity = state
        theta -= np.pi  # Adjusting the angle from the upright position

        # Kinetic Energy (T)
        T_cart = 0.5 * self.m * cart_velocity**2
        T_pendulum = 0.5 * self.I * omega**2
        T = T_cart + T_pendulum

        # Potential Energy (V)
        V = self.m * self.g * self.l * (1 - np.cos(theta))

        # Lagrangian (L = T - V)
        L = T - V
        return L

    def equations_of_motion(
        self, t: any, state: List[float], applied_force: float
    ) -> List[float]:
        """
        ### equations_of_motion/4
        Equations of motion for the system.
        Returns the derivatives of the state vector.

        Args:
            t (any): TODO: What is this?
            state (List[float]): [theta, omega, cart_position, cart_velocity]
            applied_force (float): Used in Euler-Lagrange Equation.

        Returns:
            List[float]: [theta, omega, cart_position, cart_velocity]
        """
        theta, omega, cart_position, cart_velocity = state
        # Use the angle measured from the vertical position instead of the horizontal position
        theta_from_vertical = theta - np.pi
        L = self.lagrangian(state)

        # Derivatives of Lagrangian w.r.t. theta and omega
        dL_dtheta = -self.m * self.g * self.l * np.sin(theta_from_vertical)
        dL_domega = self.I * omega

        # Euler-Lagrange Equation
        domega_dt = (
            dL_domega - dL_dtheta
        ) / self.I + applied_force * self.l / self.I * np.cos(theta_from_vertical)
        dtheta_dt = omega

        # Implement non-linear friction
        friction_force = (
            self.friction_coefficient
            * np.sign(cart_velocity)
            * np.abs(cart_velocity) ** self.friction_exponent
        )
        total_force = (
            applied_force
            - friction_force
            - self.air_resistance_coefficient * cart_velocity
        )
        dv_dt = total_force / self.m
        dx_dt = cart_velocity

        return [dtheta_dt, domega_dt, dx_dt, dv_dt]

    def simulate_step(self, voltage=0) -> List[float]:
        """
        ### simulate_step/2
        Simulate a single step of the pendulum
        Returns the new state of the systems

        Args:
            voltage (int, optional): Voltage. Defaults to 0.

        Returns:
            List[float]: [theta, omega, cart_position, cart_velocity]
        """
        force = self.apply_voltage(voltage)

        # Integrate the equations of motion
        # The state is integrated from 0 to dt
        solution = solve_ivp(
            lambda t, y: self.equations_of_motion(t, y, force),
            [0, self.dt],
            self.state,
            method="RK45",
            t_eval=[self.dt],
        )

        new_state = solution.y[:, -1]
        self.state = self.enforce_constraints(new_state, voltage)
        return self.state

    def enforce_constraints(self, state: List[float], voltage: float) -> List[float]:
        """
        ### enforce_constraints/3
        Enforce constraints on the state of the system.
        This is done to prevent the pendulum from going beyond the limits of the track
        and to prevent the pendulum from swinging too far from the vertical position.

        Args:
            state (List[float]): [theta, omega, cart_position, cart_velocity]
            force (float): TODO: Unused parameter?

        Returns:
            List[float]: [theta, omega, cart_position, cart_velocity]. Adjusted theta back to original representation.
        """
        theta, omega, cart_position, cart_velocity = state
        theta_from_vertical = theta - np.pi  # Measure angle from the vertical

        # Check if the cart is at the boundaries
        at_boundary = cart_position <= 0 or cart_position >= self.track_length

        if at_boundary or voltage == 0:
            # If at boundary, the cart should stop
            cart_position = np.clip(cart_position, 0, self.track_length)
            if cart_velocity != 0:
                # Calculate impulse due to sudden stop of the cart
                impulse = -cart_velocity * self.m
                # Apply impulse to change in pendulum's angular velocity
                omega += impulse * self.l / self.I
            cart_velocity = 0

        # Enforce angle limit with inelastic collision
        if abs(theta_from_vertical) > self.max_theta:
            omega *= -0.5  # Inelastic collision damping

        theta_from_vertical = np.clip(
            theta_from_vertical, -self.max_theta, self.max_theta
        )

        theta = theta_from_vertical + np.pi
        return [theta, omega, cart_position, cart_velocity]
