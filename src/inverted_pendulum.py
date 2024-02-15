import numpy as np
from typing import List
from scipy.integrate import solve_ivp

class InvertedPendulum:
    """
    Class representing an inverted pendulum simulator.

    Attributes:
        g (float): Acceleration due to gravity.
        m (float): Mass of the pendulum.
        l (float): Length of the pendulum.
        I (float): Moment of inertia of the pendulum.
        dt (float): Time step for simulation.
        additional_mass (float): Additional mass attached to the pendulum.
        additional_moment_of_inertia (float): Moment of inertia of the additional mass.
        max_voltage (float): Maximum voltage applied to the pendulum.
        max_force (float): Maximum force applied to the pendulum.
        voltage_to_force_scale (float): Scaling factor for converting voltage to force.
        max_theta (float): Maximum angle of the pendulum.
        track_length (float): Length of the track on which the pendulum moves.
        x (float): Initial position of the pendulum on the track.
        friction_coefficient (float): Coefficient of friction.
        air_resistance_coefficient (float): Coefficient of air resistance.
        friction_exponent (float): Exponent for the friction force calculation.
        state (List[float]): Current state of the pendulum [theta, omega, x, x_dot].
    """

    def __init__(self) -> None:
        """
        Initialize the InvertedPendulum object.
        """
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

        self.I += self.additional_moment_of_inertia

        self.max_voltage = 60
        self.max_force = 2
        self.voltage_to_force_scale = self.max_force / self.max_voltage
        self.max_theta = np.radians(25)

        self.track_length = 0.5
        self.x = 0.25

        self.friction_coefficient = 0.3
        self.air_resistance_coefficient = 0.1
        self.friction_exponent = 1.5

        self.state = [np.pi, 0.1, self.x, 0]

    def _calculate_force(self, action: np.intp) -> float:
        """
        Calculate the force based on the given action.

        Args:
            action (np.intp): Action to be performed.

        Returns:
            float: Force to be applied.
        
        Raises:
            ValueError: If the direction is invalid.
        """
        
        # In actuality, this would be the max voltage in either the positive or negative 
        # direction but for the sake of the simulation, we skip one step
        # and just use the max force immediately
        match action:
            case 1:
                return self.max_force
            case 0:
                return -self.max_force
            case _:
                raise ValueError("Invalid direction")            

    def equations_of_motion(
        self, state: List[float], applied_force: float
    ) -> List[float]:
        """
        Calculate the equations of motion for the pendulum.

        Args:
            state (List[float]): Current state of the pendulum [theta, omega, x, x_dot].
            applied_force (float): Force applied to the pendulum.

        Returns:
            List[float]: Derivatives of the state variables [dtheta_dt, domega_dt, dx_dt, dv_dt].
        """
        theta, omega, _, x_dot = state
        theta_from_vertical = theta - np.pi

        dL_dtheta = -self.m * self.g * self.l * np.sin(theta_from_vertical)
        dL_domega = self.I * omega

        domega_dt = (
            dL_domega - dL_dtheta
        ) / self.I + applied_force * self.l / self.I * np.cos(theta_from_vertical)
        dtheta_dt = omega

        # on-linear friction
        friction_force = (
            self.friction_coefficient
            * np.sign(x_dot)
            * np.abs(x_dot) ** self.friction_exponent
        )
        total_force = (
            applied_force
            - friction_force
            - self.air_resistance_coefficient * x_dot
        )
        dv_dt = total_force / self.m
        dx_dt = x_dot

        return [dtheta_dt, domega_dt, dx_dt, dv_dt]

    def simulate_step(self, action: np.intp) -> tuple[List[float], float, bool]:
        """
        Simulate a single step of the pendulum.

        Args:
            action (np.intp): Action to be performed.

        Returns:
            tuple[List[float], float, bool]: New state, reward, and terminal flag.
        """
        applied_force = self._calculate_force(action)
        solution = solve_ivp(
            lambda _, y: self.equations_of_motion(y, applied_force),
            [0, self.dt],
            self.state,
            method="RK45",
            t_eval=[self.dt],
        )
        new_state = solution.y[:, -1]
        self.state = self.enforce_constraints(new_state, 100)
        return (
            self.state,
            self.calculate_reward(self.state),
            self.terminal_state(self.state),
        )

    def enforce_constraints(self, state: List[float], duty_cycle: float) -> List[float]:
        """
        Enforce constraints on the pendulum state.

        Args:
            state (List[float]): Current state of the pendulum [theta, omega, x, x_dot].
            duty_cycle (float): Duty cycle of the pendulum.

        Returns:
            List[float]: Updated state of the pendulum.
        """
        theta, omega, x, x_dot = state
        theta_from_vertical = theta - np.pi

        at_boundary = x <= 0 or x >= self.track_length

        if at_boundary or duty_cycle == 0:
            x = np.clip(x, 0, self.track_length)
            if x_dot != 0:
                impulse = -x_dot * self.m
                omega += impulse * self.l / self.I
            x_dot = 0

        if abs(theta_from_vertical) > self.max_theta:
            omega *= -0.55

        theta_from_vertical = np.clip(
            theta_from_vertical, -self.max_theta, self.max_theta
        )

        theta = theta_from_vertical + np.pi
        return [theta, omega, x, x_dot]

    def calculate_reward(self, state):
        """
        Calculate the reward based on the current state.

        Args:
            state (List[float]): Current state of the pendulum [theta, omega, x, x_dot].

        Returns:
            float: Reward value.
        """
        theta, _, _, _ = state
        target_angle = np.pi

        angle_difference = np.abs(theta - target_angle)

        reward = 1.0 / (1.0 + angle_difference)

        return reward

    def terminal_state(self, state):
        """
        Check if the current state is a terminal state.

        Args:
            state (List[float]): Current state of the pendulum [theta, omega, x, x_dot].

        Returns:
            bool: True if terminal state, False otherwise.
        """
        theta, _, x, _ = state
        max_angle = 3.58
        min_angle = 2.71
        min_position = 0.0
        max_position = 0.5

        if theta >= max_angle or theta <= min_angle:
            return True
        if x >= max_position or x <= min_position:
            return True
        return False

    def reset(self) -> List[float]:
        """
        Reset the pendulum to its initial state.

        Returns:
            List[float]: Initial state of the pendulum [theta, omega, x, x_dot].
        """
        self.state = [np.pi, 0.1, 0.25, 0]
        return self.state
