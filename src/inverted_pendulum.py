import numpy as np
from typing import List
from scipy.integrate import solve_ivp

class Action:
    def __init__(self, duty_cycle, direction):
        self.duty_cycle = duty_cycle  # Duty Cycle value for motor speed
        self.direction = direction  # Motor direction ("forward" or "backward")

    def __str__(self):
        return f"Duty Cycle: {self.duty_cycle}, Direction: {self.direction}"


class InvertedPendulum:
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
        self.max_force = 2
        self.voltage_to_force_scale = self.max_force / self.max_voltage
        self.max_theta = np.radians(25)

        self.track_length = 0.5
        self.x = 0.25

        # Friction and air resistance constants
        self.friction_coefficient = 0.3
        self.air_resistance_coefficient = 0.1
        self.friction_exponent = 1.5

        # Starts upright with a small push
        self.state = [np.pi, 0.1, self.x, 0]

    def _calculate_force(self, action: Action) -> float:
        match action.direction:
            case "forward":
                return action.duty_cycle / 100 * self.max_force
            case "backward":
                return -(action.duty_cycle / 100 * self.max_force)
            case "stop":
                return 0
            case _:
                raise ValueError("Invalid direction")            

    def equations_of_motion(
        self, state: List[float], applied_force: float
    ) -> List[float]:
        
        theta, omega, _, x_dot = state
        # Use the angle measured from the vertical position instead of the horizontal position
        theta_from_vertical = theta - np.pi

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

    def simulate_step(self, action: Action) -> tuple[List[float], float, bool]:
        applied_force = self._calculate_force(action)
        # Integrate the equations of motion
        # The state is integrated from 0 to dt
        solution = solve_ivp(
            lambda _, y: self.equations_of_motion(y, applied_force),
            [0, self.dt],
            self.state,
            method="RK45",
            t_eval=[self.dt],
        )

        new_state = solution.y[:, -1]
        self.state = self.enforce_constraints(new_state, action.duty_cycle)
        return (
            self.state,
            self.calculate_reward(self.state),
            self.terminal_state(self.state),
        )

    def enforce_constraints(self, state: List[float], duty_cycle: float) -> List[float]:
        theta, omega, x, x_dot = state
        theta_from_vertical = theta - np.pi  # Measure angle from the vertical

        # Check if the cart is at the boundaries
        at_boundary = x <= 0 or x >= self.track_length

        if at_boundary or duty_cycle == 0:
            # If at boundary, the cart should stop
            x = np.clip(x, 0, self.track_length)
            if x_dot != 0:
                # Calculate impulse due to sudden stop of the cart
                impulse = -x_dot * self.m
                # Apply impulse to change in pendulum's angular velocity
                omega += impulse * self.l / self.I
            x_dot = 0

        # Enforce angle limit with inelastic collision
        if abs(theta_from_vertical) > self.max_theta:
            omega *= -0.55  # Inelastic collision damping

        theta_from_vertical = np.clip(
            theta_from_vertical, -self.max_theta, self.max_theta
        )

        theta = theta_from_vertical + np.pi
        return [theta, omega, x, x_dot]

    def calculate_reward(self, state):
        """
        ### calculate_reward/2
        Calculate the reward for the current state.

        Args:
            state (List[float]): [theta, omega, x, x_dot]

        Returns:
            float: Reward
        """
        theta, _, _, _ = state
        target_angle = np.pi

        angle_difference = np.abs(theta - target_angle)

        reward = 1.0 / (1.0 + angle_difference)

        return reward

    def terminal_state(self, state):
        """
        ### terminal_state/1
        Check if the state is a terminal state.

        Args:
            state (List[float]): [theta, omega, x, x_dot]

        Returns:
            bool: True if the state is terminal, False otherwise.
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
        ### reset/1
        Reset the state of the system to the initial state.

        Returns:
            List[float]: [theta, omega, x, x_dot]
        """
        self.state = [np.pi, 0.1, 0.25, 0]
        return self.state
