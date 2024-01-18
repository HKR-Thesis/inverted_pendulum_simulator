import numpy as np
from scipy.integrate import odeint

class InvertedPendulum:
    def __init__(self):
        self.g = 9.81  # gravity
        self.m = 0.5   # mass of the pendulum
        self.M = 1.0   # mass of the cart
        self.l = 0.5   # length of the pendulum
        self.b = 0.1   # friction coefficient of the cart
        self.I = 0.006 # moment of inertia of the pendulum
        self.dt = 0.01 # time step
        
        self.theta_max_voltage = 4.6  # in volts
        self.theta_min_voltage = 1.6  # in volts
        theta_voltage_range = self.theta_max_voltage - self.theta_min_voltage

        theta_max_radians = np.radians(22.5)
        self.theta_min_radians = -theta_max_radians  # assuming symmetric range
        theta_radians_range = theta_max_radians - self.theta_min_radians

        self.x_max_position = 1.0  # TODO: adjust this based on actual robot dimensions

        x_max_voltage = 0.8  # in volts
        x_full_range = 2 * self.x_max_position  # since -x_max corresponds to 0V and +x_max corresponds to 0.8V

        self.angle_scale_factor = theta_voltage_range / theta_radians_range
        self.position_scale_factor = x_max_voltage / x_full_range
        self.elasticity = 1  # elasticity of the boundary or any collision/sudden stop

        # Initial state [theta, x, omega, v]
        self.state = [np.pi, 0, 0, 0]  # starting with pendulum down and cart at center

    def equations_of_motion(self, state, t, F):
        theta, x, omega, v = state

        # Compute the derivatives
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        denominator = self.I * (self.M + self.m) + self.M * self.m * self.l ** 2 * (cos_theta ** 2)

        # Angular acceleration
        omega_dot = (self.g * sin_theta * (self.M + self.m) - cos_theta * (F + self.m * self.l * omega ** 2 * sin_theta - self.b * v)) / denominator

        # Acceleration of the cart
        x_dot = (F + self.m * self.l * (omega ** 2 * sin_theta - omega_dot * cos_theta) - self.b * v) / (self.M + self.m)

        return [omega, x_dot, omega_dot, v]

    def simulate_step(self, force):
        # Integrate the equations of motion over one timestep
        t = np.linspace(0, self.dt, 2)
        new_state = odeint(self.equations_of_motion, self.state, t, args=(force,))[-1]

        # Check if the cart has hit the boundary
        if new_state[1] <= -self.x_max_position or new_state[1] >= self.x_max_position:
            # Calculate the cart's change in velocity
            delta_v = -2 * new_state[3] * self.elasticity

            # The impulse experienced by the cart
            impulse = delta_v * self.M
            
            # Calculate the effect on the pendulum
            # The impulse causes a change in angular momentum at the pivot point
            # Assume that the impulse acts through the center of mass of the pendulum
            angular_impulse = impulse * self.l

            # Update the pendulum's angular velocity
            # The angular impulse is divided by the moment of inertia to get the change in angular velocity
            new_state[2] += angular_impulse / self.I

            # Correct the cart's position if it goes beyond the boundary
            new_state[1] = np.clip(new_state[1], -self.x_max_position, self.x_max_position)

            # If the collision is perfectly inelastic, the cart stops; otherwise, it bounces back
            new_state[3] = 0 if self.elasticity == 0 else new_state[3]

        self.state = new_state
        return new_state

    def state_to_voltage(self):
        theta, x, _, _ = self.state
        theta_voltage = ((theta - self.theta_min_radians) * self.angle_scale_factor) + self.theta_min_voltage
        x_voltage = (x + 0.5) * self.position_scale_factor + 0.4  # Adjust for the center at 0.5 meters with 0.4V
        return theta_voltage, x_voltage

    def reset(self):
        # Introduce randomness in initial conditions to generalize learning
        initial_theta = np.random.uniform(-np.pi, np.pi)  # Random initial angle
        initial_x = np.random.uniform(-self.x_max_position, self.x_max_position)
        self.state = [initial_theta, initial_x, 0, 0]
        return self.state
