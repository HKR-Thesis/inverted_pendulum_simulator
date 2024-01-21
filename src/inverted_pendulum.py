import numpy as np
from scipy.integrate import solve_ivp

class InvertedPendulum:
    def __init__(self):
        self.g = 9.81  # gravitational acceleration in m/s^2
        self.m = 1.3   # mass of the pendulum in kg
        self.l = 0.45  # length of the pendulum rod, in m
        self.I = self.m * self.l ** 2  # moment of inertia for a rod with mass at the end, in kg*m^2
        self.dt = 0.02 # time step for the simulation, in s
        
        # Additional mass screwed onto the top 1/8th part of the rod
        self.additional_mass = 0.67  # Mass of the additional weight in kg

        # Calculate the position where the additional mass is attached
        attachment_position = self.l * 7 / 8

        # Calculate the moment of inertia for the additional mass
        # Using the point mass formula I = m * r^2
        self.additional_moment_of_inertia = self.additional_mass * attachment_position**2

        # Adjust the total moment of inertia to include the additional mass
        self.I += self.additional_moment_of_inertia
        
        self.max_voltage = 0.8  # maximum voltage, in volts
        self.max_force = 6.5  # maximum force that can be applied by the motor, in N. This is a temporary guess, needs to be verified on robot, calculated using F = τ/r.
        self.voltage_to_force_scale = self.max_force / self.max_voltage
        self.max_theta = np.radians(20)  # max deviation from vertical, in radians
        
        self.track_length = 1.0  # length of the track, in m
        self.cart_position = 0.5  # Cart's initial position on the track, in m

        # Friction and air resistance constants
        self.friction_coefficient = 0.3
        self.air_resistance_coefficient = 0.1
        self.friction_exponent = 1.5

        # Initial state [theta, omega, cart_position, cart_velocity]
        # theta: pendulum angle from the vertical (downward) position, omega: angular velocity
        # cart_position: position of the cart on the track, cart_velocity: velocity of the cart
        self.state = [np.pi, 0.1, 0.5, 0]  # starts upright with a small push
        
    # Apply a voltage to the motor
    # Returns the force applied by the motor
    def apply_voltage(self, voltage):
        # Convert the applied voltage to force
        force = voltage * self.voltage_to_force_scale
        return force

    # Lagrangian for the system
    # state: [theta, omega, cart_position, cart_velocity]
    def lagrangian(self, state):
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

    # Equations of motion for the system
    # Returns the derivatives of the state vector
    # state: [theta, omega, cart_position, cart_velocity]
    def equations_of_motion(self, t, state, applied_force):
        theta, omega, cart_position, cart_velocity = state
        # Use the angle measured from the vertical position instead of the horizontal position
        theta_from_vertical = theta - np.pi
        L = self.lagrangian(state)

        # Derivatives of Lagrangian w.r.t. theta and omega
        dL_dtheta = -self.m * self.g * self.l * np.sin(theta_from_vertical)
        dL_domega = self.I * omega

        # Euler-Lagrange Equation
        domega_dt = (dL_domega - dL_dtheta) / self.I + applied_force * self.l / self.I * np.cos(theta_from_vertical)
        dtheta_dt = omega

        # Implement non-linear friction
        friction_force = self.friction_coefficient * np.sign(cart_velocity) * np.abs(cart_velocity)**self.friction_exponent
        total_force = applied_force - friction_force - self.air_resistance_coefficient * cart_velocity
        dv_dt = total_force / self.m
        dx_dt = cart_velocity

        return [dtheta_dt, domega_dt, dx_dt, dv_dt]

    # Simulate a single step of the pendulum
    # Returns the new state of the system
    def simulate_step(self, voltage=0):
        force = self.apply_voltage(voltage)

        # Integrate the equations of motion
        # The state is integrated from 0 to dt
        solution = solve_ivp(
            lambda t, y: self.equations_of_motion(t, y, force),
            [0, self.dt], self.state, method='RK45', t_eval=[self.dt]
        )
        
        new_state = solution.y[:, -1]
        self.state = self.enforce_constraints(new_state, force)
        return self.state

    # Enforce constraints on the state of the system
    # This is done to prevent the pendulum from going beyond the limits of the track
    # and to prevent the pendulum from swinging too far from the vertical position
    def enforce_constraints(self, state, force):
        theta, omega, cart_position, cart_velocity = state
        theta_from_vertical = theta - np.pi  # Measure angle from the vertical

        # Check if the cart is at the boundaries
        at_boundary = cart_position <= 0 or cart_position >= self.track_length

        if at_boundary:
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
            omega *= -0.375  # Inelastic collision damping
        
        theta_from_vertical = np.clip(theta_from_vertical, -self.max_theta, self.max_theta)

        # Adjust theta back to original representation
        theta = theta_from_vertical + np.pi
        return [theta, omega, cart_position, cart_velocity]
