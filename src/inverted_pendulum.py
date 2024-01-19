import numpy as np
from scipy.integrate import odeint

class InvertedPendulum:
    def __init__(self):
        self.g = 9.81  # acceleration due to gravity, in m/s^2
        self.m = 0.5   # mass of the pendulum, in kg
        self.l = 0.25   # length of the pendulum rod, in m
        self.b = 0.1   # damping coefficient, friction at the pivot, in kg*m^2/s
        self.I = self.m * self.l ** 2  # moment of inertia for a rod with mass at the end, in kg*m^2
        self.dt = 0.01 # time step for the simulation, in s
        
        # Additional mass at the top of the rod
        self.additional_mass = 0.1  # Mass of the copper weight in kg
        self.additional_height = 0.05  # Height of the copper weight in meters (5cm)
        self.additional_radius = 0.015  # Radius of the copper weight in meters (3cm)

        # Calculate the moment of inertia for the additional mass
        self.additional_moment_of_inertia = (1/12) * self.additional_mass * self.additional_height**2 + self.additional_mass * self.additional_radius**2

        # Adjust the total moment of inertia to include the additional mass
        self.I += self.additional_moment_of_inertia
        
        self.max_voltage = 0.8  # maximum voltage, in volts
        self.max_force = 10  # maximum force that can be applied by the motor, in N
        self.voltage_to_force_scale = self.max_force / self.max_voltage
        self.max_theta = np.radians(12.25)  # max deviation from vertical, in radians
        
        self.track_length = 1.0  # length of the track, in m
        self.cart_position = 0.5  # Cart's initial position on the track, in m

        # Initial state [theta, omega, cart_position, cart_velocity]
        # theta: pendulum angle from the vertical (downward) position, omega: angular velocity
        # cart_position: position of the cart on the track, cart_velocity: velocity of the cart
        self.state = [np.pi, 0.1, 0.5, 0]  # starts upright with a small push
        
    def apply_voltage(self, voltage):
        # Convert the applied voltage to force
        force = voltage * self.voltage_to_force_scale
        return force

    def equations_of_motion(self, state, t, applied_force):
        theta, omega, cart_position, cart_velocity = state
        
        # Adjust the angle to be measured from the upright position
        theta -= np.pi
        
        # Pendulum dynamics
        sin_theta = np.sin(theta)
        dtheta_dt = omega
        # Cart dynamics - only if the cart is moving
        dx_dt = cart_velocity
        dv_dt = applied_force / (self.m + self.I / self.l**2) if self.l != 0 else 0
        
        # Adjust theta back for the state representation
        dtheta_dt_adjusted = dtheta_dt
        
        # Adjust the moment of inertia to include the additional mass
        I_adjusted = self.I + self.additional_moment_of_inertia

        # Modify the equations of motion to include the adjusted moment of inertia
        domega_dt = (-self.g / self.l * sin_theta + applied_force / self.m / self.l * np.cos(theta)) - self.b / I_adjusted * omega

        return [dtheta_dt_adjusted, domega_dt, dx_dt, dv_dt]

    def simulate_step(self, voltage=0):
        # Calculate the force from the voltage
        force = self.apply_voltage(voltage)
        
        # Integrate the equations of motion over one timestep
        t = np.linspace(0, self.dt, 2)
        new_state = odeint(self.equations_of_motion, self.state, t, args=(force,))[-1]

        # Enforce angle limit with inelastic collision
        collision_damping = 0.5  # Adjust for less bounce
        if new_state[0] > np.pi + self.max_theta:
            new_state[0] = np.pi + self.max_theta
            new_state[1] *= -collision_damping
        elif new_state[0] < np.pi - self.max_theta:
            new_state[0] = np.pi - self.max_theta
            new_state[1] *= -collision_damping

        # Enforce track boundaries for cart position
        new_state[2] = np.clip(new_state[2], 0, self.track_length)
        
        # Update the state with the new values
        self.state = new_state
        return self.state

    def get_cart_position(self):
        # Return the current cart position
        return self.cart_position