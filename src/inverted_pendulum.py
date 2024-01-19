import numpy as np
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
import ipywidgets as widgets
import matplotlib.patches as patches

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
        # Include the term for the force applied to the cart
        domega_dt = (-self.g / self.l * sin_theta + applied_force / self.m / self.l * np.cos(theta)) - self.b / self.I * omega

        # Cart dynamics - only if the cart is moving
        dx_dt = cart_velocity
        dv_dt = applied_force / (self.m + self.I / self.l**2) if self.l != 0 else 0
        
        # Adjust theta back for the state representation
        dtheta_dt_adjusted = dtheta_dt
        
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


class InvertedPendulumVisualizer:
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.fig, self.ax = plt.subplots(figsize=(12, 6))  # Adjust figsize as needed
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, 1)  # Track is 1 meter long
        self.ax.set_ylim(-1.5 * self.pendulum.l, 1.5 * self.pendulum.l)  # Set y-axis limits
        self.cart_width = 0.2
        self.cart_height = 0.1
        self.F_max = 10
        self.last_voltage = 0
        self.cart = patches.Rectangle((self.pendulum.cart_position - self.cart_width / 2, -self.cart_height / 2),
                                      self.cart_width, self.cart_height, fc='blue')
        self.ax.add_patch(self.cart)
        self.line, = self.ax.plot([], [], 'o-', lw=2, markersize=8)
        self.ax.axhline(0, color='black', lw=2)

        # Initialize text annotations at a fixed position on the right side of the plot
        self.angle_text = self.ax.text(1.05, 0.95, '', transform=self.ax.transAxes)
        self.omega_text = self.ax.text(1.05, 0.90, '', transform=self.ax.transAxes)
        self.x_text = self.ax.text(1.05, 0.85, '', transform=self.ax.transAxes)
        self.v_text = self.ax.text(1.05, 0.80, '', transform=self.ax.transAxes)

    def update(self, frame):
        # Simulate the pendulum step
        self.pendulum.simulate_step(self.last_voltage)
        theta, omega, cart_x, cart_v = self.pendulum.state  # Unpack the state

        # Calculate the pendulum rod's end position
        pendulum_end_x = cart_x + self.pendulum.l * np.sin(theta)
        pendulum_end_y = -self.pendulum.l * np.cos(theta)

        # Update the cart and pendulum positions for visualization
        self.cart.set_xy((cart_x - self.cart_width / 2, -0.05))
        self.line.set_data([cart_x, pendulum_end_x], [0, pendulum_end_y])

        # Update text annotations with the current state
        self.angle_text.set_text(f'Angle (rad): {theta:.2f}')
        self.omega_text.set_text(f'Angular velocity (rad/s): {omega:.2f}')
        self.x_text.set_text(f'Cart position (m): {cart_x:.2f}')
        self.v_text.set_text(f'Cart velocity (m/s): {cart_v:.2f}')

        return self.cart, self.line, self.angle_text, self.omega_text, self.x_text, self.v_text

    def key_event(self, event):
        # Set the last applied voltage based on key press
        if event.key == 'left':
            self.last_voltage = -self.pendulum.max_voltage
        elif event.key == 'right':
            self.last_voltage = self.pendulum.max_voltage
        else:
            self.last_voltage = 0  # No voltage applied if any other key is pressed

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=60, interval=60, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        self.ax.set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    pendulum = InvertedPendulum()
    visualizer = InvertedPendulumVisualizer(pendulum)
    visualizer.animate()