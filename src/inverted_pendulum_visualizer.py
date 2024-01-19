import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from IPython.display import display
from inverted_pendulum import InvertedPendulum

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
                                      self.cart_width, self.cart_height, fc='black')
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
        ani = FuncAnimation(self.fig, self.update, frames=144, interval=33, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        self.ax.set_aspect('equal')
        plt.show()

