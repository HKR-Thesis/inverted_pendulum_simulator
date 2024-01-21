import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from inverted_pendulum import InvertedPendulum

class InvertedPendulumVisualizer:
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.fig, self.ax_main = plt.subplots(figsize=(12, 12))  # Adjust figsize as needed

        # Setup main plot (pendulum visualization)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlim(0, 1)
        self.ax_main.set_ylim(-1.5 * self.pendulum.l, 1.5 * self.pendulum.l)
        self.cart_width = 0.2
        self.cart_height = 0.1
        self.cart = patches.Rectangle((self.pendulum.cart_position - self.cart_width / 2, -self.cart_height / 2),
                                      self.cart_width, self.cart_height, fc='black')
        self.ax_main.add_patch(self.cart)
        self.line, = self.ax_main.plot([], [], 'o-', lw=2, markersize=8)
        self.ax_main.axhline(0, color='black', lw=2)

        self.last_voltage = 0
        self.time_elapsed = 0

        # Text for displaying information
        self.info_text = self.ax_main.text(0.5, 1.05, '', transform=self.ax_main.transAxes, fontsize=12, ha='center', va='center')

    def update(self, frame):
        self.pendulum.simulate_step(self.last_voltage)
        self.time_elapsed += 1
        theta, omega, cart_x, cart_v = self.pendulum.state

        self.cart.set_xy((cart_x - self.cart_width / 2, -0.05))
        self.line.set_data([cart_x, cart_x + self.pendulum.l * np.sin(theta)], [0, -self.pendulum.l * np.cos(theta)])

        info_template = (
            f'Angle (rad): {theta:.2f}\n'
            f'Angular velocity (rad/s): {omega:.2f}\n'
            f'Cart position (m): {cart_x:.2f}\n'
            f'Cart velocity (m/s): {cart_v:.2f}'
        )
        self.info_text.set_text(info_template)

        return [self.cart, self.line, self.info_text]

    def key_event(self, event):
        if event.key == 'left':
            self.last_voltage = -self.pendulum.max_voltage
        elif event.key == 'right':
            self.last_voltage = self.pendulum.max_voltage
        else:
            self.last_voltage = 0

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=144, interval=33, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        plt.show()

if __name__ == "__main__":
    pendulum = InvertedPendulum()
    visualizer = InvertedPendulumVisualizer(pendulum)
    visualizer.animate()