import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from inverted_pendulum import InvertedPendulum

class InvertedPendulumVisualizer:
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.fig = plt.figure(figsize=(12, 8))
        
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 5])
        
        # Create axes for the pendulum visualization and for plotting data
        self.ax_main = self.fig.add_subplot(gs[1, 0])
        self.ax_angle = self.fig.add_subplot(gs[1, 1])
        self.ax_position = self.fig.add_subplot(gs[1, 2])

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

        self.ax_angle.set_title("Pendulum Angle")
        self.ax_angle.set_xlabel("Time (s)")
        self.ax_angle.set_ylabel("Angle (rad)")
        self.ax_angle.set_ylim(np.pi - 0.349, np.pi + 0.349)

        self.ax_position.set_title("Cart Position")
        self.ax_position.set_xlabel("Time (s)")
        self.ax_position.set_ylabel("Position (m)")
        self.ax_position.set_ylim(0, 1)

        self.time_data = []
        self.angle_data = []
        self.position_data = []

        self.angle_line, = self.ax_angle.plot([], [], 'r-')
        self.position_line, = self.ax_position.plot([], [], 'b-')

        self.last_voltage = 0
        self.time_elapsed = 0

        self.text_ax = self.fig.add_subplot(gs[0, :])
        self.text_ax.axis('off')
        self.info_text = self.text_ax.text(0.5, 0.5, '', transform=self.text_ax.transAxes, fontname='Courier', fontsize=10, ha='center', va='center')

    def update(self, frame):
        self.pendulum.simulate_step(self.last_voltage)
        self.time_elapsed += 1
        theta, omega, cart_x, cart_v = self.pendulum.state

        self.cart.set_xy((cart_x - self.cart_width / 2, -0.05))
        self.line.set_data([cart_x, cart_x + self.pendulum.l * np.sin(theta)], [0, -self.pendulum.l * np.cos(theta)])

        self.time_data.append(self.time_elapsed)
        self.angle_data.append(theta)
        self.position_data.append(cart_x)

        self.angle_line.set_data(self.time_data, self.angle_data)
        self.position_line.set_data(self.time_data, self.position_data)

        info_template = (
            f'Angle (rad): {theta:.2f}\n'
            f'Angular velocity (rad/s): {omega:.2f}\n'
            f'Cart position (m): {cart_x:.2f}\n'
            f'Cart velocity (m/s): {cart_v:.2f}'
        )
        self.info_text.set_text(info_template)

        self.ax_angle.set_xlim(max(0, self.time_elapsed - 50), self.time_elapsed + 10)
        self.ax_position.set_xlim(max(0, self.time_elapsed - 50), self.time_elapsed + 10)

        return [self.cart, self.line, self.angle_line, self.position_line, self.info_text]

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