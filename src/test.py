import numpy as np
import matplotlib.pyplot as plt
from inverted_pendulum import InvertedPendulum

def run_simulation(control_input):
    # Initialize the pendulum system
    pendulum = InvertedPendulum()

    # Number of simulation steps
    num_steps = 500

    # Lists to store angle and position history
    angle_history = []
    position_history = []

    # Run the simulation
    for step in range(num_steps):
        # Use the provided control input (force)
        force = control_input(step, angle_history, position_history)

        # Simulate one step
        pendulum.simulate_step(force)

        # Store the current angle and position
        theta, x, _, _ = pendulum.state
        angle_history.append(theta)
        position_history.append(x)

    return angle_history, position_history

def main():
    # Experiment 1: Constant Force
    def constant_force(step, angle_history, position_history):
        return 10  # Constant force; replace with control logic as needed
    angle_history1, position_history1 = run_simulation(constant_force)

    # Experiment 2: Random Force
    def random_force(step, angle_history, position_history):
        return np.random.uniform(-15, 15)  # Random force within a range
    angle_history2, position_history2 = run_simulation(random_force)

    # Experiment 3: Proportional Control
    def proportional_control(step, angle_history, position_history):
        # Implement a simple proportional control based on pendulum angle
        desired_angle = 0.0  # Desired upright position
        kp = 5.0  # Proportional gain
        error = desired_angle - angle_history[step-1] if step > 0 else desired_angle
        return kp * error
    angle_history3, position_history3 = run_simulation(proportional_control)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Experiment 1: Constant Force
    plt.subplot(3, 2, 1)
    plt.plot(angle_history1, label='Pendulum Angle')
    plt.xlabel('Step')
    plt.ylabel('Angle (rad)')
    plt.title('Experiment 1: Constant Force')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(position_history1, label='Cart Position')
    plt.xlabel('Step')
    plt.ylabel('Position (m)')
    plt.title('Cart Position Over Time')
    plt.legend()

    # Experiment 2: Random Force
    plt.subplot(3, 2, 3)
    plt.plot(angle_history2, label='Pendulum Angle')
    plt.xlabel('Step')
    plt.ylabel('Angle (rad)')
    plt.title('Experiment 2: Random Force')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(position_history2, label='Cart Position')
    plt.xlabel('Step')
    plt.ylabel('Position (m)')
    plt.title('Cart Position Over Time')
    plt.legend()

    # Experiment 3: Proportional Control
    plt.subplot(3, 2, 5)
    plt.plot(angle_history3, label='Pendulum Angle')
    plt.xlabel('Step')
    plt.ylabel('Angle (rad)')
    plt.title('Experiment 3: Proportional Control')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(position_history3, label='Cart Position')
    plt.xlabel('Step')
    plt.ylabel('Position (m)')
    plt.title('Cart Position Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
