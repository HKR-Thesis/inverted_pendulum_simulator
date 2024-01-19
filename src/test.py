from inverted_pendulum import InvertedPendulum
from inverted_pendulum_visualizer import InvertedPendulumVisualizer

if __name__ == "__main__":
    pendulum = InvertedPendulum()
    visualizer = InvertedPendulumVisualizer(pendulum)
    visualizer.animate()