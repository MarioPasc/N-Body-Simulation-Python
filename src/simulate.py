from sequential_handler import SequentialHandler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

class SimulationRunner:
    def __init__(self, N, G=6.6743e-11, dt=0.01, total_time=100):
        """
        Initialize the simulation with N bodies, gravitational constant G,
        timestep dt, and total simulation time.
        """
        self.handler = SequentialHandler(N=N, G=G)
        self.dt = dt
        self.steps = int(total_time / dt)
        self.positions = []  # To store the positions of all bodies at each timestep

    def run(self, measure_time=False):
        """
        Run the simulation for the specified number of steps.
        """
        for _ in tqdm(range(self.steps), desc="Computing simulation...", colour="red"):
            self.handler.update_simulation(self.dt, measure_time=measure_time)
            self.positions.append([body.position for body in self.handler.bodies])

    def visualize(self):
        """
        Visualize the simulation using matplotlib.
        """
        fig, ax = plt.subplots()
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 100))

        lines = [ax.plot([], [], 'o')[0] for _ in range(len(self.positions[0]))]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            for j, line in enumerate(lines):
                x, y = self.positions[i][j][:2]  # Assuming 2D for simplicity
                line.set_data(x, y)
            return lines

        ani = animation.FuncAnimation(fig, animate, frames=self.steps, init_func=init, blit=True)
        plt.show()


def main():
    N = 3  # Número de cuerpos
    G = 20  # Constante gravitacional
    dt = 0.00001  # Paso de tiempo
    total_time = 10  # Tiempo total de simulación
    measure_time = False  # Flag para medir el tiempo de ejecución

    # Crear e inicializar el runner de la simulación
    simulation_runner = SimulationRunner(N=N, G=G, dt=dt, total_time=total_time)
    
    # Ejecutar la simulación
    simulation_runner.run(measure_time=measure_time)

    # Calcular y mostrar el tiempo medio por frame
    if measure_time:
        print(f"Tiempo medio por frame: {simulation_runner.handler.avg_fps_time} ms")

    # Visualizar los resultados de la simulación
    print("Visualizando resultados...")
    simulation_runner.visualize()

if __name__ == "__main__":
    main()
