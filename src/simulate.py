from sequential_handler import SequentialHandler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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
        for _ in range(self.steps):
            self.handler.update_simulation(self.dt, measure_time=measure_time)
            self.positions.append([body.position for body in self.handler.bodies])

    def visualize(self):
        """
        Visualize the simulation in 3D using matplotlib.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Configurar los límites del gráfico según sea necesario
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_zlim([0, 100])

        lines = [ax.plot([], [], [], 'o')[0] for _ in range(len(self.positions[0]))]

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        def animate(i):
            for j, line in enumerate(lines):
                x, y, z = self.positions[i][j]
                line.set_data(x, y)
                line.set_3d_properties(z)
            return lines

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=self.steps, blit=False)

        plt.show()


def main():
    N = 3  # Número de cuerpos
    G = 1  # Constante gravitacional
    dt = 0.001  # Paso de tiempo
    total_time = 10  # Tiempo total de simulación
    measure_time = False  # Flag para medir el tiempo de ejecución

    # Crear e inicializar el runner de la simulación
    simulation_runner = SimulationRunner(N=N, G=G, dt=dt, total_time=total_time)
    
    # Ejecutar la simulación
    print("Iniciando simulación...")
    simulation_runner.run(measure_time=measure_time)
    print("Simulación completada.")

    # Calcular y mostrar el tiempo medio por frame
    if measure_time:
        print(f"Tiempo medio por frame: {simulation_runner.handler.avg_fps_time} ms")

    # Visualizar los resultados de la simulación
    print("Visualizando resultados...")
    simulation_runner.visualize()

if __name__ == "__main__":
    main()
