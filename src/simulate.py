from sequential_handler import SequentialHandler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from body import Body
from typing import List, Optional
import numpy as np
from matplotlib.animation import FFMpegWriter
class SimulationRunner:
    def __init__(self, N: int, G: float = 6.6743e-11, dt: float =0.01, total_time: int =100, bodies: Optional[List[Body]] = None):
        """
        Initialize the simulation with N bodies, gravitational constant G,
        timestep dt, and total simulation time.
        """
        self.handler = SequentialHandler(N=N, G=G, bodies=bodies)
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

    def visualize(self, save: bool = False, output_file: str='simulation.mp4', speed_factor: int=1, base_interval:int =50):
        fig, ax = plt.subplots()
        ax.set_xlim((-1, 30))
        ax.set_ylim((-1, 30))

        lines = [ax.plot([], [], 'o')[0] for i in range(len(self.positions[0]))]
        trails = [ax.plot([], [], '-', linewidth=0.5)[0] for _ in range(len(self.positions[0]))]

        def init():
            for line in lines:
                line.set_data([], [])
            for trail in trails:
                trail.set_data([], [])
            return lines + trails

        def animate(i):
            for line, trail, positions in zip(lines, trails, np.array(self.positions).transpose(1, 0, 2)):
                x, y = positions[i]
                line.set_data(x, y)
                trail.set_data(positions[:i+1, 0], positions[:i+1, 1])
                for idx, (line, body) in enumerate(zip(lines, self.handler.bodies)):
                    velocity = np.linalg.norm(body.velocity)
                    line.set_label(f'Body {idx}: v={velocity:.2f}')
            ax.legend(loc='upper left')
            return lines + trails

        interval = base_interval / speed_factor
        ani = animation.FuncAnimation(fig, animate, frames=self.steps, init_func=init, blit=False, interval=interval)
        if save: 
            writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save('filename.mp4', writer=writer)
        plt.show()





def main():
    N = 3  # Número de cuerpos
    G = 1  # Constante gravitacional
    dt = 0.1  # Paso de tiempo
    total_time = 50  # Tiempo total de simulación
    measure_time = False  # Flag para medir el tiempo de ejecución
    
    """
    bodyCentral = Body(mass=50, velocity=[0,0], position=[0,0])
    body1 = Body(mass=1.5, position=[5,5], velocity=[-1.0,1.5])
    body2 = Body(mass=2.0, position=[-85,85], velocity=[-0.2,-0.6])
    body3 = Body(mass=0.5, position=[20,20], velocity=[-0.9, 0.9])
    body4 = Body(mass=1.5, position=[208,17], velocity=[0, 0.5])
    body5 = Body(mass=1.0, position=[-122.6, -10.5], velocity=[0, -0.5])
    body6 = Body(mass=1, position=[0, -350.5], velocity=[0.4, 0])
    bodies = [bodyCentral, body1, body2, body3, body4, body5, body6]
    """
    
    bodies = [
        Body(mass=1, position=[2,5], velocity=[0.5,0.5]),
        Body(mass=1, position=[5,2], velocity=[0.5,0.2]),
        Body(mass=1, position=[3,3], velocity=[0.1, 0.5]),
        Body(mass=1, position=[0.6,2.5], velocity=[0.5,0.5])
    ]
    
    # Crear e inicializar el runner de la simulación
    simulation_runner = SimulationRunner(N=len(bodies), G=G, dt=dt, total_time=total_time, bodies=bodies)
    
    # Ejecutar la simulación
    simulation_runner.run(measure_time=measure_time)

    # Calcular y mostrar el tiempo medio por frame
    if measure_time:
        print(f"Tiempo medio por frame: {simulation_runner.handler.avg_fps_time} ms")

    # Visualizar los resultados de la simulación
    print("Visualizando resultados...")
    simulation_runner.visualize(save=True, speed_factor=10)

if __name__ == "__main__":
    main()
