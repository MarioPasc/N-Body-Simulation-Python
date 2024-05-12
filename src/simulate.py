from sequential_handler import SequentialHandler
from parallel_handler import ParallelHandler
from concurrent_handler import ConcurrentThreadHandler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from body import Body
from typing import Union
import numpy as np
import subprocess
import time
import math

class SimulationRunner:
    def __init__(self, dt: float = 0.01, total_time: int = 100, 
                 simulationHandler: Union[SequentialHandler, ParallelHandler] = None) -> None:
        if simulationHandler is None:
            print("Choose a simulation handler")
            return
        self.handler = simulationHandler

        self.bodies = simulationHandler.bodies
        self.N = len(self.bodies)
        self.G = simulationHandler.G
        self.epsilon = simulationHandler.softening
        self.dt = dt
        self.steps = int(total_time / dt)
        self.positions = []  # To store the positions of all bodies at each timestep

    def run(self, measure_time=False):
        for _ in tqdm(range(self.steps), desc="Computing simulation...", colour="red"):
            self.handler.update_simulation(self.dt, measure_time=measure_time)
            self.positions.append(np.array([body.position for body in self.handler.bodies]))

    def visualize(self, save: bool = False, output_file: str='simulations/simulation', speed_factor: int=1, base_interval:int =50):
        #plt.style.use('classic')
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        num = 15
        ax.set_xlim((-1, num))
        ax.set_ylim((-1, num))
        ax.axis('off') # Desactivar los ejes
        # Obtener una paleta de colores
        cmap = plt.get_cmap('viridis', len(self.positions[0]))
        lines = [ax.plot([], [], 'o', color=cmap(i))[0] for i in range(len(self.positions[0]))]
        trails = [ax.plot([], [], '-', linewidth=0.5, color=cmap(i))[0] for i in range(len(self.positions[0]))]
        margin_x, margin_y = 0.02, 0.98  # Ajusta estos valores para cambiar el margen
        color = 'white'
        ax.annotate(f'GC: {self.G:.2e}', xy=(margin_x, margin_y), xycoords='axes fraction', color=color, verticalalignment='top', horizontalalignment='left')
        ax.annotate(f'$\epsilon$: {self.epsilon}', xy=(margin_x, margin_y - 0.05), xycoords='axes fraction', color=color, verticalalignment='top', horizontalalignment='left')
        ax.annotate(f'N: {len(self.bodies)}', xy=(margin_x, margin_y - 0.10), xycoords='axes fraction', color=color, verticalalignment='top', horizontalalignment='left')

        def init():
            for line in lines:
                line.set_data([], [])
            for trail in trails:
                trail.set_data([], [])
            return lines + trails

        def frame_generator():
            for i in range(0, len(self.positions), 10):  
                yield i

        def animate(i):
            for line, trail, positions in zip(lines, trails, np.array(self.positions).transpose(1, 0, 2)):
                x, y = positions[i]
                line.set_data(x, y)
                trail.set_data(positions[:i+1, 0], positions[:i+1, 1])
            return lines + trails

        interval = base_interval / speed_factor
        if save:
            ani_gif = animation.FuncAnimation(fig, animate, frames=frame_generator(), init_func=init, blit=False, interval=50 / speed_factor, repeat=True, cache_frame_data=False)
            writer = animation.PillowWriter(fps=15)
            ani_gif.save(f'{output_file}.gif', writer=writer, dpi=300)
        else:
            ani = animation.FuncAnimation(fig, animate, frames=self.steps, init_func=init, blit=False, interval=interval, repeat=True)

        plt.show()



def main():
    G = 1  # Constante gravitacional
    dt = 0.01  # Paso de tiempo
    total_time = 50  # Tiempo total de simulación
    measure_time = False  # Flag para medir el tiempo de ejecución


    def four_particles():
        bodies = [
        Body(mass=1, position=[2,5], velocity=[0.5,0.5]),
        Body(mass=1, position=[5,2], velocity=[0.5,0.2]),
        Body(mass=1, position=[3,3], velocity=[0.1, 0.5]),
        Body(mass=1, position=[0.6,2.5], velocity=[0.5,0.5])
        ]
        return bodies
    
    def three_particles():
        bodies = [
            Body(mass=1, position=[-1, 0], velocity=[0.2, 0.3]),
            Body(mass=1.5, position=[1, 0], velocity=[0.2, 0.3]),
            #Body(mass=1.6, position=[0, 1], velocity=[0.2, 0.2])
        ]
        return bodies
    
    def lagrange():
        bodies = [
            Body(mass=5, position=[-3, 0], velocity=[math.sqrt(3)/2, -0.5]),
            Body(mass=5, position=[3, 0], velocity=[0, 1]),
            Body(mass=5, position=[0, 3*math.sqrt(3)], velocity=[-math.sqrt(3)/2, -0.5])
        ]
        return bodies

    bodies = three_particles()
    # Crear e inicializar el runner de la simulación
    seq_handler = SequentialHandler(N=len(bodies), G=G, bodies=bodies, softening=.3)
    parallel_handler = ParallelHandler(N=len(bodies), G=G, bodies=bodies, softening=0.3)
    thread_handler = ConcurrentThreadHandler(N=len(bodies), G=G, bodies=bodies, softening=.3)

    simulation_runner = SimulationRunner(dt=dt, total_time=total_time, simulationHandler=parallel_handler)

    # Ejecutar la simulación
    simulation_runner.run(measure_time=measure_time)

    # Calcular y mostrar el tiempo medio por frame
    if measure_time:
        print(f"Tiempo medio por frame: {simulation_runner.handler.avg_fps_time} ms")

    # Visualizar los resultados de la simulación
    print("Displaying simulation...")
    simulation_runner.visualize(save=True, speed_factor=10)


if __name__ == "__main__":
    main()
