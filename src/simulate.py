from sequential_handler import SequentialHandler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import writers 
from tqdm import tqdm
from body import Body
from typing import List, Optional
import numpy as np
import subprocess
import time

class SimulationRunner:
    def __init__(self, N: int, G: float = 6.6743e-11, epsilon: float = 0.5, dt: float = 0.01, total_time: int = 100, bodies: Optional[List[Body]] = None):
        self.N = N
        self.G = G
        self.epsilon = epsilon
        self.dt = dt
        self.bodies = bodies
        self.handler = SequentialHandler(N=N, G=G, bodies=self.bodies)
        self.steps = int(total_time / dt)
        self.positions = []  # To store the positions of all bodies at each timestep

    def run(self, measure_time=False):
        for _ in tqdm(range(self.steps), desc="Computing simulation...", colour="red"):
            self.handler.update_simulation(self.dt, measure_time=measure_time)
            self.positions.append([body.position for body in self.handler.bodies])

    def create_gif(self, input_video, output_gif, fps=30, width=800):
        command = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f'fps={fps},scale={width}:-1:flags=lanczos',
            '-c:v', 'gif',
            '-loop', '0',
            output_gif
        ]

        subprocess.run(command, check=True)

    def visualize(self, save: bool = False, output_file: str='simulations/simulation', speed_factor: int=1, base_interval:int =50):
        plt.style.use('dark_background')

        fig, ax = plt.subplots()
        ax.set_xlim((-1, 30))
        ax.set_ylim((-1, 30))
        ax.axis('off') # Desactivar los ejes
        # Obtener una paleta de colores
        cmap = plt.get_cmap('viridis', len(self.positions[0]))
        lines = [ax.plot([], [], 'o', color=cmap(i))[0] for i in range(len(self.positions[0]))]
        trails = [ax.plot([], [], '-', linewidth=0.5, color=cmap(i))[0] for i in range(len(self.positions[0]))]
        margin_x, margin_y = 0.02, 0.98  # Ajusta estos valores para cambiar el margen
        ax.annotate(f'GC: {self.G:.2e}', xy=(margin_x, margin_y), xycoords='axes fraction', color='white', verticalalignment='top', horizontalalignment='left')
        ax.annotate(f'$\epsilon$: {self.epsilon}', xy=(margin_x, margin_y - 0.05), xycoords='axes fraction', color='white', verticalalignment='top', horizontalalignment='left')
        ax.annotate(f'N: {len(self.bodies)}', xy=(margin_x, margin_y - 0.10), xycoords='axes fraction', color='white', verticalalignment='top', horizontalalignment='left')


        # Función para actualizar la leyenda
        def update_legend(i):
            # Eliminar la leyenda anterior si existe
            if 'legend' in ax.__dict__:
                ax.legend.remove()
            # Crear la nueva leyenda con la información actualizada
            legend_text = [
                f"GC: {self.G:.2e}",
                f"$\epsilon$: 0.5",
                f"N: {len(self.bodies)}",
            ]
            ax.legend(legend_text, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., facecolor='black')

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
        ani = animation.FuncAnimation(fig, animate, frames=self.steps, init_func=init, blit=False, interval=interval, repeat=True)
        if save: 
            ani_gif = animation.FuncAnimation(fig, animate, frames=frame_generator(), init_func=init, blit=False, interval=50 / speed_factor, repeat=True)
            Writter = animation.writers['ffmpeg']
            writter = Writter(fps = 15, metadata={'artist': 'Me'}, bitrate=1800)
            ani_gif.save(f'{output_file}.mp4', writter)
            time.sleep(5)
            self.create_gif(f'{output_file}.mp4', f'{output_file}.gif')
        plt.show()



def main():
    G = 1  # Constante gravitacional
    dt = 0.01  # Paso de tiempo
    total_time = 60  # Tiempo total de simulación
    measure_time = True  # Flag para medir el tiempo de ejecución


    def four_particles():
        bodies = [
        Body(mass=1, position=[2,5], velocity=[0.5,0.5]),
        Body(mass=1, position=[5,2], velocity=[0.5,0.2]),
        Body(mass=1, position=[3,3], velocity=[0.1, 0.5]),
        Body(mass=1, position=[0.6,2.5], velocity=[0.5,0.5])
        ]
        return bodies
    

    bodies = four_particles()
    # Crear e inicializar el runner de la simulación
    simulation_runner = SimulationRunner(N=len(bodies), G=G, dt=dt, total_time=total_time, bodies=bodies)
    
    # Ejecutar la simulación
    simulation_runner.run(measure_time=measure_time)

    # Calcular y mostrar el tiempo medio por frame
    if measure_time:
        print(f"Tiempo medio por frame: {simulation_runner.handler.avg_fps_time} ms")

    # Visualizar los resultados de la simulación
    print("Displaying simulation...")
    simulation_runner.visualize(save=True, speed_factor=15)

if __name__ == "__main__":
    main()
