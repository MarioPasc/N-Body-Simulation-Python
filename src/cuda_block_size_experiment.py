import numpy as np
import matplotlib.pyplot as plt
from parallel_handler import ParallelHandler
from tqdm import tqdm

def run_experiment(block_sizes, simulation_time, num_bodies_list):
    refresh_times_list = []

    for num_bodies in tqdm(num_bodies_list, colour="red", desc="Running experiment"):
        refresh_times = []
        for block_size in block_sizes:
            handler = ParallelHandler(num_bodies)
            handler.update_simulation(0.01)  # Warm-up run

            total_time = 0.0
            num_iterations = 0
            while total_time < simulation_time:
                handler.update_simulation(0.01, measure_time=True)
                total_time += handler.avg_fps_time
                num_iterations += 1

            avg_refresh_time = total_time / num_iterations
            refresh_times.append(avg_refresh_time)
        refresh_times_list.append(refresh_times)

    return refresh_times_list

# Configuración del experimento
block_sizes = [32, 64, 128, 256, 512, 1024, 2048]
simulation_time = 500  # Tiempo de simulación en milisegundos (1 segundo)
num_bodies_list = [50, 150, 250, 350, 450, 550]

# Ejecutar el experimento
refresh_times_list = run_experiment(block_sizes, simulation_time, num_bodies_list)

# Graficar los resultados
plt.figure(figsize=(12, 8))
for i, num_bodies in enumerate(num_bodies_list):
    plt.plot(block_sizes, refresh_times_list[i], marker='o', label=f'{num_bodies} cuerpos')
plt.xlabel('Tamaño de bloque')
plt.ylabel('Tiempo de refresco promedio (ms)')
plt.title('Tiempo de refresco vs Tamaño de bloque')
plt.grid(True)
plt.legend()

# Ajustar los xticks para evitar el solapamiento de las etiquetas
plt.xticks(block_sizes, rotation=45, ha='right')
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig(f"./experiments/CUDA_blockSize.png")
plt.show()