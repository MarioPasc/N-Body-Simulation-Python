import csv
import matplotlib.pyplot as plt
from simulate import SimulationRunner

class SequentialExperiment:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.results = []

    def run_experiment(self, N_range: range, G: float, dt: float, total_time: int) -> None:
        for N in N_range:
            # Inicializar SimulationRunner para N cuerpos
            simulation_runner = SimulationRunner(N=N, G=G, dt=dt, total_time=total_time)
            simulation_runner.run(measure_time=True)
            
            # Obtener el tiempo promedio de refresco de frames y guardar en la lista de resultados
            avg_fps_time = simulation_runner.handler.avg_fps_time
            self.results.append((N, avg_fps_time))
            print(f"Completed: N={N}, avg_fps_time={avg_fps_time} ms")

    def save_results_to_csv(self) -> None:
        with open(self.output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['N', 'avg_tf'])
            writer.writerows(self.results)

    def plot_results(self) -> None:
        N_values, avg_tf_values = zip(*self.results)
        plt.plot(N_values, avg_tf_values, marker='o')
        plt.xlabel('Number of Bodies (N)')
        plt.ylabel('Average Frame Time (ms)')
        plt.title('Average Frame Time vs. Number of Bodies')
        plt.grid(True)
        plt.savefig(f"{self.output_path.rsplit('.', 1)[0]}.png")
        plt.show()


def main() -> None:
    experiment = SequentialExperiment(output_path='experiments/sequential_experiment_results.csv')
    experiment.run_experiment(N_range=range(1, 30), G=1, dt=0.01, total_time=60)  # Ajusta el rango y los parámetros según sea necesario
    experiment.save_results_to_csv()
    experiment.plot_results()

if __name__ == "__main__":
    main()
