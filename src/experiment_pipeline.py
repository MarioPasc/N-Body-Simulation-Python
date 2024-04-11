import csv
import matplotlib.pyplot as plt
from simulate import SimulationRunner
from sequential_handler import SequentialHandler
from parallel_handler import ParallelHandler

class Experiment:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.results = []

    def run_experiment(self, N_range: range, G: float, dt: float, total_time: int, parallel: bool = False) -> None:
        for N in N_range:
            if parallel:
                handler = ParallelHandler(N=N, G=G, softening=0.1)
            else:
                handler = SequentialHandler(N=N, G=G, softening=0.1)
            simulation_runner = SimulationRunner(dt=dt, total_time=total_time, simulationHandler=handler)
            simulation_runner.run(measure_time=True)
            # Obtener el tiempo promedio de refresco de frames y guardar en la lista de resultados
            avg_fps_time = simulation_runner.handler.avg_fps_time
            self.results.append((N, avg_fps_time))
            print(f"Completed: N={N}, avg_fps_time={avg_fps_time} ms, parallel={parallel}")

    def save_results_to_csv(self) -> None:
        with open(self.output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['N', 'avg_tf_seq', 'avg_tf_numba'])
            writer.writerows(self.results)

    def plot_results(self) -> None:
        N_values, avg_tf_seq_values, avg_tf_numba_values = zip(*self.results)
        plt.plot(N_values, avg_tf_seq_values, marker='o', label='Sequential')
        plt.plot(N_values, avg_tf_numba_values, marker='o', label='Parallel (Numba)')
        plt.xlabel('Number of Bodies (N)')
        plt.ylabel('Average Frame Time (ms)')
        plt.title('Average Frame Time vs. Number of Bodies')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.output_path.rsplit('.', 1)[0]}.png")
        plt.show()


def main() -> None:
    experiment = Experiment(output_path='experiments/experiment_results.csv')
    experiment.run_experiment(N_range=range(10, 100, 10), G=1, dt=0.01, total_time=1, parallel=True)
    experiment.run_experiment(N_range=range(10, 100, 10), G=1, dt=0.01, total_time=1, parallel=False)
    experiment.save_results_to_csv()
    experiment.plot_results()

if __name__ == "__main__":
    main()
