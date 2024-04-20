import csv
import matplotlib.pyplot as plt
from simulate import SimulationRunner
from sequential_handler import SequentialHandler
from parallel_handler import ParallelHandler
from concurrent_handler import ConcurrentProcessHandler, ConcurrentThreadHandler
from body import Body
import numpy as np

class Experiment:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.seq_results = []
        self.cuda_results = []
        self.process_results = []
        self.thread_results = []

    def run_experiment(self, N_range: range, G: float, dt: float, total_time: int, handler) -> None:
        results = []
        for N in N_range:
            handler.bodies = [Body(position=np.random.rand(2) * 50, 
                                   velocity=np.random.rand(2) * 100,
                                   mass=np.random.rand() * 500) 
                              for _ in range(N)]
            simulation_runner = SimulationRunner(dt=dt, total_time=total_time, simulationHandler=handler)
            simulation_runner.run(measure_time=True)
            avg_fps_time = simulation_runner.handler.avg_fps_time
            results.append((N, avg_fps_time))
            print(f"Completed: N={N}, avg_fps_time={avg_fps_time} ms, handler={type(handler).__name__}")

        if isinstance(handler, SequentialHandler):
            self.seq_results = results
        elif isinstance(handler, ParallelHandler):
            self.cuda_results = results
        #elif isinstance(handler, ConcurrentProcessHandler):
        #    self.process_results = results
        elif isinstance(handler, ConcurrentThreadHandler):
            self.thread_results = results

    def save_results_to_csv(self) -> None:
        with open(self.output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            #writer.writerow(['N', 'Sequential', 'PyCUDA', 'ProcessPool', 'ThreadPool'])
            writer.writerow(['N', 'Sequential', 'PyCUDA', 'ThreadPool'])
            for i in range(len(self.seq_results)):
                #row = [self.seq_results[i][0], self.seq_results[i][1], self.cuda_results[i][1], self.process_results[i][1], self.thread_results[i][1]]
                row = [self.seq_results[i][0], 
                       self.seq_results[i][1], 
                       self.cuda_results[i][1], 
                       self.thread_results[i][1]]
                writer.writerow(row)

    def plot_results(self) -> None:
        N_values, seq_values = zip(*self.seq_results)
        _, cuda_values = zip(*self.cuda_results)
        #_, process_values = zip(*self.process_results)
        _, thread_values = zip(*self.thread_results)

        plt.plot(N_values, seq_values, marker='o', label='Sequential')
        plt.plot(N_values, cuda_values, marker='o', label='PyCUDA')
        #plt.plot(N_values, process_values, marker='o', label='ProcessPool')
        plt.plot(N_values, thread_values, marker='o', label='ThreadPool')
        plt.xlabel('Number of Bodies (N)')
        plt.ylabel('Average Frame Time (ms)')
        plt.title('Average Frame Time vs. Number of Bodies')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.output_path.rsplit('.', 1)[0]}.png")
        plt.show()

def main() -> None:
    path = "/home/mariopasc/Python/Projects/NBodySimulation/N-Body-Simulation-Python/experiments/experiment_results.csv"
    experiment = Experiment(output_path=path)
    N_range = range(1, 1000, 100)
    G = 1
    dt = 0.01
    total_time = 1

    seq_handler = SequentialHandler(N=0, G=G, softening=0.1)
    cuda_handler = ParallelHandler(N=0, G=G, softening=0.1)
    #process_handler = ConcurrentProcessHandler(N=0, G=G, softening=0.1)
    thread_handler = ConcurrentThreadHandler(N=0, G=G, softening=0.1)

    experiment.run_experiment(N_range, G, dt, total_time, seq_handler)
    experiment.run_experiment(N_range, G, dt, total_time, cuda_handler)
    #experiment.run_experiment(N_range, G, dt, total_time, process_handler)
    experiment.run_experiment(N_range, G, dt, total_time, thread_handler)
    experiment.save_results_to_csv()
    experiment.plot_results()

if __name__ == "__main__":
    main()