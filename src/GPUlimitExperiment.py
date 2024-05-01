import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from parallel_handler import ParallelHandler
from tqdm import tqdm

class GPULimitExperiment:
    def __init__(self, num_bodies_list, simulation_time):
        self.num_bodies_list = num_bodies_list
        self.simulation_time = simulation_time
        self.refresh_times = []

    def run(self):
        for num_bodies in tqdm(self.num_bodies_list, colour="red", desc="Running experiment"):
            handler = ParallelHandler(num_bodies)
            handler.update_simulation(0.01)  # Warm-up run

            total_time = 0.0
            num_iterations = 0
            while total_time < self.simulation_time:
                handler.update_simulation(0.01, measure_time=True)
                total_time += handler.avg_fps_time / 1000  # Convertir de milisegundos a segundos
                num_iterations += 1

            avg_refresh_time = (total_time * 1000) / num_iterations  # Convertir de segundos a milisegundos
            self.refresh_times.append(avg_refresh_time)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.num_bodies_list, self.refresh_times, label='Datos')

        # Ajustar una recta a los datos
        model = LinearRegression()
        x = np.array(self.num_bodies_list).reshape(-1, 1)
        model.fit(x, self.refresh_times)
        y_pred = model.predict(x)

        plt.plot(self.num_bodies_list, y_pred, color='red', label='Recta ajustada')

        plt.xlabel('Número de cuerpos')
        plt.ylabel('Tiempo de refresco promedio (ms)')
        plt.title('Límite de la GPU vs Número de cuerpos')
        plt.legend()
        plt.tight_layout()
        plt.savefig("./experiments/GPU_limit.png")
        plt.show()

        # Imprimir la fórmula de la recta
        coef = model.coef_[0]
        intercept = model.intercept_
        formula = f"TRP (ms) = {intercept:.3f} + {coef:.3f} * N "
        print(f"Fórmula de la recta: {formula}")

def main() -> None:
    # Ejemplo de uso
    num_bodies_list = [100, 1000, 10000, 100000, 100000]
    simulation_time = 1  # Tiempo de simulación en milisegundos (1 segundo)

    experiment = GPULimitExperiment(num_bodies_list, simulation_time)
    experiment.run()
    experiment.plot_results()

if __name__ == "__main__":
    main()
