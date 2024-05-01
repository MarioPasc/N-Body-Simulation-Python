import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os 
# Leer los datos del fichero CSV
data = np.genfromtxt('./experiments/experiment_results.csv', delimiter=',', skip_header=1)
N = data[:, 0]  # Columna "N"
sequential = data[:, 1]  # Columna "Sequential"
pycuda = data[:, 2]  # Columna "PyCUDA"
threadpool = data[:, 3]  # Columna "ThreadPool"

# Definir los grados de los polinomios a probar
degrees = [1, 2]

# Funciones para ajustar y graficar los datos
def fit_and_plot(x, y, label):
    best_degree = None
    best_r2 = -np.inf
    threshold = 0.00019782 # Umbral para los coeficientes
    
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(x_poly, y)
        
        y_pred = model.predict(x_poly)
        r2 = r2_score(y, y_pred)
        
        if r2 > best_r2:
            best_degree = degree
            best_r2 = r2
            best_model = model
    
    # Encontrar el grado del polinomio según el umbral de los coeficientes
    coef_degree = 0
    for i in range(1, best_degree + 1):
        if abs(best_model.coef_[i]) >= threshold:
            coef_degree = i
    
    x_plot = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
    x_plot_poly = poly_features.transform(x_plot)
    y_plot = best_model.predict(x_plot_poly)
    
    plt.scatter(x, y, label=f'{label} (data)')
    plt.plot(x_plot, y_plot, label=f'{label} (degree {coef_degree}, R^2 = {best_r2:.3f})')
    
    print(f"Mejor grado para {label}: {coef_degree}")
    print(f"Coeficientes para {label}: {best_model.coef_[:coef_degree+1]}")
    print(f"Intercepto para {label}: {best_model.intercept_}")
    
    # Imprimir la fórmula del polinomio
    formula = f"{label}(N) = {best_model.intercept_:.3f}"
    for i in range(1, coef_degree + 1):
        coef = best_model.coef_[i]
        if abs(coef) >= threshold:
            if coef >= 0:
                formula += f" + {coef:.3f} * N^{i}"
            else:
                formula += f" - {abs(coef):.3f} * N^{i}"
    print(f"Fórmula para {label}: {formula}")
    print()

# Ajustar y graficar los datos para Sequential, PyCUDA y ThreadPool
plt.figure(figsize=(10, 6))

fit_and_plot(N, sequential, 'Sequential')
fit_and_plot(N, pycuda, 'PyCUDA')
fit_and_plot(N, threadpool, 'ThreadPool')

plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig(os.path.join("/home/mariopasc/Python/Projects/NBodySimulation/N-Body-Simulation-Python/experiments","ComplejidadEmpirica.png"))
plt.show()
