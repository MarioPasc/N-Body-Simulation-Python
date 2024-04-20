import numpy as np
import matplotlib.pyplot as plt
import os 
def calculate_force(distance, mass1, mass2, G, softening):
    r = distance + softening
    force = G * mass1 * mass2 / r**3
    return force

# Parámetros de la simulación
mass1 = 1.0
mass2 = 1.0
G = 1.0

# Valores de softening a evaluar
softening_values = [0.0, 0.3, 0.5, 1]

# Rango de distancias a evaluar
distances = np.linspace(0.0001, 10, 1000)

# Calcular y graficar las fuerzas para cada valor de softening
for softening in softening_values:
    forces = [calculate_force(distance, mass1, mass2, G, softening) for distance in distances]
    plt.plot(distances, forces, label=f"Softening = {softening}")

plt.xlabel("Distancia ($r$)")
plt.ylabel("Fuerza ($F$)")
plt.title("Fuerza vs Distancia para diferentes valores de Softening")
plt.xlim([0,2])
plt.ylim([0,50])
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("/home/mariopasc/Python/Projects/NBodySimulation/N-Body-Simulation-Python/experiments","Softening.png"))
plt.show()