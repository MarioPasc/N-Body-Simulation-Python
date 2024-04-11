import numpy as np
from numba import cuda, float32
from body import Body
import time
from typing import List, Optional, Tuple

@cuda.jit
def calculate_accelerations(positions, masses, accelerations, G, softening, eps):
    i = cuda.grid(1)
    if i < positions.shape[0]:
        acceleration = np.zeros(2, dtype=float32)
        for j in range(positions.shape[0]):
            if i != j:
                distance = positions[j] - positions[i]
                mod_distance = np.linalg.norm(distance)
                if mod_distance > eps:
                    acceleration += G * masses[j] / (mod_distance**3 + softening) * distance
        accelerations[i] = acceleration

@cuda.jit
def update_velocities_positions(velocities, positions, accelerations, dt):
    i = cuda.grid(1)
    if i < positions.shape[0]:
        velocities[i] += accelerations[i] * dt
        positions[i] += velocities[i] * dt

class ParallelHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, softening:float = 0.01, bodies: Optional[List[Body]] = None):
        if bodies is None:
            self.bodies = [Body(position=(np.random.rand(2) * 50), 
                                velocity=(np.random.rand(2) * 100),
                                mass=np.random.rand() * 500) 
                        for _ in range(N)]
        else:
            self.bodies = bodies    
        self.G = G
        self.softening = softening**3
        self.eps = 0.5
        self._total_time = 0.0
        self._frame_count = 0

    def update_simulation(self, dt: float, measure_time: bool = False):
        start_time = time.perf_counter() if measure_time else None

        positions = np.array([body.position for body in self.bodies], dtype=np.float32)
        velocities = np.array([body.velocity for body in self.bodies], dtype=np.float32)
        masses = np.array([body.mass for body in self.bodies], dtype=np.float32)
        accelerations = np.zeros_like(positions)

        threadsperblock = 256
        blockspergrid = (positions.shape[0] + (threadsperblock - 1)) // threadsperblock

        calculate_accelerations[blockspergrid, threadsperblock](positions, masses, accelerations, self.G, self.softening, self.eps)
        update_velocities_positions[blockspergrid, threadsperblock](velocities, positions, accelerations, dt)

        for i, body in enumerate(self.bodies):
            body.update_acceleration(accelerations[i])
            body.update_velocity(velocities[i])
            body.update_position(positions[i])

        if measure_time:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            self._total_time += duration
            self._frame_count += 1

    @property
    def avg_fps_time(self):
        if self._frame_count > 0:
            return self._total_time / self._frame_count
        else:
            return 0.0
