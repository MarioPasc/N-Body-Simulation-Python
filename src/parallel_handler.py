import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from body import Body
import time
from typing import List, Optional

class ParallelHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, softening: float = 0.01, bodies: Optional[List[Body]] = None):
        if bodies is None:
            self.bodies = [Body(position=(np.random.rand(2) * 50).astype(np.float32),
                                velocity=(np.random.rand(2) * 100).astype(np.float32),
                                mass=np.random.rand() * 500)
                           for _ in range(N)]
        else:
            self.bodies = bodies
        self.G = G
        self.softening = softening**3
        self.eps = 0.5
        self._total_time = 0.0
        self._frame_count = 0

        self.mod = SourceModule("""
            __global__ void calculate_accelerations(float *positions, float *masses, float *accelerations, float G, float softening, float eps, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    float ax = 0.0f;
                    float ay = 0.0f;
                    for (int j = 0; j < N; j++) {
                        if (i != j) {
                            float dx = positions[j * 2] - positions[i * 2];
                            float dy = positions[j * 2 + 1] - positions[i * 2 + 1];
                            float dist_sq = dx * dx + dy * dy;
                            float inv_dist = rsqrtf(dist_sq + softening);
                            float inv_dist_cube = inv_dist * inv_dist * inv_dist;
                            ax += G * masses[j] * dx * inv_dist_cube;
                            ay += G * masses[j] * dy * inv_dist_cube;
                        }
                    }
                    accelerations[i * 2] = ax;
                    accelerations[i * 2 + 1] = ay;
                }
            }

            __global__ void update_velocities_positions(float *velocities, float *positions, float *accelerations, float dt, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    velocities[i * 2] += accelerations[i * 2] * dt;
                    velocities[i * 2 + 1] += accelerations[i * 2 + 1] * dt;
                    positions[i * 2] += velocities[i * 2] * dt;
                    positions[i * 2 + 1] += velocities[i * 2 + 1] * dt;
                }
            }
        """)

        self.calculate_accelerations = self.mod.get_function("calculate_accelerations")
        self.update_velocities_positions = self.mod.get_function("update_velocities_positions")

    def update_simulation(self, dt: float, measure_time: bool = False):
        start_time = time.perf_counter() if measure_time else None

        positions = np.array([body.position for body in self.bodies], dtype=np.float32).flatten()
        velocities = np.array([body.velocity for body in self.bodies], dtype=np.float32).flatten()
        masses = np.array([body.mass for body in self.bodies], dtype=np.float32)
        accelerations = np.zeros_like(positions)

        N = len(self.bodies)
        block_size = 256
        grid_size = (N + block_size - 1) // block_size

        positions_gpu = cuda.mem_alloc(positions.nbytes)
        velocities_gpu = cuda.mem_alloc(velocities.nbytes)
        masses_gpu = cuda.mem_alloc(masses.nbytes)
        accelerations_gpu = cuda.mem_alloc(accelerations.nbytes)

        cuda.memcpy_htod(positions_gpu, positions)
        cuda.memcpy_htod(velocities_gpu, velocities)
        cuda.memcpy_htod(masses_gpu, masses)

        self.calculate_accelerations(positions_gpu, masses_gpu, accelerations_gpu, np.float32(self.G), np.float32(self.softening), np.float32(self.eps), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))
        self.update_velocities_positions(velocities_gpu, positions_gpu, accelerations_gpu, np.float32(dt), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

        cuda.memcpy_dtoh(positions, positions_gpu)
        cuda.memcpy_dtoh(velocities, velocities_gpu)
        cuda.memcpy_dtoh(accelerations, accelerations_gpu)

        for i, body in enumerate(self.bodies):
            body.update_acceleration(accelerations[i * 2:i * 2 + 2])
            body.update_velocity(velocities[i * 2:i * 2 + 2])
            body.update_position(positions[i * 2:i * 2 + 2])

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