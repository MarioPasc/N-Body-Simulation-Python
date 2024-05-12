import numpy as np
from body import Body
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class ConcurrentThreadHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, softening: float = 0.1, bodies: Optional[List[Body]] = None):
        if bodies is None:
            self.bodies = [Body(position=np.random.rand(2) * 50, 
                                velocity=np.random.rand(2) * 100,
                                mass=np.random.rand() * 500) 
                           for _ in range(N)]
        else:
            self.bodies = bodies
        self.G = G
        self.softening = softening
        self._total_time = 0.0
        self._frame_count = 0
        self.lock = Lock()
        self.positions = []  # To store the positions of all bodies at each timestep

    def calculate_acceleration(self, body: Body, other_bodies: List[Body]):
        acceleration = np.zeros(2)
        for other_body in other_bodies:
            if body != other_body:
                distance_vector = other_body.position - body.position
                distance = np.linalg.norm(distance_vector)
                acceleration += self.G * other_body.mass / (distance**3 + self.softening) * distance_vector
        return acceleration

    def update_body(self, body_index: int, dt: float):
        body = self.bodies[body_index]
        acceleration = self.calculate_acceleration(body, self.bodies)
        body.update_acceleration(acceleration)
        body.update_velocity(body.velocity + body.acceleration * dt)
        body.update_position(body.position + body.velocity * dt)

    def update_simulation(self, dt: float, measure_time: bool = False):
        start_time = time.perf_counter() if measure_time else None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.update_body, i, dt) for i in range(len(self.bodies))]
            as_completed(futures)  # Wait for all threads to complete

        # Store the current positions of all bodies
        with self.lock:
            positions = [body.position for body in self.bodies]
            self.positions.append(positions)

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
