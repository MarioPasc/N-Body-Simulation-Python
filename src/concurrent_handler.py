import numpy as np
from body import Body
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock


class ConcurrentProcessHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, softening: float = 0.1, bodies: Optional[List[Body]] = None):
        if bodies is None:
            self.bodies = [Body(position=np.random.rand(2) * 50, 
                                velocity=np.random.rand(2) * 100,
                                mass=np.random.rand() * 500) 
                           for _ in range(N)]
        else:
            self.bodies = bodies
        self.G = G
        self.softening = softening**3
        self._total_time = 0.0
        self._frame_count = 0

    def calculate_acceleration(self, body: Body, other_bodies: List[Body]):
        for i, body in enumerate(self.bodies):
            acceleration = np.zeros(2)
            for j, other_body in enumerate(self.bodies):
                if i != j:
                    distance_vector = other_body.position - body.position
                    distance = np.linalg.norm(distance_vector + self.softening)
                    acceleration += self.G * other_body.mass / (distance**3) * distance_vector
        return acceleration

    def update_body(self, body: Body, dt: float, other_bodies: List[Body]):
        acceleration = self.calculate_acceleration(body, other_bodies)
        body.update_velocity(body.velocity + acceleration * dt)
        body.update_position(body.position + body.velocity * dt)
        return body

    def update_simulation(self, dt: float, measure_time: bool = False):
        start_time = time.perf_counter() if measure_time else None

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.update_body, body, dt, self.bodies) for body in self.bodies]
            self.bodies = [future.result() for future in as_completed(futures)]

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
        with self.lock:
            body.update_acceleration(acceleration)
            body.update_velocity(dt)
            body.update_position(dt)

    def update_simulation(self, dt: float, measure_time: bool = False):
        start_time = time.perf_counter() if measure_time else None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.update_body, i, dt) for i in range(len(self.bodies))]
            as_completed(futures)  # Wait for all threads to complete

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
