import numpy as np
from body import Body
from physics import Physics
import time
from functools import wraps

class SequentialHandler:
    def __init__(self, N: int, G: float = 6.6743e-11):
        """
        Initializes a new SequentialHandler instance which manages N bodies in a simulation.
        
        Parameters:
        - N (int): Number of bodies to simulate.
        - G (float): Gravitational constant, passed to the Physics instance.
        """
        self.bodies = [Body(position=np.random.rand(2) * 100, 
                            velocity=np.random.rand(2) * 50,
                            mass=np.random.rand() * 10 + 50) 
                       for _ in range(N)]
        self.physics = Physics(G)
        self._total_time = 0.0
        self._frame_count = 0

    def update_simulation(self, dt: float, measure_time: bool = False):
        """
        Updates the simulation by one time step, calculating interactions between all bodies.
        Optionally measures and prints the execution time of the update in milliseconds.
        
        Parameters:
        - dt (float): The time step for the simulation update.
        - measure_time (bool): If True, measures and prints the execution time.
        """
        start_time = time.perf_counter() if measure_time else None

        for i, body in enumerate(self.bodies):
            total_acceleration = np.zeros(2)
            for j in range(i+1, len(self.bodies)):
                if i != j:
                    other_body = self.bodies[j]
                    distance_vector = other_body.position - body.position
                    acceleration = self.physics.calculate_acceleration(other_body.mass, distance_vector)
                    total_acceleration += acceleration
            velocity, position = self.physics.update_velocity_position(body.velocity, body.position, total_acceleration, dt)
            body.velocity = velocity
            body.position = position

        if measure_time:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            self._total_time += duration
            self._frame_count += 1

    @property
    def avg_fps_time(self):
        """
        Returns the average time per frame (update) in milliseconds. 
        This value is only meaningful after the simulation has been updated at least once with measure_time=True.
        """
        if self._frame_count > 0:
            return self._total_time / self._frame_count
        else:
            return 0.0
                

