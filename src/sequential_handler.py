import numpy as np
from body import Body
import time
from typing import List, Optional, Tuple

class SequentialHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, epsilon: float = 0.5, softening: float = 0.1, 
                 bodies: Optional[List[Body]] = None):
        """
        Initializes a new SequentialHandler instance which manages N bodies in a simulation.
        
        Parameters:
        - N (int): Number of bodies to simulate.
        - G (float): Gravitational constant, passed to the Physics instance.
        """
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

    def update_simulation(self, dt: float, measure_time: bool = False):
        """
        Updates the simulation by one time step, calculating interactions between all bodies.
        Optionally measures and prints the execution time of the update in milliseconds.
        
        Parameters:
        - dt (float): The time step for the simulation update.
        - measure_time (bool): If True, measures and prints the execution time.
        """
        start_time = time.perf_counter() if measure_time else None

        # Calculate new accelerations for each body
        for i, body in enumerate(self.bodies):
            new_acceleration = np.zeros(2)
            for j, other_body in enumerate(self.bodies):
                if i != j:
                    distance_vector = other_body.position - body.position
                    mod_distance = np.linalg.norm(distance_vector) + self.softening
                    acceleration = self.G * other_body.mass / (mod_distance**3) * distance_vector
                    new_acceleration += acceleration
            body.update_acceleration(new_acceleration)

        # Update velocities and positions for each body
        for body in self.bodies:
            body.update_velocity(body.velocity + body.acceleration * dt)
            body.update_position(body.position + body.velocity * dt)
        

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
                


