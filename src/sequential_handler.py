import numpy as np
from body import Body
from physics import Physics
import time
from functools import wraps
from typing import List, Optional
class SequentialHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, bodies: Optional[List[Body]] = None):
        """
        Initializes a new SequentialHandler instance which manages N bodies in a simulation.
        
        Parameters:
        - N (int): Number of bodies to simulate.
        - G (float): Gravitational constant, passed to the Physics instance.
        """
        if bodies is None:
            self.bodies = [Body(position=np.random.rand(2) * 4, 
                                velocity=np.random.rand(2),
                                radius = 0.1,
                                mass=np.random.rand()) 
                        for _ in range(N)]
        else:
            self.bodies = bodies    
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

        # Calculate new accelerations for each body
        for i, body in enumerate(self.bodies):
            acceleration = np.zeros(2)
            for j in range(i+1, len(self.bodies)):
                other_body = self.bodies[j]
                if i != j:
                    distance_vector = other_body.position - body.position
                    # If the objects are too close, compute the collision method, if not, compute acceleration
                    if np.linalg.norm(distance_vector) <= (body.radius + other_body.radius):
                        self.physics.resolve_collision(body1=body, body2=other_body)
                    else:
                        acceleration = self.physics.calculate_acceleration(other_body.mass, distance_vector)
            body.update_acceleration(acceleration)

        # Update velocities and positions for each body
        for body in self.bodies:
            new_velocity, new_position = self.physics.update_velocity_position(body.velocity, body.position, body.acceleration, dt)
            body.update_velocity(new_velocity)
            body.update_position(new_position)
        

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
                


