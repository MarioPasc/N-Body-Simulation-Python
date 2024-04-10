import numpy as np
from body import Body
import time
from typing import List, Optional, Tuple


class Physics:
    def __init__(self, G: float = 6.6743e-11):
        """
        Initializes a new Physics instance with a given gravitational constant.

        Parameters:
        - G (float): The gravitational constant to be used in the simulation.

        Attributes:
        - eps (float): A small epsilon value to prevent division by zero and ensure numerical stability.
        """
        self.G = G
        self.softening = 0.01**3
        self.eps = .5

    def calculate_acceleration(self, mass2: float, distance: np.ndarray) -> np.ndarray:
        """
        Calculates the acceleration on a body due to the gravitational effect of another body.
        
        Parameters:
        - mass2 (float): The mass of the body exerting the gravitational force.
        - distance_vector (np.ndarray): The vector distance between the two bodies.
        
        Returns:
        - np.ndarray: The calculated acceleration vector.
        """
        mod_distance = np.linalg.norm(distance)
        if mod_distance > self.eps:
            return self.G * mass2 / (mod_distance**3 + self.softening) * distance
        else:
            return np.zeros_like(distance)
        
    def resolve_collisions(self, body1: Body, body2: Body, distance: np.ndarray) -> None:
        if np.linalg.norm(distance) == 0:
            body1.velocity, body2.velocity = body2.velocity, body1.velocity
        pass    
        
    def update_velocity_position(self, velocity: np.ndarray, position: np.ndarray, acceleration: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the velocity and position of a body using the Euler method.
        
        Parameters:
        - velocity (np.ndarray): Initial velocity of the body.
        - position (np.ndarray): Initial position of the body.
        - acceleration (np.ndarray): Acceleration of the body.
        - dt (float): Time step.
        
        Returns:
        - Tuple[np.ndarray, np.ndarray]: Updated velocity and position.
        """
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        return new_velocity, new_position

class SequentialHandler:
    def __init__(self, N: int, G: float = 6.6743e-11, bodies: Optional[List[Body]] = None):
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
            new_acceleration = np.zeros(2)
            for j, other_body in enumerate(self.bodies):
                if i != j:
                    distance_vector = other_body.position - body.position
                    acceleration = self.physics.calculate_acceleration(other_body.mass, distance_vector)
                    new_acceleration += acceleration
                    self.physics.resolve_collisions(body1=body, body2=other_body, distance=distance_vector)
            body.update_acceleration(new_acceleration)

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
                


