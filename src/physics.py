import numpy as np
from typing import List, Tuple

class InvalidTimeStepError(ValueError):
    """Exception raised when an invalid timestep is provided to the simulation."""

class Physics:
    """
    Handles the calculation of physical quantities for the N-body simulation
    using the Euler method for numerical approximation.
    
    Attributes:
    - G (float): Gravitational constant, modifiable at the object instantiation.
    """

    def __init__(self, G: float = 6.6743e-11):
        """
        Initializes a new Physics instance with a given gravitational constant.

        Parameters:
        - G (float): The gravitational constant to be used in the simulation.

        Attributes:
        - eps (float): A small epsilon value to prevent division by zero and ensure numerical stability.
        """
        self.G = G
        self.eps = np.finfo(np.float32).eps

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
            return self.G * mass2 / (mod_distance**3) * distance
        else:
            return np.zeros_like(distance)

    def resolve_collision(self, body1, body2):
        body1.velocity, body2.velocity = body2.velocity, body1.velocity


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
        if dt <= self.eps:
            raise InvalidTimeStepError(f"dt must be greater than {self.eps}")
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        return new_velocity, new_position