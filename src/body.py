import numpy as np
from typing import List

class Body:
    """
    Represents a celestial body in a 2D N-body simulation.
    
    Attributes:
    - position (np.ndarray): The body's position in 3D space.
    - velocity (np.ndarray): The body's velocity in 3D space.
    - mass (float): The mass of the body.
    - acceleration (np.ndarray): The body's acceleration in 3D space.
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray, mass: float):
        """
        Initializes a new instance of the Body class.
        
        Parameters:
        - position (np.ndarray): The initial position of the body.
        - velocity (np.ndarray): The initial velocity of the body.
        - mass (float): The mass of the body.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.acceleration = np.zeros(2, dtype=float)

    def update_position(self, new_position: np.ndarray):
        """
        Updates the body's position.
        
        Parameters:
        - new_position (np.ndarray): The new position of the body.
        """
        self.position = np.array(new_position, dtype=float)

    def update_velocity(self, new_velocity: np.ndarray):
        """
        Updates the body's velocity.
        
        Parameters:
        - new_velocity (np.ndarray): The new velocity of the body.
        """
        self.velocity = np.array(new_velocity, dtype=float)

    def update_acceleration(self, new_acceleration: np.ndarray):
        """
        Updates the body's acceleration.
        
        Parameters:
        - new_acceleration (np.ndarray): The new acceleration of the body.
        """
        self.acceleration = np.array(new_acceleration, dtype=float)
