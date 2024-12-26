import numpy as np
import unittest
from abc import ABC, abstractmethod

# Base Class Definitions
#-------------------------------------------------------
class BasePerceptron(ABC):

    def __init__(self, weights=np.ones(2), bias=0.0):
        self.weights = weights
        self.bias = bias

    @abstractmethod
    def activate_neuron():
        pass

    @abstractmethod
    def fire():
        pass

    def validate_input_vector(self, input_vector): 
        """ Ensures that the input is a one-dimensional numpy array of floats. Raises a ValueError if the input is not valid. """ 

        # Check if input is a numpy array 
        if not isinstance(input_vector, np.ndarray): 
           raise ValueError("Input must be a numpy array!") 
    
        # Check if input is one-dimensional 
        if input_vector.ndim != 1: 
            raise ValueError("Input must be a one-dimensional vector!") 
    
        # Check if input contains only floats 
        if not np.issubdtype(input_vector.dtype, float): 
            raise ValueError("Input vector must contain only floating point values!")
    
        # Check if input is same size as specified in the perceptron
        if not input_vector.size == self.weights.size:
            raise ValueError("Input vector must match the size of the perceptron!")
    
        # If all checks pass, return the input vector return
        return input_vector

# Class Definitions
#-------------------------------------------------------
class Perceptron(BasePerceptron):
    """Standard Linear Perceptron"""

    def activate_neuron(self, theta: float) -> float:
        return theta

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector)
        theta = np.dot(self.weights, input_vector) + self.bias
        return self.activate_neuron(theta)
    
# Unit Tests
#-------------------------------------------------------
class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.perceptron = Perceptron

    def test_01(self):
        self.perceptron = Perceptron()
        self.assertEqual(self.perceptron.fire(np.array([0.0, 0.0])), 0.0)

    def test_02(self):
        self.perceptron = Perceptron(np.array([2.0, -1.0]), 7.0)
        self.assertEqual(self.perceptron.fire(np.array([3.0, 4.0])), 9.0)

    def test_03(self):
        self.perceptron = Perceptron(np.array([2.0, -1.0, 3.0]), -2.0)
        self.assertEqual(self.perceptron.fire(np.array([3.0, 4.0, 5.0])), 15.0)

# Main
#-------------------------
if __name__ == '__main__':
    unittest.main()