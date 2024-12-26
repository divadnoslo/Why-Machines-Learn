import numpy as np
import unittest
from abc import ABC, abstractmethod

class LogicalPerceptron(ABC):
    """Base Class for all Logical Perceptrons"""

    def __init__(self, size = 2):
        self.size = size

    @abstractmethod
    def fire(self, input_vector):
        pass

    def validate_input_vector(self, input_vector: np.ndarray[float], size: int) -> None: 
        """ Ensures that the input is a one-dimensional numpy array of floats. Raises a ValueError if the input is not valid. """ 
    
        # Check if input is one-dimensional 
        if input_vector.ndim != 1: 
            raise ValueError("Input must be a one-dimensional vector!") 
    
        # Check if input is same size as specified in the perceptron
        if not input_vector.size == size:
            raise ValueError("Input vector must match the size of the perceptron!")
    
        # Check if input contains only values 0.0 and 1.0 
        if not np.all(np.isin(input_vector, [0.0, 1.0])):
            raise ValueError("Input vector must only contain float values 0.0 and 1.0!") 
    
class AndMcp(LogicalPerceptron):
    """AND Operation as a McColloch-Pitts Perceptron"""

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector, self.size)
        theta = np.sum(input_vector)
        return float(theta == float(self.size))

class OrMcp(LogicalPerceptron):
    """OR Operation as a McColloch-Pitts Perceptron"""

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector, self.size)
        theta = np.sum(input_vector)
        return float(theta >= 1.0)

class NandMcp(LogicalPerceptron):
    """NAND Operation as a McColloch-Pitts Perceptron"""

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector, self.size)
        theta = np.sum(input_vector)
        return float(theta != float(self.size))

class NorMcp(LogicalPerceptron):
    """NOR Operation as a McColloch-Pitts Perceptron"""

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector, self.size)
        theta = np.sum(input_vector)
        return float(theta == 0.0)
    
class XorMcp(LogicalPerceptron):
    """XOR Operation as a McColloch-Pitts Perceptron"""

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.validate_input_vector(input_vector, self.size)
        theta = np.sum(input_vector)
        return float(theta == 1.0)

class TestAndMcp(unittest.TestCase):

    def setUp(self):
        self.mcp = AndMcp()

    def test_fire_with_two_inputs(self):
        self.mcp.size = 2
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0])), 1.0)

    def test_fire_with_three_inputs(self):
        self.mcp.size = 3
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 1.0])), 1.0)

class TestOrMcp(unittest.TestCase):

    def setUp(self):
        self.mcp = OrMcp()

    def test_fire_with_two_inputs(self):
        self.mcp.size = 2
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0])), 1.0)

    def test_fire_with_three_inputs(self):
        self.mcp.size = 3
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 1.0])), 1.0)

class TestNandMcp(unittest.TestCase):

    def setUp(self):
        self.mcp = NandMcp()

    def test_fire_with_two_inputs(self):
        self.mcp.size = 2
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0])), 0.0)

    def test_fire_with_three_inputs(self):
        self.mcp.size = 3
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 1.0])), 0.0)

class TestNorMcp(unittest.TestCase):

    def setUp(self):
        self.mcp = NorMcp()

    def test_fire_with_two_inputs(self):
        self.mcp.size = 2
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0])), 0.0)

    def test_fire_with_three_inputs(self):
        self.mcp.size = 3
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 1.0])), 0.0)

class TestXorMcp(unittest.TestCase):

    def setUp(self):
        self.mcp = XorMcp()

    def test_fire_with_two_inputs(self):
        self.mcp.size = 2
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0])), 0.0)

    def test_fire_with_three_inputs(self):
        self.mcp.size = 3
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 0.0, 1.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([0.0, 1.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 0.0])), 1.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 0.0, 1.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 0.0])), 0.0)
        self.assertEqual(self.mcp.fire(np.array([1.0, 1.0, 1.0])), 0.0)

if __name__ == '__main__':
    unittest.main()