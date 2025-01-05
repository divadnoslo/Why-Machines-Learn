import numpy as np
import unittest

class Perceptron:
    """Perceptron"""

    def __init__(self, number_of_inputs: int=2, learning_rate: float=0.1):
        if number_of_inputs < 1:
            raise ValueError("Number of inputs must be greater than 1!")
        self.__number_of_inputs = number_of_inputs
        self.__weights = np.zeros(number_of_inputs + 1)
        self.learning_rate = learning_rate

    def __validate_input_vector(self, input_vector: np.ndarray[float]):
        if input_vector.ndim != 1:
            raise ValueError("The input must be a vector of only one dimension!")
        if input_vector.size != self.__number_of_inputs:
            raise ValueError(f"The perceptron has {self.__number_of_inputs} inputs, the input vector must be of this length!")
    
    def __classify(self, theta: float) -> float:
        if theta > 0.0:
            return 1.0
        else:
            return -1.0
        
    def set_weights(self, weights: np.ndarray[float]):
        if weights.ndim != 1:
            raise ValueError("Weights must be a one-dimensional vector!")
        if weights.size != self.__number_of_inputs + 1:
            raise ValueError("Weights must be one element larger than the number of inputs where the first element is the bias!")
        self.__weights = weights

    def get_weights(self) -> np.ndarray[float]:
        return self.__weights
        
    def set_bias(self, bias: float):
        self.__weights[0] = bias

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.__validate_input_vector(input_vector)
        return self.__classify(np.dot(self.__weights, np.insert(input_vector, 0, 1.0)))
    
    def train(self, input_vector: np.ndarray[float], classification: float):
        self.__validate_input_vector(input_vector)
        if classification * np.dot(self.__weights, np.insert(input_vector, 0, 1.0)) <= 0.0:
            self.__weights += self.learning_rate * classification * np.insert(input_vector, 0, 1.0)


class TestPerceptron(unittest.TestCase):

    def setUp(self):
        pass

    def test_01_initialize(self):
        a = Perceptron(2)
        b = Perceptron(3)
        c = Perceptron(4)

    def test_02_set_get_weights(self):
        p = Perceptron(2)
        weights = np.array([1.0, 3.0, 4.5])
        p.set_weights(weights)
        new_weights = p.get_weights()
        self.assertEqual(weights[0], new_weights[0])
        self.assertEqual(weights[1], new_weights[1])
        self.assertEqual(weights[2], new_weights[2])

    def test_03_set_bias(self):
        p = Perceptron(2)
        p.set_bias(6.9)
        weights = p.get_weights()
        self.assertEqual(weights[0], 6.9)

    def test_04_fire_01(self):
        p = Perceptron()
        y = p.fire(np.array([3.0, 4.0]))
        self.assertEqual(y, -1.0)

    def test_05_fire_02(self):
        p = Perceptron()
        p.set_weights(np.array([2.0, 3.0, 4.0]))
        y = p.fire(np.array([1.0, 1.0]))
        self.assertEqual(y, 1.0)

    def test_05_fire_03(self):
        p = Perceptron()
        p.set_bias(7.0)
        y = p.fire(np.array([1.0, 1.0]))
        self.assertEqual(y, 1.0)

    def test_06_train(self):
        p = Perceptron()
        p.train(np.array([2.0, 3.0]), 1.0)
    

if __name__ == '__main__':
    unittest.main()