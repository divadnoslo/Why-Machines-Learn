import numpy as np
import unittest

class Perceptron:
    """Perceptron"""

    def __init__(self, number_of_inputs: int=2):
        if number_of_inputs < 1:
            raise ValueError("Number of Inputs must be greater than 1!")
        self.__number_of_inputs = number_of_inputs
        self.__weights = np.zeros(number_of_inputs + 1)

    def __validate_input_vector(self, input_vector: np.ndarray[float]):
        if input_vector.ndim != 1:
            raise ValueError("The input must be a vector of only one dimension!")
        if input_vector.size != self.__number_of_inputs:
            raise ValueError(f"The perceptron has {self.__number_of_inputs} inputs, the input vector must be of this length!")
    
    def __classify(self, theta: float) -> float:
        if theta > 0:
            return 1.0
        else:
            return -1.0
        
    def set_weights(self, weights: np.ndarray[float]):
        #if weights.ndim != 1:
        #    raise ValueError("Weights must be a one-dimensional vector!")
        #elif weights.size != self.__number_of_inputs - 1:
        #    raise ValueError("Weights must be one element larger than the number of inputs where the first element is the bias!")
        #else:
        self.__weights = weights
        
    def set_bias(self, bias: float):
        self.__weights[0] = bias

    def fire(self, input_vector: np.ndarray[float]) -> float:
        self.__validate_input_vector(input_vector)
        self.__classify(np.dot(self.__weights, np.concatinate(1.0, input_vector)))


class TestPerceptron(unittest.TestCase):

    def setUp(self):
        pass

    def test_01_initialize(self):
        a = Perceptron(2)
        b = Perceptron(3)
        c = Perceptron(4)

    def test_02_set_weights(self):
        p = Perceptron(2)
        p.set_weights(np.ndarray([3.0, 4.5]))

    '''def test_02_basic_fire(self):
        perceptron = Perceptron()
        y = perceptron.fire(np.ndarray([1.0, 1.0]))
        self.assertEqual(y, -1.0)'''
    

if __name__ == '__main__':
    unittest.main()