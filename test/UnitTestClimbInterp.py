import unittest
import numpy as np
from scipy import optimize
from ClimbInterpPackage import ClimbInterp

class TestClimbInterp(unittest.TestCase):
    def setUp(self):
        """Set the variables that are going to be used in the test"""
        self.x_simple = [1, 2, 3]
        self.y_simple = [1, 4, 9]
        self.x_single = [5]
        self.y_single = [10]

    def test_initialization(self):
        """Verify if the class is being initialized correctly"""
        interp = ClimbInterp(self.x_simple, self.y_simple)
        self.assertEqual(interp.x_value, self.x_simple)
        self.assertEqual(interp.y_value, self.y_simple)
        self.assertFalse(interp.show_graph)
        self.assertIsNone(interp.graph_title)
    
    def test_sort_points_by_x(self):
        """Test for static method _sort_points_by_x"""
        x = [3, 1, 2]
        y = [9, 1, 4]
        sorted_x, sorted_y = ClimbInterp._sort_points_by_x(x, y)
        self.assertEqual(sorted_x, (1, 2, 3))
        self.assertEqual(sorted_y, (1, 4, 9))
    
    def test_arrange_points(self):
        """Test for static method arrange_points"""
        interp = ClimbInterp(self.x_negative, self.y_negative)
        # Verify if it removed the x-values with y-values lowers than the previous point 
        self.assertEqual(interp.x_value, [1, 2, 3])
        self.assertEqual(interp.y_value, [1, 4, 9])
        
        # Test with repeated x-values
        interp = ClimbInterp([1, 2, 2, 3], [1, 3, 4, 5])
        self.assertEqual(interp.x_value, [1, 2, 3])
        self.assertEqual(interp.y_value, [1, 4, 5])
    
    def test_exp_function(self):
        """Test for the static method _exp"""
        result = ClimbInterp._exp(2, 1, 1)
        self.assertAlmostEqual(result, np.exp(2), places=5)
    
