"""Test the solver functionality in OracleStrategy."""
import unittest
import pulp
from src.solvers import solve_optimization_problem

class TestSolver(unittest.TestCase):
    def test_simple_optimization(self):
        """Test that the solver can solve a simple optimization problem."""
        try:
            # Create a simple optimization problem
            prob = pulp.LpProblem("TestOptimization", pulp.LpMinimize)
            
            # Add a simple objective
            x = pulp.LpVariable("x", lowBound=0)
            y = pulp.LpVariable("y", lowBound=0)
            prob += x + y
            
            # Add a constraint
            prob += x + 2*y >= 10

            status, objective_value = solve_optimization_problem(prob)
            
            # Check that the solver worked
            self.assertIsNotNone(status)
            self.assertEqual(pulp.LpStatus[status], "Optimal")
            self.assertIsNotNone(objective_value)
            self.assertAlmostEqual(objective_value, 5.0)
            self.assertAlmostEqual(x.value(), 0.0)
            self.assertAlmostEqual(y.value(), 5.0)
        except Exception as e:
            self.fail(f"Solver failed with exception: {str(e)}")
        
