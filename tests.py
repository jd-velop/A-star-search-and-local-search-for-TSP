import unittest
import json
from main import create_grid, assign_move_costs, astar

# helper function to make sure a move is legal
def is_legal_move(prev, curr, m, n):
    row1, col1 = prev
    row2, col2 = curr
    # move is in bounds
    if not (0 <= row2 < m and 0 <= col2 < n):
        return False
    # move is one of the four legal moves
    row_delta, col_delta = abs(row1 - row2), abs(col1 - col2)
    return (row_delta == 1 and col_delta == 0) or (row_delta == 0 and col_delta == 1)

# include at least: 

# a*:
# a test that checks the returned path starts at S and ends at G

# a test that recomputes the total cost of the returned path and matches total_cost
class TestAStar(unittest.TestCase):
    def setUp(self):
        # Example config for testing
        self.config = {
            "paradigm": "astar",
            "m": 5,
            "n": 5,
            "start": [0, 0],
            "goal": [4, 4],
            "min_cost": 1,
            "max_cost": 5,
            "seed": 123
        }

    # checks every move in the path is legal (up, down, left, right) and within bounds
    def test_moves_are_legal(self):
        grid = create_grid(self.config['m'], self.config['n'], self.config['start'], self.config['goal'])
        path, *_ = astar(self.config)
        for i in range(1, len(path)):
            self.assertTrue(is_legal_move(path[i-1], path[i], self.config['m'], self.config['n']))


# tsp:
# a test that verifies every city appears exactly once in the returned tour
# a test that verifies the tour is a closed cycle (returns to the start)
# a sanity test that confirms hill climbing terminates (no improving neighbor exists)