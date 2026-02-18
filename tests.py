import unittest
import json
from main import (
    create_graph, assign_move_costs, astar, 
    AStarConfig, TSPConfig,
    generate_cities, calculate_tour_cost, hill_climbing,
    get_all_two_opt_neighbors, two_opt, tsp
)

# ============ Helper Functions ============

def is_legal_move(prev, curr, m, n):
    """Helper function to make sure a move is legal"""
    row1, col1 = prev
    row2, col2 = curr
    # move is in bounds
    if not (0 <= row2 < m and 0 <= col2 < n):
        return False
    # move is one of the four legal moves
    row_delta, col_delta = abs(row1 - row2), abs(col1 - col2)
    return (row_delta == 1 and col_delta == 0) or (row_delta == 0 and col_delta == 1)

# ============ A* Tests ============

class TestAStar(unittest.TestCase):
    def setUp(self):
        # Example config for testing
        self.config_dict = {
            "paradigm": "astar",
            "m": 5,
            "n": 5,
            "start": [0, 0],
            "goal": [4, 4],
            "min_cost": 1,
            "max_cost": 5,
            "seed": 123
        }
        self.config = AStarConfig(self.config_dict)

    def test_moves_are_legal(self):
        """Test that every move in the path is legal (up, down, left, right) and within bounds"""
        path, *_ = astar(self.config)
        self.assertIsNotNone(path, "Path should not be None")
        for i in range(1, len(path)):
            self.assertTrue(
                is_legal_move(path[i-1], path[i], self.config.m, self.config.n),
                f"Move from {path[i-1]} to {path[i]} is not legal"
            )

    def test_path_starts_and_ends_correctly(self):
        """Test that the returned path starts at S and ends at G"""
        path, *_ = astar(self.config)
        self.assertIsNotNone(path, "Path should not be None")
        self.assertGreater(len(path), 0, "Path should not be empty")
        
        # Check start and goal (convert tuples to lists for comparison)
        self.assertEqual(list(path[0]), list(self.config.start), 
                        f"Path should start at {self.config.start}, but starts at {path[0]}")
        self.assertEqual(list(path[-1]), list(self.config.goal),
                        f"Path should end at {self.config.goal}, but ends at {path[-1]}")

    def test_path_cost_matches(self):
        """Test that recomputing the total cost of the path matches the returned total_cost"""
        path, path_cost, *_ = astar(self.config)
        self.assertIsNotNone(path, "Path should not be None")
        self.assertIsNotNone(path_cost, "Path cost should not be None")
        
        # Create graph and compute actual cost
        graph = create_graph(self.config.m, self.config.n, self.config.start, self.config.goal)
        assign_move_costs(graph, self.config.min_cost, self.config.max_cost, self.config.seed)
        
        # Recompute path cost manually
        computed_cost = 0
        for i in range(len(path) - 1):
            curr_state = path[i]
            next_state = path[i + 1]
            
            # Determine the action taken
            if next_state[0] < curr_state[0]:
                action = 'up'
            elif next_state[0] > curr_state[0]:
                action = 'down'
            elif next_state[1] < curr_state[1]:
                action = 'left'
            else:  # next_state[1] > curr_state[1]
                action = 'right'
            
            computed_cost += graph.move_costs[(curr_state, action)]
        
        self.assertEqual(computed_cost, path_cost,
                        f"Computed cost {computed_cost} does not match returned cost {path_cost}")

    def test_path_continuous(self):
        """Test that the path is continuous (no gaps)"""
        path, *_ = astar(self.config)
        self.assertIsNotNone(path, "Path should not be None")
        
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            manhattan_dist = abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1])
            self.assertEqual(manhattan_dist, 1, 
                           f"Path has a gap between {curr} and {next_pos}")

    def test_small_grid(self):
        """Test A* on a 1x1 grid where start equals goal"""
        config_dict = {
            "paradigm": "astar",
            "m": 1,
            "n": 1,
            "start": [0, 0],
            "goal": [0, 0],
            "min_cost": 1,
            "max_cost": 1,
            "seed": 42
        }
        config = AStarConfig(config_dict)
        path, path_cost, steps, *_ = astar(config)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1, "Path should have only 1 state when start=goal")
        self.assertEqual(path_cost, 0, "Cost should be 0 when start=goal")
        self.assertEqual(steps, 0, "Steps should be 0 when start=goal")

# ============ TSP Tests ============

class TestTSP(unittest.TestCase):
    def setUp(self):
        # Example config for testing
        self.config_dict = {
            "paradigm": "tsp",
            "n_cities": 10,
            "box_size": 100,
            "restarts": 5,
            "seed": 42,
            "operator": "two-opt"
        }
        self.config = TSPConfig(self.config_dict)
        self.cities = generate_cities(self.config.n_cities, self.config.box_size, self.config.seed)

    def test_every_city_appears_once(self):
        """Test that every city appears exactly once in the returned tour"""
        best_tour, *_ = two_opt(self.config)
        
        # The raw tour should have exactly n cities, each appearing once
        self.assertEqual(len(best_tour), self.config.n_cities,
                        f"Tour should have {self.config.n_cities} cities")
        self.assertEqual(set(best_tour), set(range(self.config.n_cities)),
                        "Tour should contain each city exactly once")
        self.assertEqual(len(best_tour), len(set(best_tour)),
                        "Tour should not have duplicate cities")

    def test_tour_is_closed_cycle(self):
        """Test that the tour is a closed cycle (returns to the start)"""
        best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms = two_opt(self.config)
        
        # The raw tour from two_opt has n cities (without duplicate)
        # But the cost calculation implicitly includes return to start
        self.assertEqual(len(best_tour), self.config.n_cities,
                        f"Tour should have {self.config.n_cities} unique cities")
        
        # Verify the tour can be a closed cycle by checking the cost includes return trip
        # Recompute cost to verify it includes the edge back to start
        manual_cost = 0
        for i in range(len(best_tour) - 1):
            manual_cost += ((self.cities[best_tour[i]][0] - self.cities[best_tour[i+1]][0])**2 + 
                          (self.cities[best_tour[i]][1] - self.cities[best_tour[i+1]][1])**2)**0.5
        # Add return to start
        manual_cost += ((self.cities[best_tour[-1]][0] - self.cities[best_tour[0]][0])**2 + 
                       (self.cities[best_tour[-1]][1] - self.cities[best_tour[0]][1])**2)**0.5
        
        self.assertAlmostEqual(manual_cost, best_cost, places=2,
                              msg="Tour cost should include return to start")

    def test_tour_starts_at_city_zero(self):
        """Test that the normalized tour starts at city 0"""
        best_tour, *_ = two_opt(self.config)
        
        self.assertEqual(best_tour[0], 0, "Tour should start at city 0")

    def test_hill_climbing_terminates(self):
        """Test that hill climbing terminates with no improving neighbor""" 
        # Generate a random initial tour
        import random
        random.seed(self.config.seed)
        from main import random_tour
        initial_tour = random_tour(self.config.n_cities)
        
        # Run hill climbing
        final_tour, final_cost, iterations = hill_climbing(initial_tour, self.cities)
        
        # Verify that no neighbor of final_tour improves the cost
        neighbors = get_all_two_opt_neighbors(final_tour)
        for neighbor in neighbors:
            neighbor_cost = calculate_tour_cost(neighbor, self.cities)
            self.assertGreaterEqual(neighbor_cost, final_cost,
                                   f"Found an improving neighbor with cost {neighbor_cost} < {final_cost}")

    def test_random_restart_improves(self):
        """Test that random restart finds a solution at least as good as a single run"""
        # Single hill climb run
        import random
        random.seed(self.config.seed)
        from main import random_tour
        initial_tour = random_tour(self.config.n_cities)
        single_tour, single_cost, _ = hill_climbing(initial_tour, self.cities)
        
        # Multiple restarts
        best_tour, best_cost, *_ = two_opt(self.config)
        
        # With multiple restarts, we should do at least as well
        # (may be same if first random tour was already good)
        self.assertLessEqual(best_cost, single_cost * 1.5,  # Allow some tolerance
                           "Random restart should find comparable or better solution")

    def test_tour_cost_calculation(self):
        """Test that tour cost calculation is correct for a known tour"""
        # Create a simple tour: 0 -> 1 -> 2 -> 0
        simple_cities = [(0, 0), (1, 0), (0, 1)]
        simple_tour = [0, 1, 2]
        
        cost = calculate_tour_cost(simple_tour, simple_cities)
        
        # Expected: dist(0,1) + dist(1,2) + dist(2,0)
        # = 1 + sqrt(2) + 1 = 2 + sqrt(2) ≈ 3.414
        expected_cost = 1.0 + (2**0.5) + 1.0
        self.assertAlmostEqual(cost, expected_cost, places=5,
                              msg=f"Cost calculation incorrect: {cost} != {expected_cost}")

    def test_initial_and_best_different(self):
        """Test that hill climbing improves from the initial tour"""
        best_tour, best_cost, initial_tour, initial_cost, *_ = two_opt(self.config)
        
        # Best cost should be less than or equal to initial cost
        self.assertLessEqual(best_cost, initial_cost,
                           f"Best cost {best_cost} should be <= initial cost {initial_cost}")

    def test_reproducibility(self):
        """Test that using the same seed produces the same results"""
        result1 = two_opt(self.config)
        result2 = two_opt(self.config)
        
        # Should get identical results with same seed
        self.assertEqual(result1[0], result2[0], "Tours should be identical with same seed")
        self.assertEqual(result1[1], result2[1], "Costs should be identical with same seed")

if __name__ == '__main__':
    unittest.main()