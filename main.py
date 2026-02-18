import json
import random
import time

# ============ Configuration Classes ============

class AStarConfig:
    """Configuration for A* algorithm"""
    def __init__(self, config_dict):
        self.m = config_dict['m']
        self.n = config_dict['n']
        self.start = tuple(config_dict['start'])  # Convert to tuple for hashing
        self.goal = tuple(config_dict['goal'])
        self.min_cost = config_dict['min_cost']
        self.max_cost = config_dict['max_cost']
        self.seed = config_dict['seed']
        self.heuristic = config_dict.get('heuristic', 'manhattan')
    
    def to_dict(self):
        """Convert config to dictionary for output"""
        return {
            'm': self.m,
            'n': self.n,
            'start': list(self.start),
            'goal': list(self.goal),
            'min_cost': self.min_cost,
            'max_cost': self.max_cost,
            'seed': self.seed,
            'heuristic': self.heuristic
        }

class TSPConfig:
    """Configuration for TSP algorithm"""
    def __init__(self, config_dict):
        self.n_cities = config_dict['n_cities']
        self.box_size = config_dict['box_size']
        self.restarts = config_dict['restarts']
        self.seed = config_dict['seed']
        self.operator = config_dict['operator']
    
    def to_dict(self):
        """Convert config to dictionary for output"""
        return {
            'n_cities': self.n_cities,
            'box_size': self.box_size,
            'restarts': self.restarts,
            'seed': self.seed,
            'operator': self.operator
        }

class ConfigLoader:
    """Handles loading and parsing configuration files"""
    
    @staticmethod
    def load_from_file(filepath='config.json'):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as config_file:
            config_dict = json.load(config_file)
        return ConfigLoader.parse_config(config_dict)
    
    @staticmethod
    def parse_config(config_dict):
        """Parse config dictionary and return appropriate config object"""
        paradigm = config_dict.get('paradigm')
        
        if paradigm == 'astar':
            return 'astar', AStarConfig(config_dict)
        elif paradigm == 'tsp':
            return 'tsp', TSPConfig(config_dict)
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

# ============ Core Classes ============

class Node:
    """Node class for A* search"""
    def __init__(self, state=None, parent=None, action=None, path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost  # g(n)
        self.heuristic = heuristic  # h(n)
        self.f_cost = path_cost + heuristic  # f(n) = g(n) + h(n)

class PriorityQueue: # priority queue class for A* frontier, ligns up with pseudocode from class
    """Priority queue for A* frontier, sorted by f(n) = g(n) + h(n)"""
    def __init__(self):
        self.elements = []

    def push(self, node):
        self.elements.append(node)
    
    def pop(self):
        self.elements.sort(key=lambda x: x.f_cost) # sort by f(n) = g(n) + h(n)
        return self.elements.pop(0) # pop the node with the lowest f(n)
    
    def is_empty(self):
        return len(self.elements) == 0
    
class Graph:
    """Graph class to represent the grid and move costs for A* search"""
    def __init__(self, m, n, start, goal):
        self.m = m
        self.n = n
        self.start = start
        self.goal = goal

        self.move_costs = {} # dictionary to store move costs: (state, action) -> cost    

def create_graph(m, n, start, goal):
    return Graph(m, n, start, goal)

def assign_move_costs(graph, min_cost, max_cost, seed):
    random.seed(seed)
    for row in range(graph.m):
        for col in range(graph.n):
            state = (row, col)
            for action in ['up', 'down', 'left', 'right']:
                if action == 'up' and row == 0:
                    continue
                elif action == 'down' and row == graph.m - 1:
                    continue
                elif action == 'left' and col == 0:
                    continue
                elif action == 'right' and col == graph.n - 1:
                    continue
                graph.move_costs[(state, action)] = random.randint(min_cost, max_cost) # seeded

def ExtractPath(node): # copy+paste from project 1
    """Follow parent pointers of given node to extract the path from start to goal"""
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1] # reverse the resulting path

def count_steps(node): # also copy+paste from project 1
    """Follow parent pointers to count the number of steps from start to goal"""
    steps = 0
    while node.parent:
        steps += 1
        node = node.parent
    return steps

def manhattan_distance(state, goal):
    """Calculate Manhattan distance heuristic"""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def Succ(state, graph): # successor function for A* (generates neighbors and their costs)
    """Successor function for A*: given a state, return a list of (action, new_state, cost) tuples for legal moves"""
    results = []
    for action in ['up', 'down', 'left', 'right']:
        row, col = state
        if action == 'up' and row > 0:
            new_state = (row - 1, col)
        elif action == 'down' and row < graph.m - 1:
            new_state = (row + 1, col)
        elif action == 'left' and col > 0:
            new_state = (row, col - 1)
        elif action == 'right' and col < graph.n - 1:
            new_state = (row, col + 1)
        else:
            continue
        cost = graph.move_costs[(state, action)]
        results.append((action, new_state, cost))
    return results

def astar(config):
    """A* algorithm using AStarConfig object"""
    graph = create_graph(config.m, config.n, config.start, config.goal)
    assign_move_costs(graph, config.min_cost, config.max_cost, config.seed)

    # logic
    start_time = time.time()
    frontier = PriorityQueue()
    h_start = manhattan_distance(config.start, config.goal)
    frontier.push(Node(state=config.start, path_cost=0, heuristic=h_start))
    bestCost = {}
    expanded_states = 0
    generated_nodes = 0
    max_frontier_size = 0
    while not frontier.is_empty():
        max_frontier_size = max(max_frontier_size, len(frontier.elements))
        n = frontier.pop()
        if n.state == config.goal:
            runtime_ms = round((time.time() - start_time) * 1000, 2)
            return ExtractPath(n), n.path_cost, count_steps(n), expanded_states, generated_nodes, max_frontier_size, runtime_ms
        # expand node
        if n.state not in bestCost or n.path_cost < bestCost[n.state]:
            bestCost[n.state] = n.path_cost
            expanded_states += 1
            # generate neighbors
            for action, child_state, cost in Succ(n.state, graph):
                child_path_cost = n.path_cost + cost
                child_heuristic = manhattan_distance(child_state, config.goal)
                child_node = Node(state=child_state, parent=n, action=action, path_cost=child_path_cost, heuristic=child_heuristic)
                frontier.push(child_node)
                generated_nodes += 1
    runtime_ms = round((time.time() - start_time) * 1000, 2)
    return None, None, None, expanded_states, generated_nodes, max_frontier_size, runtime_ms # failure if frontier is empty and goal not found


# ============ TSP Helper Functions ============

def generate_cities(n_cities, box_size, seed):
    """Generate n random cities within a bounding box [0, box_size] x [0, box_size]"""
    random.seed(seed)
    cities = []
    for i in range(n_cities):
        x = random.uniform(0, box_size)
        y = random.uniform(0, box_size)
        cities.append((x, y))
    return cities

def euclidean_distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5

def calculate_tour_cost(tour, cities):
    """Calculate total cost of a tour (closed cycle)"""
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += euclidean_distance(cities[tour[i]], cities[tour[i + 1]])
    # Add distance from last city back to first (closed cycle)
    cost += euclidean_distance(cities[tour[n - 1]], cities[tour[0]])
    return cost

def random_tour(n_cities):
    """Generate a random tour (permutation of cities)"""
    tour = list(range(n_cities))
    random.shuffle(tour)
    return tour

def normalize_tour(tour):
    """Normalize tour to start from city 0
    
    Since TSP tours are cycles, we can rotate them to start from any city.
    This function rotates the tour so it always starts from city 0 for consistency.
    """
    if not tour or tour[0] == 0:
        return tour
    
    # Find the index of city 0
    idx = tour.index(0)
    
    # Rotate the tour to start from city 0
    return tour[idx:] + tour[:idx]

def two_opt_swap(tour, i, j):
    """Perform 2-opt swap: reverse segment between indices i and j
    
    2-opt removes two edges and reconnects by reversing the segment.
    For a tour [0, 1, 2, 3, 4, 5], if i=1 and j=4:
    - Remove edges (tour[i-1], tour[i]) and (tour[j], tour[j+1])
    - Reverse segment tour[i:j+1]
    - Result: [0, 4, 3, 2, 1, 5]
    """
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def get_all_two_opt_neighbors(tour):
    """Generate all possible 2-opt neighbors of a tour"""
    neighbors = []
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            neighbor = two_opt_swap(tour, i, j)
            neighbors.append(neighbor)
    return neighbors

def hill_climbing(initial_tour, cities):
    """Hill climbing for TSP using 2-opt neighborhood
    
    Args:
        initial_tour: starting tour (permutation of city indices)
        cities: list of (x, y) coordinates
    
    Returns:
        best_tour, best_cost, iterations
    """
    current_tour = initial_tour[:]
    current_cost = calculate_tour_cost(current_tour, cities)
    iterations = 0
    
    while True:
        improved = False
        neighbors = get_all_two_opt_neighbors(current_tour)
        

        # Best-improvement: evaluate all neighbors, pick the best
        best_neighbor = None
        best_neighbor_cost = current_cost
        
        for neighbor in neighbors:
            neighbor_cost = calculate_tour_cost(neighbor, cities)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
        
        if best_neighbor is not None:
            current_tour = best_neighbor
            current_cost = best_neighbor_cost
            improved = True
            iterations += 1
            
        if not improved:
            break
    
    # Normalize tour to start from city 0
    current_tour = normalize_tour(current_tour)
    return current_tour, current_cost, iterations

def random_restart_hill_climbing(n_cities, cities, restarts, seed):
    """Random-restart hill climbing for TSP
    
    Run hill climbing from multiple random initial tours and keep the best.
    
    Returns:
        best_tour, best_cost, initial_tour, initial_cost, iterations_per_restart, total_iterations
    """
    random.seed(seed)
    
    # Track the overall best
    overall_best_tour = None
    overall_best_cost = float('inf')
    initial_tour = None
    initial_cost = None
    iterations_per_restart = []
    
    for restart in range(restarts):
        # Generate random initial tour
        tour = random_tour(n_cities)
        tour_cost = calculate_tour_cost(tour, cities)
        
        # save the initial tour and cost
        if restart == 0:
            initial_tour = normalize_tour(tour[:])
            initial_cost = tour_cost
        
        # Run hill climbing from this initial tour
        best_tour, best_cost, iterations = hill_climbing(tour, cities)
        iterations_per_restart.append(iterations)
        
        # Update overall best
        if best_cost < overall_best_cost:
            overall_best_tour = best_tour
            overall_best_cost = best_cost
    
    total_iterations = sum(iterations_per_restart)
    
    return overall_best_tour, overall_best_cost, initial_tour, initial_cost, iterations_per_restart, total_iterations

# ============ TSP Main Functions ============

def tsp(config):
    """TSP algorithm using TSPConfig object"""
    if config.operator == 'two-opt':
        return two_opt(config)
    elif config.operator == 'swap':
        # implement swap operator 
        raise NotImplementedError("Swap operator not implemented")
    elif config.operator == 'insert':
        # TODO: implement insert operator
        raise NotImplementedError("Insert operator not implemented")
    else:
        raise ValueError(f"Unknown neighborhood operator: {config.operator}")

def two_opt(config):
    """Solves a 2D geometric TSP instance using hill climbing with 2-opt and random restarts"""
    start_time = time.time()
    
    # Generate cities
    cities = generate_cities(config.n_cities, config.box_size, config.seed)
    
    # Run random-restart hill climbing
    best_tour, best_cost, initial_tour, initial_cost, iterations_list, total_iterations = \
        random_restart_hill_climbing(config.n_cities, cities, config.restarts, config.seed)
    
    runtime_ms = round((time.time() - start_time) * 1000, 2)
    
    return best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms



def build_astar_output(config, path, path_cost, steps, expanded_states, generated_nodes, max_frontier_size, runtime_ms):
    """Build output dictionary for A* results"""
    status = "success" if path else "failure"
    
    output = {
        "algorithm": "astar",
        **config.to_dict(), # include all config parameters in output
        "path": path,
        "steps": steps,
        "total_cost": path_cost,
        "expanded_states": expanded_states,
        "generated_nodes": generated_nodes,
        "max_frontier_size": max_frontier_size,
        "runtime_ms": runtime_ms,
        "status": status
    }
    return output

def build_tsp_output(config, best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms):
    """Build output dictionary for TSP results"""
    # Add return to start city to make the closed cycle explicit
    initial_tour_with_return = initial_tour + [initial_tour[0]]
    best_tour_with_return = best_tour + [best_tour[0]]
    
    output = {
        "algorithm": "tsp_local_search",
        "n_cities": config.n_cities,
        "seed": config.seed,
        "restarts": config.restarts,
        "operator": config.operator,
        "initial_tour": initial_tour_with_return,
        "initial_cost": round(initial_cost, 2),
        "best_tour": best_tour_with_return,
        "best_cost": round(best_cost, 2),
        "iterations": iterations_list,
        "total_iterations": sum(iterations_list),
        "runtime_ms": runtime_ms
    }
    return output

def main():
    # Load configuration
    paradigm, config = ConfigLoader.load_from_file('config.json')
    
    # Run appropriate algorithm
    if paradigm == 'astar':
        path, path_cost, steps, expanded_states, generated_nodes, max_frontier_size, runtime_ms = astar(config)
        output = build_astar_output(config, path, path_cost, steps, expanded_states, generated_nodes, max_frontier_size, runtime_ms)
        # Generate filename: astar_<m>x<n>_seed<seed>.json
        filename = f"results_astar_{config.m}x{config.n}_seed{config.seed}.json"
    elif paradigm == 'tsp':
        best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms = tsp(config)
        output = build_tsp_output(config, best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms)
        # Generate filename: tsp_<n_cities>cities_<restarts>restarts_seed<seed>.json
        filename = f"results_tsp_{config.n_cities}cities_{config.restarts}restarts_seed{config.seed}.json"
    
    # Print results to terminal
    print(json.dumps(output))
    
    # Save results to JSON file
    with open(filename, 'w') as f:
        json.dump(output, f)
    
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()