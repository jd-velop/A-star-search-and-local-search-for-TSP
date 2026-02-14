# implement and compare two different AI problem-solving paradigms: A* heuristic search for shortest-path planning and local search for the Traveling Salesman Problem (TSP). Then run experiments to analyze when each approach is appropriate.


# ========== What to build ==========
# Solves a grid-based shortest-path problem using A* (Graph Search)
# Solves a 2D geometric TSP instance using hill climbing and random restarts.
# Uses a random seed so experiments are reproducible.
# Prints results to the terminal and saves results to a JSON file.
# NOTE:
# No obstactles in the grid world (every cell is reachable unless the grid is 1 x 1).
# A* must use an admissible heuristic (manhattan distance required)
# For TSP local search, you may choose one neighborhood operator: 2-opt, swap, or insert.
# ==================================

# ======= Problem Setup (A*) ========
# (grid world for A*)
# State: the agent's current location: state = (row,col) where 0 <= row < m, 0 <= col < n
# Actions: 4 grid moves: up, down, left, right (no obstacles, stay within grid bounds)
# Costs: Each legal move has a cost generated randomly using a user-specified range: min_cost <= cost <= max_cost. You may define the move cost: cost of the directed edge from one cell to the next
# Heuristic: You must implement an admissible heuristic for A*: Manhattan distance (required): h(state) = abs(row - row_goal) + abs(col - col_goal). Euclidean distance (optional): h(state) = sqrt((row - row_goal)^2 + (col - col_goal)^2)
# Reproducibility: Program must accept a random seed as input to ensure reproducibility of results.
# ==================================

# ===== Algorithm to Implement =====
# A* (Graph Search) using a Priority Queue ordered by f(n) = g(n) + h(n)
# An explored set (or equivalent closed set) to avoid repeated expansions
# A bestCost map (or equivalent) to keep the lowest g(n) found so far for a state

# A* pseudocode from class is required. In particular:
# Always pop from the frontier by minimum f(n)
# Update and reinsert a state if you discover a cheaper g(n) path to it
# Return the least-cost path (optimal) when using an admissible heuristic
# ==================================

# ======= Problem Setup (TSP) ======
# (2D geometric TSP)
# State: a complete TSP tour (a permutation of the cities): tour = [city1, city2, ..., cityN] where each city appears exactly once
# Objective function: The cost of a tour is the total Euclidean length: distance(cityI, cityI+1) for i = 1 to n-1, plus distance(cityN, city1) to return to the start
# Neighborhood operator (choose 1): 2-opt: remove two edges and reconnect (segment reversal), swap: swap two city positions, insertL remove one city and insert it elsewhere. The operator used must be specified and justified in the report.
# ==================================

# ===== Algorithm to Implement =====
# (Local Search for TSP)
# Hill climbing (first-improvement or best-improvement)
# Random-restart hill climbing (run hill climbing from multiple random initial tours; keep the best)
# You must maintain a valid tour at all times. Every neightbor generated must both contain each city exactly once and represent a closed cycle (returns to the start).
# ==================================

# ========= Program Inputs =========
# A* inputs:
# Grid size: m, n
# Start: S = (row_start, col_start)
# Goal: G = (row_goal, col_goal)
# Cost range: min_cost, max_cost (integers)
# Random seed: seed
# TSP inputs:
# Number of cities: n
# Coordinate range (or bounding box), e.g., 0..100
# Number of random restarts: restarts
# Random seed: seed
# Neighborhood operator: two-opt, swap, or insert
# ==================================

# ======== Program Outputs =========
# Each run, results should be printted to the terminal and saved to a JSON file
# A* outputs:
# algorithm ("astar")
# m, n, start, goal
# min_cost, max_cost, seed
# heuristic ("manhattan" or "euclidean")
# path (sequence of states or actions)
# steps (number or moves)
# total_cost
# expanded_states
# generated_nodes
# max_frontier_size
# runtime_ms
# status ("success" or "failure")

# TSP outputs:
# algorithm ("tsp_local_search")
# n_cities, seed, restarts, operator
# initial_tour (optional), initial_cost
# best_tour (a valid permutation of cities)
# best_cost
# iterations (to converge per restart)
# runtime_ms
# ==================================

import json
import random
import time

class Node:
    def __init__(self, state=None, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

class PriorityQueue: # priority queue class for A* frontier, ligns up with pseudocode from class
    def __init__(self):
        self.elements = []

    def push(self, node):
        self.elements.append(node)
    
    def pop(self):
        self.elements.sort(key=lambda x: x.path_cost) # sort by f(n) = g(n) + h(n)
        return self.elements.pop(0) # pop the node with the lowest f(n)
    
    def is_empty(self):
        return len(self.elements) == 0
    
class Graph:
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
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1] # reverse the resulting path

def Succ(state, graph): # successor function for A* (generates neighbors and their costs)
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
    m, n = config['m'], config['n']
    start = tuple(config['start'])
    goal = tuple(config['goal'])
    min_cost, max_cost = config['min_cost'], config['max_cost']
    seed = config['seed']

    graph = create_graph(m, n, start, goal)
    assign_move_costs(graph, min_cost, max_cost, seed)

    # logic
    frontier = PriorityQueue()
    frontier.push(Node(state=start, path_cost=0))
    bestCost = {}
    while not frontier.is_empty():
        n = frontier.pop()
        if n.state == goal: return ExtractPath(n)
        # expand node
        if n.state not in bestCost or n.path_cost < bestCost[n.state]:
            bestCost[n.state] = n.path_cost
            # generate neighbors
            for action, child_state, cost in Succ(n.state, graph):
                child_path_cost= n.path_cost + cost
                child_node = Node(state=child_state, parent=n, action=action, path_cost=child_path_cost)
                frontier.push(child_node)
    return None # failure if frontier is empty and goal not found


# Number of cities: n
# Coordinate range (or bounding box), e.g., 0..100
# Number of random restarts: restarts
# Random seed: seed
# Neighborhood operator: two-opt, swap, or insert

def tsp(config):
    n = config['n_cities']
    box_size = config['box_size']
    restarts = config['restarts']
    seed = config['seed']
    operator = config['operator']

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    paradigm = config['paradigm']
    if paradigm == 'astar':
        result = astar(config)
        print("Path found:", result)
    elif paradigm == 'tsp':
        tsp(config)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

if __name__ == "__main__":
    main()