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

def astar(config):
    m, n = config['m'], config['n']
    start = tuple(config['start'])
    goal = tuple(config['goal'])
    min_cost, max_cost = config['min_cost'], config['max_cost']
    seed = config['seed']
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
        astar(config)
    elif paradigm == 'tsp':
        tsp(config)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

if __name__ == "__main__":
    main()