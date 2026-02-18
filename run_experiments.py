"""
Experiment runner for A* and TSP algorithms
Runs all required experiments and generates summary statistics
"""

import json
import os
from main import astar, tsp, AStarConfig, TSPConfig

# Experiment parameters
SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]

# A* parameters
ASTAR_GRID_SIZES = [(10, 10), (25, 25), (50, 50)]
ASTAR_MIN_COST = 1
ASTAR_MAX_COST = 10

# TSP parameters
TSP_CITY_COUNTS = [20, 30, 50]
TSP_BOX_SIZE = 100
TSP_RESTARTS = 10

def run_astar_experiments():
    """Run all A* experiments"""
    print("=" * 60)
    print("Running A* Experiments")
    print("=" * 60)
    
    all_results = []
    
    for grid_size in ASTAR_GRID_SIZES:
        m, n = grid_size
        print(f"\nGrid Size: {m}x{n}")
        
        for seed in SEEDS:
            print(f"  Running seed {seed}...", end=" ")
            
            # Create config
            config_dict = {
                "paradigm": "astar",
                "m": m,
                "n": n,
                "start": [0, 0],
                "goal": [m-1, n-1],
                "min_cost": ASTAR_MIN_COST,
                "max_cost": ASTAR_MAX_COST,
                "seed": seed
            }
            config = AStarConfig(config_dict)
            
            # Run A*
            path, path_cost, steps, expanded_states, generated_nodes, max_frontier_size, runtime_ms = astar(config)
            
            # Store results
            result = {
                "grid_size": f"{m}x{n}",
                "m": m,
                "n": n,
                "seed": seed,
                "steps": steps,
                "total_cost": path_cost,
                "expanded_states": expanded_states,
                "generated_nodes": generated_nodes,
                "max_frontier_size": max_frontier_size,
                "runtime_ms": runtime_ms,
                "status": "success" if path else "failure"
            }
            all_results.append(result)
            
            # Save individual result
            filename = f"results_astar_{m}x{n}_seed{seed}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✓ (cost: {path_cost}, steps: {steps}, time: {runtime_ms}ms)")
    
    # Save all results
    with open("astar_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def run_tsp_experiments():
    """Run all TSP experiments"""
    print("\n" + "=" * 60)
    print("Running TSP Experiments")
    print("=" * 60)
    
    all_results = []
    
    for n_cities in TSP_CITY_COUNTS:
        print(f"\nCities: {n_cities}")
        
        for seed in SEEDS:
            print(f"  Running seed {seed}...", end=" ")
            
            # Create config
            config_dict = {
                "paradigm": "tsp",
                "n_cities": n_cities,
                "box_size": TSP_BOX_SIZE,
                "restarts": TSP_RESTARTS,
                "seed": seed,
                "operator": "two-opt"
            }
            config = TSPConfig(config_dict)
            
            # Run TSP
            best_tour, best_cost, initial_tour, initial_cost, iterations_list, runtime_ms = tsp(config)
            
            # Store results
            result = {
                "n_cities": n_cities,
                "seed": seed,
                "restarts": TSP_RESTARTS,
                "initial_cost": round(initial_cost, 2),
                "best_cost": round(best_cost, 2),
                "improvement": round(initial_cost - best_cost, 2),
                "improvement_percent": round((initial_cost - best_cost) / initial_cost * 100, 2),
                "iterations_per_restart": iterations_list,
                "total_iterations": sum(iterations_list),
                "avg_iterations": round(sum(iterations_list) / len(iterations_list), 2),
                "runtime_ms": runtime_ms
            }
            all_results.append(result)
            
            # Save individual result
            filename = f"results_tsp_{n_cities}cities_{TSP_RESTARTS}restarts_seed{seed}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✓ (best: {best_cost:.1f}, improvement: {result['improvement_percent']:.1f}%, time: {runtime_ms}ms)")
    
    # Save all results
    with open("tsp_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def compute_astar_summary(results):
    """Compute summary statistics for A* experiments"""
    print("\n" + "=" * 60)
    print("A* Summary Statistics")
    print("=" * 60)
    
    summary = []
    
    for grid_size in ASTAR_GRID_SIZES:
        m, n = grid_size
        grid_name = f"{m}x{n}"
        
        # Filter results for this grid size
        grid_results = [r for r in results if r["grid_size"] == grid_name]
        
        if not grid_results:
            continue
        
        # Compute averages
        avg_steps = sum(r["steps"] for r in grid_results) / len(grid_results)
        avg_cost = sum(r["total_cost"] for r in grid_results) / len(grid_results)
        avg_expanded = sum(r["expanded_states"] for r in grid_results) / len(grid_results)
        avg_generated = sum(r["generated_nodes"] for r in grid_results) / len(grid_results)
        avg_frontier = sum(r["max_frontier_size"] for r in grid_results) / len(grid_results)
        avg_runtime = sum(r["runtime_ms"] for r in grid_results) / len(grid_results)
        
        stats = {
            "grid_size": grid_name,
            "avg_steps": round(avg_steps, 1),
            "avg_cost": round(avg_cost, 1),
            "avg_expanded_states": round(avg_expanded, 1),
            "avg_generated_nodes": round(avg_generated, 1),
            "avg_max_frontier": round(avg_frontier, 1),
            "avg_runtime_ms": round(avg_runtime, 2)
        }
        summary.append(stats)
        
        # Print row
        print(f"\n{grid_name}:")
        print(f"  Avg Steps:           {stats['avg_steps']}")
        print(f"  Avg Cost:            {stats['avg_cost']}")
        print(f"  Avg Expanded States: {stats['avg_expanded_states']}")
        print(f"  Avg Generated Nodes: {stats['avg_generated_nodes']}")
        print(f"  Avg Max Frontier:    {stats['avg_max_frontier']}")
        print(f"  Avg Runtime:         {stats['avg_runtime_ms']} ms")
    
    # Save summary
    with open("astar_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def compute_tsp_summary(results):
    """Compute summary statistics for TSP experiments"""
    print("\n" + "=" * 60)
    print("TSP Summary Statistics")
    print("=" * 60)
    
    summary = []
    
    for n_cities in TSP_CITY_COUNTS:
        # Filter results for this problem size
        city_results = [r for r in results if r["n_cities"] == n_cities]
        
        if not city_results:
            continue
        
        # Compute averages
        avg_initial_cost = sum(r["initial_cost"] for r in city_results) / len(city_results)
        avg_best_cost = sum(r["best_cost"] for r in city_results) / len(city_results)
        avg_improvement = sum(r["improvement"] for r in city_results) / len(city_results)
        avg_improvement_pct = sum(r["improvement_percent"] for r in city_results) / len(city_results)
        avg_iterations = sum(r["avg_iterations"] for r in city_results) / len(city_results)
        avg_runtime = sum(r["runtime_ms"] for r in city_results) / len(city_results)
        
        stats = {
            "n_cities": n_cities,
            "avg_initial_cost": round(avg_initial_cost, 1),
            "avg_best_cost": round(avg_best_cost, 1),
            "avg_improvement": round(avg_improvement, 1),
            "avg_improvement_percent": round(avg_improvement_pct, 1),
            "avg_iterations_per_restart": round(avg_iterations, 1),
            "avg_runtime_ms": round(avg_runtime, 2)
        }
        summary.append(stats)
        
        # Print row
        print(f"\n{n_cities} Cities:")
        print(f"  Avg Initial Cost:    {stats['avg_initial_cost']}")
        print(f"  Avg Best Cost:       {stats['avg_best_cost']}")
        print(f"  Avg Improvement:     {stats['avg_improvement']} ({stats['avg_improvement_percent']}%)")
        print(f"  Avg Iterations:      {stats['avg_iterations_per_restart']}")
        print(f"  Avg Runtime:         {stats['avg_runtime_ms']} ms")
    
    # Save summary
    with open("tsp_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def generate_latex_table(astar_summary, tsp_summary):
    """Generate LaTeX table code for the report"""
    print("\n" + "=" * 60)
    print("LaTeX Tables for Report")
    print("=" * 60)
    
    # A* table
    print("\nA* Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|r|r|r|r|r|r|}")
    print("\\hline")
    print("Grid Size & Avg Steps & Avg Cost & Avg Expanded & Avg Generated & Avg Max Frontier & Avg Runtime (ms) \\\\")
    print("\\hline")
    for stats in astar_summary:
        print(f"{stats['grid_size']} & {stats['avg_steps']:.1f} & {stats['avg_cost']:.1f} & "
              f"{stats['avg_expanded_states']:.1f} & {stats['avg_generated_nodes']:.1f} & "
              f"{stats['avg_max_frontier']:.1f} & {stats['avg_runtime_ms']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{A* Performance by Grid Size (Averaged over 10 seeds)}")
    print("\\end{table}")
    
    # TSP table
    print("\nTSP Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|r|r|r|r|r|}")
    print("\\hline")
    print("Cities & Avg Initial Cost & Avg Best Cost & Avg Improvement (\\%) & Avg Iterations & Avg Runtime (ms) \\\\")
    print("\\hline")
    for stats in tsp_summary:
        print(f"{stats['n_cities']} & {stats['avg_initial_cost']:.1f} & {stats['avg_best_cost']:.1f} & "
              f"{stats['avg_improvement_percent']:.1f}\\% & {stats['avg_iterations_per_restart']:.1f} & "
              f"{stats['avg_runtime_ms']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{TSP Performance by Problem Size (Averaged over 10 seeds × 10 restarts)}")
    print("\\end{table}")

def main():
    """Run all experiments and generate summaries"""
    print("\n" + "=" * 60)
    print("EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"\nA* Experiments: {len(ASTAR_GRID_SIZES)} grid sizes × {len(SEEDS)} seeds = {len(ASTAR_GRID_SIZES) * len(SEEDS)} runs")
    print(f"TSP Experiments: {len(TSP_CITY_COUNTS)} sizes × {len(SEEDS)} seeds × {TSP_RESTARTS} restarts = {len(TSP_CITY_COUNTS) * len(SEEDS) * TSP_RESTARTS} runs")
    print(f"\nTotal runs: {len(ASTAR_GRID_SIZES) * len(SEEDS) + len(TSP_CITY_COUNTS) * len(SEEDS)} (not counting individual restarts)")
    
    input("\nPress Enter to start experiments...")
    
    # Run A* experiments
    astar_results = run_astar_experiments()
    astar_summary = compute_astar_summary(astar_results)
    
    # Run TSP experiments
    tsp_results = run_tsp_experiments()
    tsp_summary = compute_tsp_summary(tsp_results)
    
    # Generate LaTeX tables
    generate_latex_table(astar_summary, tsp_summary)
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("\nFiles generated:")
    print("  - astar_all_results.json (all A* runs)")
    print("  - tsp_all_results.json (all TSP runs)")
    print("  - astar_summary.json (A* statistics)")
    print("  - tsp_summary.json (TSP statistics)")
    print("  - results_*.json (individual run files)")
    print("\nYou can now use these results in your report!")

if __name__ == "__main__":
    main()
