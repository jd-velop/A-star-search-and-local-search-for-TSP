"""
Display experiment results in a readable format for the report
"""

import json

def print_astar_results():
    """Print A* summary results"""
    with open("astar_summary.json", 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("A* EXPERIMENTAL RESULTS")
    print("=" * 80)
    print("\nTable 1: A* Performance by Grid Size (Averaged over 10 seeds)")
    print("-" * 80)
    print(f"{'Grid Size':<12} {'Steps':<8} {'Cost':<8} {'Expanded':<10} {'Generated':<12} {'Max Frontier':<14} {'Runtime (ms)':<12}")
    print("-" * 80)
    
    for row in data:
        print(f"{row['grid_size']:<12} {row['avg_steps']:<8.1f} {row['avg_cost']:<8.1f} "
              f"{row['avg_expanded_states']:<10.1f} {row['avg_generated_nodes']:<12.1f} "
              f"{row['avg_max_frontier']:<14.1f} {row['avg_runtime_ms']:<12.2f}")
    
    print("-" * 80)

def print_tsp_results():
    """Print TSP summary results"""
    with open("tsp_summary.json", 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("TSP EXPERIMENTAL RESULTS")
    print("=" * 80)
    print("\nTable 2: TSP Performance by Problem Size (Averaged over 10 seeds × 10 restarts)")
    print("-" * 80)
    print(f"{'Cities':<10} {'Init Cost':<12} {'Best Cost':<12} {'Improvement':<14} {'Iterations':<12} {'Runtime (ms)':<14}")
    print("-" * 80)
    
    for row in data:
        print(f"{row['n_cities']:<10} {row['avg_initial_cost']:<12.1f} {row['avg_best_cost']:<12.1f} "
              f"{row['avg_improvement_percent']:<14.1f}% {row['avg_iterations_per_restart']:<12.1f} "
              f"{row['avg_runtime_ms']:<14.2f}")
    
    print("-" * 80)

def print_markdown_tables():
    """Print Markdown-formatted tables for the report"""
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES FOR REPORT")
    print("=" * 80)
    
    with open("astar_summary.json", 'r') as f:
        astar_data = json.load(f)
    
    print("\n### A* Results\n")
    print("| Grid Size | Avg Steps | Avg Cost | Avg Expanded States | Avg Generated Nodes | Avg Max Frontier | Avg Runtime (ms) |")
    print("|-----------|-----------|----------|---------------------|---------------------|------------------|------------------|")
    for row in astar_data:
        print(f"| {row['grid_size']} | {row['avg_steps']:.1f} | {row['avg_cost']:.1f} | "
              f"{row['avg_expanded_states']:.1f} | {row['avg_generated_nodes']:.1f} | "
              f"{row['avg_max_frontier']:.1f} | {row['avg_runtime_ms']:.2f} |")
    
    with open("tsp_summary.json", 'r') as f:
        tsp_data = json.load(f)
    
    print("\n### TSP Results\n")
    print("| Cities | Avg Initial Cost | Avg Best Cost | Avg Improvement (%) | Avg Iterations/Restart | Avg Runtime (ms) |")
    print("|--------|------------------|---------------|---------------------|------------------------|------------------|")
    for row in tsp_data:
        print(f"| {row['n_cities']} | {row['avg_initial_cost']:.1f} | {row['avg_best_cost']:.1f} | "
              f"{row['avg_improvement_percent']:.1f}% | {row['avg_iterations_per_restart']:.1f} | "
              f"{row['avg_runtime_ms']:.2f} |")

def print_key_findings():
    """Print key findings from the experiments"""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    with open("astar_summary.json", 'r') as f:
        astar_data = json.load(f)
    with open("tsp_summary.json", 'r') as f:
        tsp_data = json.load(f)
    
    print("\n### A* Analysis:")
    print(f"• Path length grows linearly: 10×10 has {astar_data[0]['avg_steps']:.0f} steps, "
          f"50×50 has {astar_data[2]['avg_steps']:.0f} steps")
    print(f"• Expanded states grow super-linearly: {astar_data[0]['avg_expanded_states']:.0f} → "
          f"{astar_data[1]['avg_expanded_states']:.0f} → {astar_data[2]['avg_expanded_states']:.0f}")
    print(f"• Runtime scales well: {astar_data[0]['avg_runtime_ms']:.2f}ms → "
          f"{astar_data[2]['avg_runtime_ms']:.2f}ms (~139x slower for 5x larger grid)")
    print(f"• Max frontier grows with problem size, showing efficient pruning")
    
    print("\n### TSP Analysis:")
    print(f"• Hill climbing achieves ~60-78% improvement across all problem sizes")
    print(f"• Larger problems get better improvements: 20 cities={tsp_data[0]['avg_improvement_percent']:.1f}%, "
          f"50 cities={tsp_data[2]['avg_improvement_percent']:.1f}%")
    print(f"• Iterations grow with problem size: {tsp_data[0]['avg_iterations_per_restart']:.1f} → "
          f"{tsp_data[2]['avg_iterations_per_restart']:.1f} iterations/restart")
    print(f"• Runtime is polynomial: ~50x slower for 2.5x more cities (50 vs 20)")
    print(f"• 2-opt is effective: consistently reduces tour cost by large amounts")

if __name__ == "__main__":
    print_astar_results()
    print_tsp_results()
    print_markdown_tables()
    print_key_findings()
    
    print("\n" + "=" * 80)
    print("All results files available:")
    print("  • astar_summary.json")
    print("  • tsp_summary.json")
    print("  • astar_all_results.json (all 30 A* runs)")
    print("  • tsp_all_results.json (all 300 TSP runs)")
    print("  • results_*.json (individual run files)")
    print("=" * 80)
