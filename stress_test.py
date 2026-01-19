import subprocess
import re
import csv
import os

# --- Define Test Scenarios ---
# Format: (Row, Bay, Level, [List of Densities])
# Density affects TotalBoxes: R * B * L * Density
test_scenarios = [
    (2, 2, 2, [0.5, 0.8]),        # Small-scale baseline tests
    (4, 4, 4, [0.3, 0.6, 0.8]),   # Medium-scale scalability tests
    (6, 11, 8, [0.4, 0.6, 0.85])  # Realistic scale (Low, Mid, High density)
]

def run_test(r, b, l, density):
    """
    Executes a single test case by calling the generator and solver.
    """
    total_boxes = int(r * b * l * density)
    # Set mission count: 30 for large yards, or 50% of total boxes for small ones
    mission_count = min(30, int(total_boxes * 0.5)) if total_boxes > 0 else 0
    
    print(f">>> Testing Config: {r}x{b}x{l} | Density: {int(density*100)}% | Missions: {mission_count}")

    # 1. Invoke DataGenerator to create CSV files
    subprocess.run(["./generator", str(r), str(b), str(l), str(total_boxes), str(mission_count)], 
                   capture_output=True)

    # 2. Invoke Solver and capture terminal output
    result = subprocess.run(["./solver"], capture_output=True, text=True)
    output = result.stdout

    # 3. Parse output data using Regular Expressions
    # These patterns look for the specific strings printed by main.cpp
    orig_cost = re.search(r"Original Cost\s+:\s+(\d+)", output)
    opt_cost = re.search(r"Optimized Cost\s+:\s+(\d+)", output)
    ga_time = re.search(r"GA Optimization\s+:\s+([\d.]+)", output)
    total_time = re.search(r"Total Elapsed Time\s+:\s+([\d.]+)", output)

    return {
        "Scenario": f"{r}x{b}x{l}",
        "Density": f"{int(density*100)}%",
        "Total_Boxes": total_boxes,
        "Missions": mission_count,
        "Original_Cost": int(orig_cost.group(1)) if orig_cost else "N/A",
        "Optimized_Cost": int(opt_cost.group(1)) if opt_cost else "N/A",
        "GA_Time_Sec": ga_time.group(1) if ga_time else "N/A",
        "Total_Time_Sec": total_time.group(1) if total_time else "N/A"
    }

# --- Batch Execution Loop ---
all_results = []
for r, b, l, densities in test_scenarios:
    for d in densities:
        data = run_test(r, b, l, d)
        all_results.append(data)

# --- Generate Summary Report ---
report_file = "stress_test_report.csv"
with open(report_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n[Test Completed] Run finished for {len(all_results)} cases. Results saved to {report_file}")