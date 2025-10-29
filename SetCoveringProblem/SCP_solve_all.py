import os, time, pandas as pd, sys
from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

def solve_all_instances(algorithm_name, csv_filename, folder="SCP-Instances", num_instances=0, *args, **kwargs):
    """
    Runs the given solver across all SCP instances and saves results to 'results/'.

    Adds a 'solution_sets' column listing the selected sets in the final solution.

    Prints a clear summary at the end with:
        - Average, min, and max deviation
        - Number of optimal solutions found
        - Timing statistics (min, max, average, total)
    """

    solver_func = globals().get(algorithm_name)
    if solver_func is None or not callable(solver_func):
        raise ValueError(f"Solver '{algorithm_name}' not found or not callable.")

    # Prepare output folder
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_filename)

    # Load instances
    files = sorted(f for f in os.listdir(folder) if f.lower().startswith("scp"))
    if num_instances > 0:
        files = files[:num_instances]
    total_instances = len(files)

    results = []
    print(f"Solver: {algorithm_name}")
    start_all = time.time()

    for i, _ in enumerate(files, start=1):
        inst = SCPInstance(i - 1, folder=folder)
        opt = inst.opt_value if inst.opt_value is not None else 0

        t0 = time.time()
        sol = solver_func(inst, *args, **kwargs)
        elapsed = time.time() - t0

        cost = getattr(sol, "cost", None)
        selected_sets = getattr(sol, "selected", None)

        # Convert selected sets to CSV-friendly string
        if selected_sets is not None:
            if isinstance(selected_sets, (set, list)):
                solution_str = ",".join(map(str, sorted(selected_sets)))
            else:
                solution_str = str(selected_sets)
        else:
            solution_str = ""

        deviation = round(100 * (cost - opt) / opt, 2) if opt and cost is not None else None

        results.append({
            "instance_name": inst.name,
            "opt_value": opt,
            "solver": algorithm_name,
            "solution_cost": cost,
            "deviation_%": deviation,
            "time_sec": round(elapsed, 4),
            "solution_sets": solution_str
        })

        # Live progress line
        sys.stdout.write(f"\rInstance {i}/{total_instances}")
        sys.stdout.flush()

    print()  # newline after loop

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # === Summary Statistics ===
    valid_devs = [r["deviation_%"] for r in results if r["deviation_%"] is not None]
    valid_times = [r["time_sec"] for r in results if r["time_sec"] is not None]
    valid_opts = [r for r in results if r["opt_value"] and r["solution_cost"] is not None]

    avg_dev = round(sum(valid_devs) / len(valid_devs), 2) if valid_devs else None
    min_dev = round(min(valid_devs), 2) if valid_devs else None
    max_dev = round(max(valid_devs), 2) if valid_devs else None

    num_opt_found = sum(1 for r in valid_opts if r["solution_cost"] == r["opt_value"])
    fastest_time = min(valid_times) if valid_times else 0
    slowest_time = max(valid_times) if valid_times else 0
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
    total_time = time.time() - start_all

    # === Print Summary ===
    print("\n==== Summary ====")
    print(f"Solver: {algorithm_name}")
    print(f"Average deviation from optimum : {avg_dev:+.2f}%")
    print(f"Minimum deviation              : {min_dev:+.2f}%")
    print(f"Maximum deviation              : {max_dev:+.2f}%")
    print(f"Optimal solutions found        : {num_opt_found}/{total_instances}")
    print(f"Time [fastest, slowest]        : {fastest_time:.2f}s  , {slowest_time:.2f}s")
    print(f"Average time per instance      : {avg_time:.2f}s")
    print(f"Total elapsed time             : {total_time:.2f}s")
    print("==================")

    return df