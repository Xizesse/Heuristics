import os, time, pandas as pd, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Import project modules ---
from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *
from SCP_GRASP import *

# ============================================================
# ============= Sequential Version ===========================
# ============================================================

def solve_all_instances(algorithm_name, csv_filename, folder="SCP-Instances",
                        num_instances=0, *args, **kwargs):
    """
    Runs the given solver across all SCP instances and saves results to 'results/'.

    Works with either:
      - algorithm_name = 'string_name'
      - algorithm_name = function reference

    Produces a CSV with performance metrics and prints summary stats.
    """
    import sys

    # --- Resolve solver function robustly ---
    if callable(algorithm_name):
        solver_func = algorithm_name
        algorithm_name = solver_func.__name__
    else:
        solver_func = globals().get(algorithm_name)
        if solver_func is None:
            # search across all loaded modules
            for mod in sys.modules.values():
                if hasattr(mod, algorithm_name):
                    solver_func = getattr(mod, algorithm_name)
                    break
        if solver_func is None or not callable(solver_func):
            raise ValueError(f"Solver '{algorithm_name}' not found or not callable.")

    # --- Prepare output folder ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_filename)

    # --- Load instances ---
    files = sorted(f for f in os.listdir(folder) if f.lower().startswith("scp"))
    if num_instances > 0:
        files = files[:num_instances]
    total_instances = len(files)

    results = []
    print(f"Solver: {algorithm_name}")
    start_all = time.time()

    # --- Loop through instances sequentially ---
    for i, _ in enumerate(files, start=1):
        inst = SCPInstance(i - 1, folder=folder)
        opt = inst.opt_value if inst.opt_value is not None else 0

        t0 = time.time()
        sol = solver_func(inst, *args, **kwargs)
        elapsed = time.time() - t0

        cost = getattr(sol, "cost", None)
        selected_sets = getattr(sol, "selected", None)
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

        sys.stdout.write(f"\rInstance {i}/{total_instances}")
        sys.stdout.flush()

    print()
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # --- Summary statistics ---
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

    print("\n==== Summary ====")
    print(f"Solver: {algorithm_name}")
    print(f"Average deviation from optimum : {avg_dev:+.2f}%")
    print(f"Minimum deviation              : {min_dev:+.2f}%")
    print(f"Maximum deviation              : {max_dev:+.2f}%")
    print(f"Optimal solutions found        : {num_opt_found}/{total_instances}")
    print(f"Time [fastest, slowest]        : {fastest_time:.2f}s , {slowest_time:.2f}s")
    print(f"Average time per instance      : {avg_time:.2f}s")
    print(f"Total elapsed time             : {total_time:.2f}s")
    print("==================")

    return df


# ============================================================
# ============= Parallel Version =============================
# ============================================================

def _run_single_instance(i, filename, folder, algorithm_name, solver_func, *args, **kwargs):
    """Executed in parallel worker process."""
    inst = SCPInstance(i, folder=folder)
    opt = inst.opt_value if inst.opt_value is not None else 0

    t0 = time.time()
    sol = solver_func(inst, *args, **kwargs)
    elapsed = time.time() - t0

    cost = getattr(sol, "cost", None)
    deviation = round(100 * (cost - opt) / opt, 2) if opt and cost is not None else None
    selected_sets = getattr(sol, "selected", [])
    if isinstance(selected_sets, (set, list)):
        solution_str = ",".join(map(str, sorted(selected_sets)))
    else:
        solution_str = str(selected_sets)

    return {
        "instance_name": inst.name,
        "opt_value": opt,
        "solver": algorithm_name,
        "solution_cost": cost,
        "deviation_%": deviation,
        "time_sec": round(elapsed, 4),
        "solution_sets": solution_str,
    }

# ============================================================
# ============= Parallel Version =============================
# ============================================================

def solve_all_instances_parallel(algorithm_name, csv_filename, folder="SCP-Instances",
                                 num_instances=0, num_workers=4, *args, **kwargs):
    """
    Parallel version of solve_all_instances.
    Runs each SCP instance in a separate process using ProcessPoolExecutor.

    Parameters
    ----------
    algorithm_name : str or callable
        The solver function or its name.
    csv_filename : str
        Output CSV file name (saved to /results).
    folder : str
        Folder containing SCP instances.
    num_instances : int
        Limit number of instances (0 = all).
    num_workers : int
        Number of parallel worker processes.
    *args, **kwargs :
        Passed directly to the solver function.
    """
    import sys

    # --- Resolve solver function robustly ---
    if callable(algorithm_name):
        solver_func = algorithm_name
        algorithm_name = solver_func.__name__
    else:
        solver_func = globals().get(algorithm_name)
        if solver_func is None:
            for mod in sys.modules.values():
                if hasattr(mod, algorithm_name):
                    solver_func = getattr(mod, algorithm_name)
                    break
        if solver_func is None or not callable(solver_func):
            raise ValueError(f"Solver '{algorithm_name}' not found or not callable.")

    # --- Prepare output folder ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_filename)

    # --- Load instances ---
    files = sorted(f for f in os.listdir(folder) if f.lower().startswith("scp"))
    if num_instances > 0:
        files = files[:num_instances]
    total_instances = len(files)

    print(f"Solver (parallel): {algorithm_name} using {num_workers} workers")
    start_all = time.time()

    # --- Run instances in parallel ---
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _run_single_instance,
                i,
                filename,
                folder,
                algorithm_name,
                solver_func,
                *args,
                **kwargs,
            )
            for i, filename in enumerate(files)
        ]
        for k, f in enumerate(as_completed(futures), start=1):
            try:
                res = f.result()
                results.append(res)
                sys.stdout.write(f"\rCompleted {k}/{total_instances}")
                sys.stdout.flush()
            except Exception as e:
                print(f"\n⚠️ Error in instance {futures[f]}: {e}")

    print()
    elapsed_total = time.time() - start_all

    # --- Save results to CSV ---
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # --- Compute summary stats ---
        # --- Compute summary stats (same format as sequential) ---
    valid_results = [r for r in results if "error" not in r]
    valid_devs = [r["deviation_%"] for r in valid_results if r["deviation_%"] is not None]
    valid_times = [r["time_sec"] for r in valid_results if r["time_sec"] is not None]
    valid_opts = [r for r in valid_results if r["opt_value"] and r["solution_cost"] is not None]

    avg_dev = round(sum(valid_devs) / len(valid_devs), 2) if valid_devs else None
    min_dev = round(min(valid_devs), 2) if valid_devs else None
    max_dev = round(max(valid_devs), 2) if valid_devs else None
    num_opt_found = sum(1 for r in valid_opts if r["solution_cost"] == r["opt_value"])
    fastest_time = min(valid_times) if valid_times else 0
    slowest_time = max(valid_times) if valid_times else 0
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
    total_time = time.time() - start_all

    print("\n==== Parallel Summary ====")
    print(f"Solver: {algorithm_name}")
    print(f"Average deviation from optimum : {avg_dev:+.2f}%" if avg_dev is not None else "N/A")
    print(f"Minimum deviation              : {min_dev:+.2f}%" if min_dev is not None else "N/A")
    print(f"Maximum deviation              : {max_dev:+.2f}%" if max_dev is not None else "N/A")
    print(f"Optimal solutions found        : {num_opt_found}/{total_instances}")
    print(f"Time [fastest, slowest]        : {fastest_time:.2f}s , {slowest_time:.2f}s")
    print(f"Average time per instance      : {avg_time:.2f}s")
    print(f"Total elapsed time             : {total_time:.2f}s using {num_workers} workers")
    print("============================")



    return df