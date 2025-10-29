from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

import random, time

from concurrent.futures import ProcessPoolExecutor, as_completed


def grasp_sequential(instance, alpha=0.1, max_time=600.0, max_iter=50, seed=42, verbose=False):
    """
    Basic GRASP metaheuristic for the Set Covering Problem.
    Combines the randomized greedy constructor and local search.

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    alpha : float
        Greediness-randomness control (0 = greedy, 1 = random).
    max_time : float
        Time limit (seconds).
    max_iter : int
        Maximum number of GRASP iterations.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, prints progress (iteration, cost, improvements).
    """
    random.seed(seed)
    best_sol = None
    best_cost = float("inf")

    opt = instance.opt_value or None  # may be None if unknown
    start = time.time()
    iteration = 0

    if verbose:
        print(f"\n🌀 Starting GRASP | alpha={alpha:.2f}, max_iter={max_iter}, max_time={max_time:.1f}s")
        if opt:
            print(f"   Known optimum for {instance.name}: {opt:.2f}\n")

    while time.time() - start < max_time and iteration < max_iter:
        iteration += 1

        # 1️⃣ Construct randomized solution
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        # 2️⃣ Apply local search
        sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time)
        # 3️⃣ Prune redundancy
        sol.prune_by_cost()

        # Track best
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol.copy()

            if opt:
                dev = 100 * (best_cost - opt) / opt
                print(f"  Iter {iteration:3d}: ✨ New best cost = {best_cost:.2f} "
                      f"(deviation = {dev:+.2f}%)")
            elif verbose:
                print(f"  Iter {iteration:3d}: ✨ New best cost = {best_cost:.2f}")
        elif verbose:
            #print(f"  Iter {iteration:3d}: Cost = {sol.cost:.2f} (no improvement)")
            pass

    elapsed = time.time() - start
    if verbose:
        print(f"✅ GRASP finished after {iteration} iterations ({elapsed:.2f}s)")
        if opt:
            final_dev = 100 * (best_cost - opt) / opt
            print(f"   ➤ Best cost found: {best_cost:.2f} (deviation = {final_dev:+.2f}%)\n")
        else:
            print(f"   ➤ Best cost found: {best_cost:.2f}\n")

    return best_sol


def grasp_single_run(instance, alpha=0.05, max_time=30.0, seed=None):
    """
    Performs one GRASP iteration (randomized construction + local search).
    Designed for parallel execution.
    """
    if seed is not None:
        random.seed(seed)

    # 1️⃣ Construct a randomized solution
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
    # 2️⃣ Local search
    sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time)
    # 3️⃣ Redundancy elimination
    sol.prune_by_cost()

    return sol


def grasp_parallel(instance, alpha=0.05, max_time=900.0, num_iter=20, num_workers=4, seed=42):
    """
    Parallel GRASP metaheuristic for the Set Covering Problem.
    Launches multiple independent GRASP runs in parallel and returns the best solution.

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    alpha : float
        GRASP greediness/randomness control.
    max_time : float
        Total time budget (seconds).
    num_iter : int
        Total number of independent GRASP runs to perform.
    num_workers : int
        Number of parallel processes (usually ≤ CPU core count).
    seed : int
        Base random seed for reproducibility.
    """
    random.seed(seed)
    best_sol = None
    best_cost = float("inf")
    start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                grasp_single_run,
                instance,
                alpha,
                max_time,  #TODO split this time among runs
                random.randint(0, 1_000_000)
            )
            for _ in range(num_iter)
        ]
        i = 0
        for f in as_completed(futures):
            sol = f.result()
            if sol.cost < best_cost:
                #print(f" Iter {i:3d} New best cost found in parallel: {sol.cost:.2f}")
                best_cost = sol.cost
                best_sol = sol
            else:
                pass
                #print(f"  - Completed run with cost: {sol.cost:.2f}")
            i += 1

    elapsed = time.time() - start
    # Optional: print summary line
    opt = instance.opt_value
    if opt:
        dev = 100 * (best_cost - opt) / opt
        print(f"✅ GRASP Parallel [{instance.name}] "
              f"best cost={best_cost:.2f} (dev={dev:+.2f}%) "
              f"in {elapsed:.2f}s using {num_workers} workers.")
    else:
        print(f"✅ GRASP Parallel [{instance.name}] "
              f"best cost={best_cost:.2f} in {elapsed:.2f}s using {num_workers} workers.")

    return best_sol



def grasp_single_run_fixed_RCL(instance, desired_RCL=10, max_time=30.0, seed=None):
    """
    Performs one GRASP iteration (randomized construction + local search).
    Designed for parallel execution.
    """
    # Keep RCL size roughly constant across instance sizes
    alpha = min(1.0, desired_RCL / instance.n)

    if seed is not None:
        random.seed(seed)

    # 1️⃣ Construct a randomized solution
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
    # 2️⃣ Local search
    sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time)
    # 3️⃣ Redundancy elimination
    sol.prune_by_cost()

    return sol


def grasp_parallel_fixed_RCL(instance, desired_RCL=10, max_time=900.0, num_iter=20, num_workers=4, seed=42):
    """
    Parallel GRASP metaheuristic for the Set Covering Problem.
    Uses a dynamic alpha = desired_RCL / n.
    """
    random.seed(seed)
    best_sol = None
    best_cost = float("inf")
    start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                grasp_single_run_fixed_RCL,
                instance,
                desired_RCL,
                max_time / num_iter,  # optional: split time per run
                random.randint(0, 1_000_000)
            )
            for _ in range(num_iter)
        ]

        for i, f in enumerate(as_completed(futures)):
            sol = f.result()
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_sol = sol

    elapsed = time.time() - start
    opt = getattr(instance, "opt_value", None)
    if opt:
        dev = 100 * (best_cost - opt) / opt
        print(f"✅ GRASP Parallel [{instance.name}] best cost={best_cost:.2f} "
              f"(dev={dev:+.2f}%) in {elapsed:.2f}s using {num_workers} workers.")
    else:
        print(f"✅ GRASP Parallel [{instance.name}] best cost={best_cost:.2f} "
              f"in {elapsed:.2f}s using {num_workers} workers.")

    return best_sol



def grasp_single_run_BI_fixed_RCL(instance, desired_RCL=10, max_time=30.0, seed=None):
    """
    Performs one GRASP iteration (randomized construction + Best Improvement local search).
    Uses a fixed RCL size based on desired_RCL parameter.
    Designed for parallel execution.
    """
    # Compute alpha dynamically so RCL has approximately desired_RCL elements
    alpha = min(1.0, desired_RCL / instance.n)

    if seed is not None:
        random.seed(seed)

    # 1️⃣ Construct a randomized solution
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
    # 2️⃣ Apply Best Improvement local search
    sol = best_improvement_drop_or_swap_loop(sol, max_time=max_time)
    # 3️⃣ Redundancy elimination
    sol.prune_by_cost()

    return sol


def grasp_parallel_BI_fixed_RCL(instance, desired_RCL=10, max_time=900.0, num_iter=20, num_workers=4, seed=42):
    """
    Parallel GRASP (Best Improvement + Fixed RCL version) for the Set Covering Problem.

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    desired_RCL : int
        Target number of elements in the Restricted Candidate List (controls alpha adaptively).
    max_time : float
        Total runtime budget (seconds).
    num_iter : int
        Number of independent GRASP runs.
    num_workers : int
        Number of parallel processes (typically <= number of CPU cores).
    seed : int
        Base random seed for reproducibility.
    """
    random.seed(seed)
    best_sol = None
    best_cost = float("inf")
    start = time.time()

    # Split available time among all runs to avoid total overrun
    per_run_time = max_time / num_iter

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                grasp_single_run_BI_fixed_RCL,
                instance,
                desired_RCL,
                per_run_time,
                random.randint(0, 1_000_000)
            )
            for _ in range(num_iter)
        ]

        for i, f in enumerate(as_completed(futures)):
            sol = f.result()
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_sol = sol
            # Optional: print live progress
            # print(f"[{i+1}/{num_iter}] Completed with cost {sol.cost:.2f}")

    elapsed = time.time() - start
    opt = getattr(instance, "opt_value", None)
    if opt:
        dev = 100 * (best_cost - opt) / opt
        print(f"✅ GRASP-BI-FixedRCL [{instance.name}] best cost={best_cost:.2f} "
              f"(dev={dev:+.2f}%) in {elapsed:.2f}s using {num_workers} workers.")
    else:
        print(f"✅ GRASP-BI-FixedRCL [{instance.name}] best cost={best_cost:.2f} "
              f"in {elapsed:.2f}s using {num_workers} workers.")

    return best_sol

