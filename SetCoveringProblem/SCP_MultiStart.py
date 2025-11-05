from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *
from SCP_GRASP import *

import random, time
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed


import random, time
from SCPConstructive import greedy_randomized_adaptive
from SCPLocalSearch import first_improvement_drop_or_swap_loop
from SCP_TABU import *




def hamming_distance(sol_a, sol_b):
    """
    Compute normalized Hamming distance between two SCP solutions
    (based on selected sets).
    """
    set_a = set(sol_a.selected)
    set_b = set(sol_b.selected)
    diff = set_a.symmetric_difference(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(diff) / len(union)



def grasp_top_k_solutions(instance, k=5, alpha=0.1, max_time=60, verbose=True, seed=42):
    """
    Run GRASP for a given time and keep the k best solutions found.

    Parameters
    ----------
    instance : SCPInstance
        The SCP problem instance.
    k : int
        Number of top solutions to keep.
    alpha : float
        Greediness–randomness control.
    max_time : float
        Maximum runtime in seconds.
    verbose : bool
        Print progress information.
    seed : int
        Random seed.
    """
    random.seed(seed)
    start_time = time.time()
    top_solutions = []
    opt = getattr(instance, "opt_value", None)
    iterations = 0

    if verbose:
        print(f"\n🌀 Running GRASP for top-{k} solutions | α={alpha:.2f} | time={max_time}s | instance={instance.name}")

    while time.time() - start_time < max_time:
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        sol = first_improvement_drop_or_swap_loop(sol, max_time=2.0)
        sol.prune_by_cost()

        # Add and keep only the k best
        top_solutions.append(sol.copy())
        top_solutions = sorted(top_solutions, key=lambda s: s.cost)[:k]
        iterations += 1

        if verbose and iterations % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  I\ter {iterations} | elapsed={elapsed:.1f}s | best={top_solutions[0].cost:.2f}")

    # Print summary
    print("\n🏁 Finished collecting top solutions")
    for i, s in enumerate(top_solutions):
        dev = 100 * (s.cost - opt) / opt if opt else 0
        print(f"  #{i+1}: cost={s.cost:.2f} (dev={dev:+.2f}%) | size={len(s.selected)}")

    # Compute Hamming distance matrix
    dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dist_matrix[i, j] = hamming_distance(top_solutions[i], top_solutions[j])

    print("\n🔹 Pairwise Hamming Distances:")
    for i in range(k):
        print(f"  {i+1}: " + "  ".join(f"{dist_matrix[i,j]:.2f}" for j in range(k)))

    total_time = time.time() - start_time
    print(f"\n⏱ Total runtime: {total_time:.2f}s ({iterations} iterations)")
    return top_solutions, dist_matrix

def select_diverse_elite(solutions, dist_matrix, n=5, w_d=1.0, w_c=0.5):
    """
    Select a diverse subset of elite solutions balancing cost and diversity.

    Parameters
    ----------
    solutions : list of SCPSolution
        Candidate pool.
    dist_matrix : np.ndarray
        Pairwise Hamming distances between candidates.
    n : int
        Number of solutions to select.
    w_d : float
        Weight for diversity importance.
    w_c : float
        Weight for cost penalty.

    Returns
    -------
    list of int
        Indices of selected solutions.
    """
    costs = np.array([s.cost for s in solutions])
    best_idx = np.argmin(costs)
    selected = [best_idx]
    best_cost = costs[best_idx]

    while len(selected) < n:
        scores = []
        for i in range(len(solutions)):
            if i in selected:
                scores.append(-np.inf)
                continue
            avg_dist = np.mean([dist_matrix[i, j] for j in selected])
            cost_penalty = (costs[i] - best_cost) / best_cost
            score = w_d * avg_dist - w_c * cost_penalty
            scores.append(score)
        next_idx = int(np.argmax(scores))
        selected.append(next_idx)

    return selected



def run_multistart_adaptive_grasp_tabu(instance,
                                       grasp_time=300,          # seconds for GRASP
                                       best_sols_num=2,         # how many top/diverse sols to refine
                                       tabu_time=300,           # seconds per TABU run
                                       elite_k=10,              # how many GRASP sols to collect
                                       reactive=True,
                                       # GRASP params
                                       alpha_start=0.05,
                                       alpha_max=0.3,
                                       alpha_min=0.0,
                                       alpha_factor_up=1.2,
                                       alpha_factor_down=0.8,
                                       no_improve_limit=10,
                                       # TABU params
                                       tabu_tenure_init=10,
                                       tabu_tenure_min=3,
                                       tabu_tenure_max=40,
                                       max_no_improve=100,
                                       alpha=0.9,
                                       shake_intensity=0.1,
                                       # Selection weights
                                       w_d=1.0,
                                       w_c=0.5,
                                       verbose=True):
    """
    Run Adaptive GRASP to collect diverse top solutions, then refine each with (Reactive) TABU Search.
    Example: grasp_time=300s, best_sols_num=2, tabu_time=300s → total ≈ 15 min.
    """

    t0 = time.time()
    if verbose:
        print(f"\n🚀 MultiStart GRASP+TABU | GRASP={grasp_time}s | TABU/sol={tabu_time}s | instance={instance.name}")

    # ============================================================
    # 1️⃣ Generate top elite GRASP solutions (Adaptive α)
    # ============================================================
    start = time.time()
    solutions = []
    while time.time() - start < grasp_time:
        sol = grasp_adaptive_alpha(instance,
                                   alpha_start=alpha_start,
                                   alpha_max=alpha_max,
                                   alpha_min=alpha_min,
                                   alpha_factor_up=alpha_factor_up,
                                   alpha_factor_down=alpha_factor_down,
                                   no_improve_limit=no_improve_limit,
                                   max_time=min(grasp_time - (time.time() - start), 60),
                                   verbose=False)
        solutions.append(sol)
        solutions = sorted(solutions, key=lambda s: s.cost)[:elite_k]

    # Compute pairwise Hamming distances
    k = len(solutions)
    dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            set_i, set_j = set(solutions[i].selected), set(solutions[j].selected)
            diff = set_i.symmetric_difference(set_j)
            union = set_i.union(set_j)
            dist_matrix[i, j] = len(diff) / len(union) if union else 0.0

    if verbose:
        print(f"\n🏁 Collected {len(solutions)} elite GRASP solutions.")

    # ============================================================
    # 2️⃣ Select diverse subset for TABU
    # ============================================================
    chosen_idx = select_diverse_elite(solutions, dist_matrix,
                                      n=min(best_sols_num, len(solutions)),
                                      w_d=w_d, w_c=w_c)
    if verbose:
        print("\n🏆 Selected solutions for TABU:")
        for i in chosen_idx:
            print(f"  idx={i}, cost={solutions[i].cost:.2f}")

    # ============================================================
    # 3️⃣ Run TABU Search for each selected GRASP solution
    # ============================================================
    improved = []
    for i in chosen_idx:
        init_sol = solutions[i]
        print(f"\n🔥 Running {'Reactive' if reactive else 'Standard'} TABU on sol {i} "
              f"| cost={init_sol.cost:.2f} | time={tabu_time}s")

        if reactive:
            best = reactive_tabu_search_core(instance,
                                             init_sol,
                                             tabu_tenure_init=tabu_tenure_init,
                                             tabu_tenure_min=tabu_tenure_min,
                                             tabu_tenure_max=tabu_tenure_max,
                                             max_no_improve=max_no_improve,
                                             alpha=alpha,
                                             shake_intensity=shake_intensity,
                                             max_time=tabu_time,
                                             verbose=verbose)
        else:
            best = tabu_search_core(instance,
                                    init_sol,
                                    tabu_tenure=tabu_tenure_init,
                                    shake_intensity=shake_intensity,
                                    max_time=tabu_time,
                                    verbose=verbose)
        improved.append(best)

    # ============================================================
    # 4️⃣ Final summary
    # ============================================================
    print("\n✅ Final TABU Refinement Summary:")
    opt = getattr(instance, "opt_value", None)
    for i, s in enumerate(improved):
        dev = 100 * (s.cost - opt) / opt if opt else 0
        print(f"  {i+1:2d}. cost={s.cost:.2f} (dev={dev:+.2f}%)")

    total_time = time.time() - t0
    print(f"\n⏱ Total runtime: {total_time/60:.2f} min (≈ {total_time:.1f} s)")
    best_sol = min(improved, key=lambda s: s.cost) if improved else None
    return best_sol