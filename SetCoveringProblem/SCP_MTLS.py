# ============================================================
# SCP_MTLS.py
# Multi-Trajectory Local Search (Diversity + Quality version)
# ============================================================

import time, random
from SCPConstructive import greedy_randomized_adaptive
from SCPLocalSearch import first_improvement_drop_or_swap_loop
from SCPDefinitions import SCPSolution
from SCP_TABU import *


# ------------------------------------------------------------
# Helper: Hamming Distance
# ------------------------------------------------------------
def hamming_distance(setA, setB):
    """Hamming distance between two set-based solutions."""
    return len(setA.symmetric_difference(setB))


# ------------------------------------------------------------
# Core MTLS (Diversity + Quality Filtering)
# ------------------------------------------------------------
def mtls(instance,
         n_trajectories=10,
         alpha=0.1,
         distance_thresh=0.05,
         max_dev=20.0,       # ✅ New: maximum allowed deviation (%)
         max_time=600.0,
         seed=42,
         verbose=True):
    """
    Multi-Trajectory Local Search (MTLS)
    ------------------------------------
    1️⃣ Generate randomized feasible solutions (via GRASP constructor)
    2️⃣ Improve each with local search (First Improvement)
    3️⃣ Keep only sufficiently diverse and good-quality solutions

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    n_trajectories : int
        Target population size (number of diverse trajectories).
    alpha : float
        GRASP alpha (controls greediness/randomness).
    distance_thresh : float
        Minimum normalized Hamming distance to keep a new solution.
    max_dev : float
        Maximum allowed deviation (%) from the known optimum (if available).
    max_time : float
        Total time budget in seconds.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress details.
    """
    random.seed(seed)
    start = time.time()
    n = instance.n
    opt = getattr(instance, "opt_value", None)

    # --- Initialization ---
    population = []
    iteration = 0
    best_sol, best_cost = None, float("inf")

    if verbose:
        print(f"\n🌐 Starting MTLS | n_trajectories={n_trajectories} "
              f"| α={alpha} | max_dev={max_dev}% | t_max={max_time:.1f}s")

    # --- Main generation loop ---
    while time.time() - start < max_time and len(population) < n_trajectories:
        iteration += 1

        # 1️⃣ Construct randomized solution
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))

        # 2️⃣ Local search improvement
        #sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time / 50)
        sol.prune_by_cost()

        # 3️⃣ Evaluate quality
        if opt and opt > 0:
            dev = 100 * (sol.cost - opt) / opt
            if dev > max_dev:
                if verbose:
                    print(f"[{iteration:3d}] ❌ Rejected (cost dev {dev:.1f}% > {max_dev:.1f}%)")
                continue
        else:
            dev = None

        # 4️⃣ Compute diversity
        is_similar = False
        min_dist = 1.0
        for s2 in population:
            dist = hamming_distance(sol.selected, s2.selected) / n
            if dist < min_dist:
                min_dist = dist
            if dist < distance_thresh:
                is_similar = True
                break

        # 5️⃣ Accept or reject
        if not is_similar:
            population.append(sol)
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_sol = sol.copy()
            if verbose:
                print(f"[{iteration:3d}] ✅ Added sol | cost={sol.cost:.2f} "
                      f"(dev={dev:+.2f}%) | pop={len(population)} | min_dist={min_dist:.3f}")
        else:
            if verbose:
                print(f"[{iteration:3d}] ⚠️ Discarded (too similar) "
                      f"| min_dist={min_dist:.3f} < thresh={distance_thresh:.3f}")

    # --- Summary & Statistics ---
    elapsed = time.time() - start
    if population:
        best_sol = min(population, key=lambda s: s.cost)
        best_cost = best_sol.cost

    if verbose:
        if opt:
            dev = 100 * (best_cost - opt) / opt
            print(f"\n✅ MTLS finished in {elapsed:.1f}s | "
                  f"best={best_cost:.2f} (dev={dev:+.2f}%) | "
                  f"population={len(population)}")
        else:
            print(f"\n✅ MTLS finished in {elapsed:.1f}s | best={best_cost:.2f} | "
                  f"population={len(population)}")

        # Diversity stats
        if len(population) > 1:
            pairwise = [
                hamming_distance(a.selected, b.selected) / n
                for i, a in enumerate(population)
                for b in population[i+1:]
            ]
            avg_div = sum(pairwise) / len(pairwise)
            min_div = min(pairwise)
            max_div = max(pairwise)
            print(f"📊 Diversity stats | avg={avg_div:.3f} | min={min_div:.3f} | max={max_div:.3f}")
        else:
            print("📊 Only one solution in population — no diversity stats.")

    return best_sol, population


def mtls_tabu_search(instance,
                     n_trajectories=10,
                     alpha_start=0.0,           # 👈 start greedy
                     alpha_increase=1.25,       # 👈 multiply alpha when stuck
                     alpha_max=1.0,             # 👈 cap alpha
                     n_similar_max=5,           # 👈 how many similar rejections before alpha increases
                     distance_thresh=0.05,
                     max_dev=20.0,
                     ls_time_frac=0.05,
                     tabu_time_frac=0.1,
                     max_time=600.0,
                     tabu_tenure=10,
                     seed=42,
                     verbose=True):
    """
    MTLS + Tabu Search hybrid with adaptive alpha.
    Increases GRASP alpha (randomness) when too many consecutive solutions are rejected
    for being too similar.
    """
    random.seed(seed)
    start = time.time()
    n = instance.n
    opt = getattr(instance, "opt_value", None)

    population = []
    best_sol, best_cost = None, float("inf")

    # Adaptive alpha tracking
    alpha = alpha_start
    similar_count = 0

    if verbose:
        print(f"\n🚀 Starting MTLS+Tabu (Adaptive α) | start α={alpha_start:.2f} | "
              f"max α={alpha_max:.2f} | t_max={max_time:.1f}s")

    # --------------------------------------------------------
    # 1️⃣ Generate initial diverse population
    # --------------------------------------------------------
    while time.time() - start < max_time * 0.5 and len(population) < n_trajectories:
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        #sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time * ls_time_frac)
        sol.prune_by_cost()

        # Quality filter
        if opt and opt > 0:
            dev = 100 * (sol.cost - opt) / opt
            if dev > max_dev:
                if verbose:
                    print(f"❌ Rejected (dev={dev:.1f}% > {max_dev:.1f}%)")
                continue

        # Diversity check
        is_similar = False
        min_dist = 1.0
        for s2 in population:
            dist = hamming_distance(sol.selected, s2.selected) / n
            if dist < min_dist:
                min_dist = dist
            if dist < distance_thresh:
                is_similar = True
                break

        if not is_similar:
            population.append(sol)
            similar_count = 0  # reset streak
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_sol = sol.copy()
            if verbose:
                print(f"✅ Added sol | α={alpha:.2f} | cost={sol.cost:.2f} "
                      f"| pop={len(population)} | min_dist={min_dist:.3f}")
        else:
            similar_count += 1
            if verbose:
                print(f"⚠️ Similar (#{similar_count}) | α={alpha:.4f} | dist={min_dist:.3f}")
            # 🔁 Adaptive α increase when too many similar
            if similar_count >= n_similar_max:
                old_alpha = alpha
                alpha = min(alpha * alpha_increase, alpha_max)
                similar_count = 0
                if verbose:
                    print(f"🔄 Increased α: {old_alpha:.4f} → {alpha:.4f} (diversify search)")

    if not population:
        print("❌ No valid initial solutions found.")
        return None, []

    # --------------------------------------------------------
    # 2️⃣ Tabu Search refinement
    # --------------------------------------------------------
    improved_pop = []
    tabu_time_each = max_time * tabu_time_frac

    for i, sol in enumerate(population, 1):
        if verbose:
            print(f"\n🧩 Trajectory {i}/{len(population)} — Tabu Search phase")
        new_sol = reactive_tabu_search_core(
            instance,
            init_sol=sol,
            max_time=tabu_time_each,
            verbose=False
        )
        new_sol.prune_by_cost()
        improved_pop.append(new_sol)
        if new_sol.cost < best_cost:
            best_cost = new_sol.cost
            best_sol = new_sol.copy()

        if verbose:
            print(f"   Finished Tabu | cost={new_sol.cost:.2f}")

    # --------------------------------------------------------
    # 3️⃣ Post-Tabu Diversity Analysis
    # --------------------------------------------------------
    print("\n📏 Post-Tabu Diversity Analysis")
    for i, a in enumerate(improved_pop):
        for j, b in enumerate(improved_pop):
            if j <= i: 
                continue
            dist = hamming_distance(a.selected, b.selected) / n
            if dist < distance_thresh:
                print(f"⚠️ Solutions {i+1} & {j+1} are too similar (dist={dist:.3f})")

    if len(improved_pop) > 1:
        pairwise = [
            hamming_distance(a.selected, b.selected) / n
            for i, a in enumerate(improved_pop)
            for b in improved_pop[i+1:]
        ]
        print(f"📊 Diversity summary | avg={sum(pairwise)/len(pairwise):.3f} "
              f"| min={min(pairwise):.3f} | max={max(pairwise):.3f}")

    elapsed = time.time() - start
    if opt:
        dev_best = 100 * (best_cost - opt) / opt
        print(f"\n✅ Completed MTLS+Tabu in {elapsed:.1f}s | "
              f"best={best_cost:.2f} (dev={dev_best:+.2f}%)")
    else:
        print(f"\n✅ Completed MTLS+Tabu in {elapsed:.1f}s | best={best_cost:.2f}")

    return best_sol, improved_pop

def hamming_distance(setA, setB):
    """Hamming distance between two set-based solutions."""
    return len(setA.symmetric_difference(setB))

def grasp_diverse_sampler(instance,
                          n_solutions=10,
                          alpha_min=0.0,
                          alpha_max=0.3,
                          max_time=300.0,
                          target_div=0.10,
                          seed=42,
                          verbose=True):
    """
    GRASP-based sampler that adaptively adjusts alpha (randomness)
    based on diversity of the generated solutions.
    Returns a set of mutually diverse, high-quality solutions.
    """
    random.seed(seed)
    start = time.time()
    n = instance.n
    opt = getattr(instance, "opt_value", None)
    candidates = []
    alpha = alpha_min

    if verbose:
        print(f"\n🎯 GRASP Diverse Sampler (Adaptive α) | "
              f"α₀={alpha_min:.2f}→{alpha_max:.2f} | "
              f"target_div={target_div:.2f} | t_max={max_time:.1f}s | instance={instance.name}")

    iter_count = 0
    while time.time() - start < max_time:
        iter_count += 1

        # --- GRASP construction + local search ---
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time / 100)
        sol.prune_by_cost()
        candidates.append(sol)

        # --- Adaptive α adjustment based on diversity ---
        if len(candidates) > 4:
            # measure average diversity of last few solutions
            recent = candidates[-5:]
            pairwise = [
                hamming_distance(a.selected, b.selected) / n
                for i, a in enumerate(recent)
                for b in recent[i+1:]
            ]
            avg_div = sum(pairwise) / len(pairwise)

            # adapt alpha up/down depending on diversity level
            if avg_div < target_div:
                alpha = min(alpha * 1.2 + 0.01, alpha_max)
            else:
                alpha = max(alpha * 0.9 - 0.01, alpha_min)

            if verbose and iter_count % 5 == 0:
                print(f"🔄 α adapted → {alpha:.3f} (avg_div={avg_div:.3f})")

        if verbose and iter_count % 10 == 0:
            print(f"   {iter_count:3d} solutions generated...")

    if not candidates:
        print("⚠️ No feasible solutions generated.")
        return None, []

    # --- Sort by cost ---
    candidates.sort(key=lambda s: s.cost)

    # --- Greedy selection of diverse subset ---
    diverse = [candidates[0]]
    while len(diverse) < n_solutions and len(diverse) < len(candidates):
        max_min_dist, next_sol = -1, None
        for sol in candidates:
            if sol in diverse:
                continue
            dists = [hamming_distance(sol.selected, s.selected) / n for s in diverse]
            min_dist_to_all = min(dists)
            if min_dist_to_all > max_min_dist:
                max_min_dist, next_sol = min_dist_to_all, sol
        if next_sol is None:
            break
        diverse.append(next_sol)
        if verbose:
            print(f"✅ Added sol #{len(diverse):02d} | cost={next_sol.cost:.2f} | "
                  f"min_dist={max_min_dist:.3f}")

    # --- Diversity report ---
    if verbose and len(diverse) > 1:
        pairwise = [
            hamming_distance(a.selected, b.selected) / n
            for i, a in enumerate(diverse)
            for b in diverse[i+1:]
        ]
        avg_div = sum(pairwise) / len(pairwise)
        min_div = min(pairwise)
        max_div = max(pairwise)
        print(f"\n📊 Final Diversity Stats | avg={avg_div:.3f} | "
              f"min={min_div:.3f} | max={max_div:.3f}")

        print("\n📏 Pairwise Hamming Distances:")
        header = "      " + "  ".join([f"S{i+1:02d}" for i in range(len(diverse))])
        print(header)
        for i, a in enumerate(diverse):
            row = [f"S{i+1:02d}"]
            for j, b in enumerate(diverse):
                if j < i:
                    row.append("    ")
                else:
                    dist = hamming_distance(a.selected, b.selected) / n
                    row.append(f"{dist:5.3f}")
            print("  ".join(row))
    else:
        print("📊 Only one diverse solution found — no diversity stats.")

    best_sol = min(diverse, key=lambda s: s.cost)
    elapsed = time.time() - start
    if verbose:
        if opt:
            dev = 100 * (best_sol.cost - opt) / opt
            print(f"\n🏁 Completed after {elapsed:.1f}s | "
                  f"best={best_sol.cost:.2f} (dev={dev:+.2f}%) | "
                  f"diverse_count={len(diverse)}")
        else:
            print(f"\n🏁 Completed after {elapsed:.1f}s | "
                  f"best={best_sol.cost:.2f} | "
                  f"diverse_count={len(diverse)}")

    return best_sol, diverse

import time
from SCP_TABU import reactive_tabu_search_core

def diverse_tabu_intensification(instance,
                                 diverse_solutions,
                                 tabu_time_each=60.0,
                                 tabu_tenure=10,
                                 verbose=True):
    """
    Runs Tabu Search independently on each solution from a diverse population.
    Returns the best found solution and the improved population.
    """
    start = time.time()
    best_sol, best_cost = None, float("inf")
    improved_pop = []

    if verbose:
        print(f"\n🚀 Starting Tabu Intensification | {len(diverse_solutions)} initial solutions | "
              f"t_each={tabu_time_each:.1f}s | tenure={tabu_tenure}")

    for i, sol in enumerate(diverse_solutions, 1):
        if verbose:
            print(f"\n🧩 Trajectory {i}/{len(diverse_solutions)} — running Tabu Search...")

        # Run a focused tabu search from this starting point
        new_sol = reactive_tabu_search_core(
            instance,
            init_sol=sol,
            max_time=tabu_time_each,
            tabu_tenure_init=tabu_tenure,
            tabu_tenure_min=3,
            tabu_tenure_max=40,
            verbose=False
        )
        new_sol.prune_by_cost()
        improved_pop.append(new_sol)

        if new_sol.cost < best_cost:
            best_cost = new_sol.cost
            best_sol = new_sol.copy()

        if verbose:
            print(f"   ✅ Finished Tabu | cost={new_sol.cost:.2f}")

    elapsed = time.time() - start
    if verbose:
        print(f"\n🏁 Tabu Intensification done in {elapsed:.1f}s | "
              f"best={best_cost:.2f} | improved_pop={len(improved_pop)}")

    return best_sol, improved_pop

