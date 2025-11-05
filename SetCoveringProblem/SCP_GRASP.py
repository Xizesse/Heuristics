from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

import random, time

from concurrent.futures import ProcessPoolExecutor, as_completed


import random, time
from SCPConstructive import greedy_randomized_adaptive
from SCPLocalSearch import first_improvement_drop_or_swap_loop




def grasp_sequential(instance,
                   alpha=0.1,
                   max_time=600.0,
                   verbose=True,
                   seed=42):

    """
    Time-based sequential GRASP for the Set Covering Problem.

    Keeps sampling randomized greedy solutions + local search
    until the time limit (max_time) is reached.

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    alpha : float
        Greediness-randomness control (0 = greedy, 1 = random).
    max_time : float
        Total runtime budget in seconds.
    seed : int
        Random seed.
    verbose : bool
        If True, prints best improvement messages.
    """
    random.seed(seed)
    start = time.time()
    best_sol, best_cost = None, float("inf")

    opt = getattr(instance, "opt_value", None)
    iteration = 0

    if verbose:
        print(f"\n🌀 Starting GRASP (time-based) | α={alpha:.2f} | "
              f"max_time={max_time:.1f}s | instance={instance.name}")

    # --- main loop until time limit ---
    while time.time() - start < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start)

        # 1️⃣ Construct randomized solution
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        # 2️⃣ Local search (use remaining time so it doesn’t overrun)
        sol = first_improvement_drop_or_swap_loop(sol, max_time=remaining)
        sol.prune_by_cost()

        # 3️⃣ Update best
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol.copy()

            if opt and opt > 0:
                dev = 100 * (best_cost - opt) / opt
                if verbose:
                    print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f} (dev={dev:+.2f}%)")
            elif verbose:
                print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f}")

    elapsed = time.time() - start
    if verbose:
        if opt and opt > 0:
            dev = 100 * (best_cost - opt) / opt
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} (dev={dev:+.2f}%)")
        else:
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f}")

    return best_sol



from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

import random, time

from concurrent.futures import ProcessPoolExecutor, as_completed


import random, time
from SCPConstructive import greedy_randomized_adaptive
from SCPLocalSearch import first_improvement_drop_or_swap_loop




def grasp_sequential(instance,
                    alpha=0.1,
                   max_time=600.0,

                   verbose=True,
                   seed=42):

    """
    Time-based sequential GRASP for the Set Covering Problem.

    Keeps sampling randomized greedy solutions + local search
    until the time limit (max_time) is reached.

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    alpha : float
        Greediness-randomness control (0 = greedy, 1 = random).
    max_time : float
        Total runtime budget in seconds.
    seed : int
        Random seed.
    verbose : bool
        If True, prints best improvement messages.
    """
    random.seed(seed)
    start = time.time()
    best_sol, best_cost = None, float("inf")

    opt = getattr(instance, "opt_value", None)
    iteration = 0

    if verbose:
        print(f"\n🌀 Starting GRASP (time-based) | α={alpha:.2f} | "
              f"max_time={max_time:.1f}s | instance={instance.name}")

    # --- main loop until time limit ---
    while time.time() - start < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start)

        # 1️⃣ Construct randomized solution
        sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=random.randint(0, 1_000_000))
        # 2️⃣ Local search (use remaining time so it doesn’t overrun)
        sol = first_improvement_drop_or_swap_loop(sol, max_time=remaining)
        sol.prune_by_cost()

        # 3️⃣ Update best
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol.copy()

            if opt and opt > 0:
                dev = 100 * (best_cost - opt) / opt
                if verbose:
                    print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f} (dev={dev:+.2f}%)")
            elif verbose:
                print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f}")

    elapsed = time.time() - start
    if verbose:
        if opt and opt > 0:
            dev = 100 * (best_cost - opt) / opt
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} (dev={dev:+.2f}%)")
        else:
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f}")

    return best_sol


import random, time
from SCPConstructive import greedy_randomized_adaptive
from SCPLocalSearch import first_improvement_drop_or_swap_loop


def grasp_normalized_alpha(instance,
                           alpha_base=0.01,
                           m_ref=200,
                           exponent=2.0,
                           alpha_min=0.0001,
                           alpha_max=0.3,
                           max_time=600.0,
                           verbose=True,
                           seed=42):
    """
    GRASP with normalized α scaling (based on the number of attributes m)
    for the Set Covering Problem (SCP).

    The effective α is scaled inversely with the instance size (m)
    according to a power law:

        α_eff = clip( α_base * (m_ref / m)^exponent, α_min, α_max )

    This ensures that smaller instances keep the same α behavior as the
    reference case (m_ref), while larger instances automatically become
    greedier (less random).

    Parameters
    ----------
    instance : SCPInstance
        Problem instance.
    alpha_base : float
        Base α value for the reference instance (m = m_ref).
    m_ref : int
        Reference number of attributes (e.g., 200).
    exponent : float
        Power-law exponent controlling how quickly α shrinks with size.
        - 1.0 → linear inverse scaling
        - 2.0 → quadratic inverse scaling (default)
    alpha_min, alpha_max : float
        Minimum and maximum allowed α values.
    max_time : float
        Total runtime limit (seconds).
    verbose : bool
        Print progress if True.
    seed : int
        Random seed.
    """

    random.seed(seed)
    start = time.time()
    best_sol, best_cost = None, float("inf")
    opt = getattr(instance, "opt_value", None)

    # --- Normalized alpha based on number of attributes ---
    m = instance.m
    alpha_eff = alpha_base * (m_ref / m) ** exponent
    alpha_eff = max(alpha_min, min(alpha_max, alpha_eff))

    if verbose:
        print(f"\n🌀 Starting GRASP (normalized α) | base={alpha_base:.4f}, "
              f"exponent={exponent:.2f}, m={m}, α_eff={alpha_eff:.6f}, "
              f"time={max_time:.0f}s | instance={instance.name}")

    iteration = 0
    while time.time() - start < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start)

        # 1️⃣ Construct randomized solution
        sol = greedy_randomized_adaptive(instance, alpha=alpha_eff, seed=random.randint(0, 1_000_000))

        # 2️⃣ Local search refinement
        sol = first_improvement_drop_or_swap_loop(sol, max_time=remaining)
        sol.prune_by_cost()

        # 3️⃣ Update best
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol.copy()
            if verbose:
                if opt and opt > 0:
                    dev = 100 * (best_cost - opt) / opt
                    print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f} "
                          f"(dev={dev:+.2f}%) α={alpha_eff:.6f}")
                else:
                    print(f"✨ Iter {iteration:3d}: New best {best_cost:.2f}")

    elapsed = time.time() - start
    if verbose:
        if opt and opt > 0:
            dev = 100 * (best_cost - opt) / opt
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} "
                  f"(dev={dev:+.2f}%) | α_eff={alpha_eff:.6f}")
        else:
            print(f"✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} | α_eff={alpha_eff:.6f}")

    return best_sol


def grasp_adaptive_alpha(instance,
                         alpha_start=0.05,
                         alpha_max=0.3,
                         alpha_min=0.0,
                         alpha_factor_up=1.2,
                         alpha_factor_down=0.8,
                         max_time=600.0,
                         no_improve_limit=10,
                         seed=42,
                         verbose=True):
    """
    GRASP with reactive adaptive α for the Set Covering Problem.

    - α decreases (more greedy) when improvement occurs
    - α increases (more random) when stagnating
    - Fixed per-iteration local search time to allow many restarts

    Parameters
    ----------
    instance : SCPInstance
    alpha_start : float
        Initial alpha value (controls greediness/randomness).
    alpha_max : float
        Maximum allowed alpha (full randomness).
    alpha_min : float
        Minimum allowed alpha (fully greedy).
    alpha_factor_up : float
        Multiplicative factor to increase alpha after stagnation.
    alpha_factor_down : float
        Multiplicative factor to decrease alpha after improvement.
    max_time : float
        Total time budget in seconds.
    no_improve_limit : int
        Number of iterations without improvement before increasing alpha.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, prints progress.
    """

    random.seed(seed)
    start = time.time()
    best_sol, best_cost = None, float("inf")
    opt = getattr(instance, "opt_value", None)

    alpha = alpha_start
    iteration = 0
    no_improve = 0

    if verbose:
        print(f"\n🌀 Starting GRASP (adaptive α) | α₀={alpha_start:.5f} "
              f"→ [{alpha_min:.5f}, {alpha_max:.5f}] | time={max_time:.1f}s | instance={instance.name}")

    # --- main loop ---
    while time.time() - start < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start)

        # small adaptive random variation to α
        alpha_iter = max(alpha_min, min(alpha_max, alpha * random.uniform(0.9, 1.1)))

        # --- Construct + Local Search ---
        sol = greedy_randomized_adaptive(instance, alpha=alpha_iter, seed=random.randint(0, 1_000_000))
        sol = first_improvement_drop_or_swap_loop(sol, max_time=max_time / 25)
        sol.prune_by_cost()

        # --- Check for improvement ---
        if sol.cost < best_cost:
            best_cost = sol.cost
            best_sol = sol.copy()
            no_improve = 0
            alpha = max(alpha_min, alpha * alpha_factor_down)  # go greedier
            if verbose:
                if opt and opt > 0:
                    dev = 100 * (best_cost - opt) / opt
                    print(f"✨ Iter {iteration:3d}: NEW BEST {best_cost:.2f} (dev={dev:+.2f}%) α={alpha_iter:.3f}")
                else:
                    print(f"✨ Iter {iteration:3d}: NEW BEST {best_cost:.2f} α={alpha_iter:.5f}")
        else:
            no_improve += 1
            if no_improve >= no_improve_limit:
                old_alpha = alpha
                alpha = min(alpha * alpha_factor_up, alpha_max)
                no_improve = 0
                if verbose:
                    print(f"↗️  Stagnation → α {old_alpha:.5f} → {alpha:.5f}")

    # --- summary ---
    elapsed = time.time() - start
    if verbose:
        if opt and opt > 0:
            dev = 100 * (best_cost - opt) / opt
            print(f"\n✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} (dev={dev:+.2f}%) | α_final={alpha:.3f}")
        else:
            print(f"\n✅ Finished after {elapsed:.1f}s | best={best_cost:.2f} | α_final={alpha:.3f}")

    return best_sol