from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

import random, time


def tabu_neighborhood(sol, instance, tabu_dict, iteration, tabu_tenure=7):
    """
    Explore admissible (non-tabu or aspirational) drop (1x0) and swap (1x1) moves.
    Return (best_neighbor_solution, best_move).
    """
    n = instance.n
    best_candidate, best_move = None, None
    best_cost = float("inf")

    selected = list(sol.selected)
    non_selected = [t for t in range(n) if t not in sol.selected]

    for s in selected:
        sol.remove(s)

        # --- 1x0 drop move ---
        if sol.is_feasible():
            move = (s, None)
            tabu_active = (move in tabu_dict and tabu_dict[move] > iteration)
            # aspiration: allow tabu if improves global best (stored in tabu_dict["best_cost"])
            if (not tabu_active or sol.cost < tabu_dict.get("best_cost", float("inf"))) and sol.cost < best_cost:
                best_candidate, best_move, best_cost = sol.copy(), move, sol.cost

        # --- 1x1 swap moves ---
        for t in non_selected:
            sol.add(t)
            if sol.is_feasible():
                move = (s, t)
                tabu_active = (move in tabu_dict and tabu_dict[move] > iteration)
                if (not tabu_active or sol.cost < tabu_dict.get("best_cost", float("inf"))) and sol.cost < best_cost:
                    best_candidate, best_move, best_cost = sol.copy(), move, sol.cost
            sol.remove(t)

        sol.add(s)

    return best_candidate, best_move


def tabu(instance, max_time=60.0, tabu_tenure=7, max_no_improve=100):
    """
    Tabu Search for SCP using move-based tabu memory.
    All reporting is in deviation (%) relative to instance.opt_value, clamped to >= 0.
    """
    def deviation(cost, opt):
        # clamp to 0 to avoid negative printed values
        if opt is None or opt <= 0:
            return 0.0
        return max(0.0, 100.0 * (cost - opt) / opt)

    start_time = time.time()

    # --- initial solution (you can swap this constructor if you prefer) ---
    current = greedy_cost_efficiency(instance)
    current.prune_by_cost()
    best = current.copy()

    # reference optimum for deviations
    opt_cost = getattr(instance, "opt_value", None)
    # if not provided, fallback to current cost to avoid division by zero
    if opt_cost is None or opt_cost <= 0:
        opt_cost = best.cost

    # bookkeeping
    iteration = 0
    no_improve = 0
    tabu_dict = {}
    tabu_dict["best_cost"] = best.cost  # for aspiration checks

    # --- initial report ---
    print(f"▶️ Start Tabu — init cost: {current.cost:.2f} "
          f"(dev={deviation(current.cost, opt_cost):.2f}% vs opt={opt_cost:.2f})")

    # --- main loop ---
    while (time.time() - start_time < max_time) and (no_improve < max_no_improve):
        iteration += 1

        # explore neighborhood w/ tabu control
        candidate, move = tabu_neighborhood(current, instance, tabu_dict, iteration, tabu_tenure)

        if candidate is None:
            print("⚠️ No admissible move, stopping.")
            break

        # apply best admissible move
        current = candidate.copy()

        # update tabu list for the executed move
        tabu_dict[move] = iteration + tabu_tenure
        # remove expired entries
        tabu_dict = {m: e for m, e in tabu_dict.items() if (m == "best_cost") or (e > iteration)}
        # keep aspiration reference up to date
        tabu_dict["best_cost"] = best.cost

        # update global best
        improved = False
        if current.cost < best.cost:
            best = current.copy()
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        # progress print (every 10 iters or when improved)
        if improved or (iteration % 10 == 0):
            dev_curr = deviation(current.cost, opt_cost)
            dev_best = deviation(best.cost, opt_cost)
            print(f"[Iter {iteration:3d}] "
                  f"Best={best.cost:.2f} (dev={dev_best:.2f}%), "
                  f"Curr={current.cost:.2f} (dev={dev_curr:.2f}%), "
                  f"NI={no_improve}")

    # --- final report ---
    print(f"✅ Done after {iteration} iters. "
          f"Best={best.cost:.2f} (dev={deviation(best.cost, opt_cost):.2f}% vs opt={opt_cost:.2f})")
    return best
