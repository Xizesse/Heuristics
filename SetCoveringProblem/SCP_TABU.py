import time, random
from SCPDefinitions import *
from SCPConstructive import *
from SCPLocalSearch import *

# -----------------------------------------------------------
# Helper: format move for logging
# -----------------------------------------------------------
def _format_move(move):
    if not move: return "None"
    t = move[0]
    if t == "swap": return f"swap {move[1]}→{move[2]}"
    if t == "remove": return f"remove {move[1]}"
    if t == "add": return f"add {move[1]}"
    return str(move)


# -----------------------------------------------------------
# Helper: apply move + update tabu list
# -----------------------------------------------------------
def _apply_move(current, move, tabu_list, iteration, tabu_tenure):
    """Applies move and updates tabu list (in place)."""
    if move[0] == "remove":
        current.remove(move[1])
    elif move[0] == "add":
        current.add(move[1])
    elif move[0] == "swap":
        current.remove(move[1])
        current.add(move[2])

    # update tabu
    for idx in move[1:]:
        if idx is not None:
            tabu_list[idx] = iteration + tabu_tenure

    # cleanup expired entries
    tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
    return current.copy(), tabu_list

def _shake_solution(sol, intensity=0.1, verbose=False):
    """
    Diversification: randomly remove a % of sets, then greedily repair.
    intensity ∈ (0,1) = fraction of sets to remove.
    """
    instance = sol.instance
    new_sol = sol.copy()

    num_remove = max(1, int(len(new_sol.selected) * intensity))
    to_remove = random.sample(list(new_sol.selected), num_remove)

    for j in to_remove:
        new_sol.remove(j)

    if verbose:
        print(f"🔀 Shake: removed {num_remove} sets → repairing...")

    # Greedy repair: add cheapest covering sets until feasible again
    while not new_sol.is_feasible():
        uncovered = new_sol.uncovered_attributes()
        best_j, best_score = None, float("inf")
        for j in range(instance.n):
            if j in new_sol.selected:
                continue
            new_cover = len(instance.attr_of_set[j].intersection(uncovered))
            if new_cover == 0:
                continue
            score = instance.costs[j] / new_cover
            if score < best_score:
                best_score, best_j = score, j
        if best_j is None:
            break
        new_sol.add(best_j)

    new_sol.prune_by_cost()
    return new_sol



# -----------------------------------------------------------
# Helper: evaluate all neighbors and return best admissible
# -----------------------------------------------------------
def _get_neighbors(current, instance, tabu_list, iteration, best, alpha,
                   sample_remove=None, sample_swap=30):
    """
    Returns (best_neighbor_solution, best_move).
    Adds sampling for swaps and early stopping when strong improvement found.
    """
    selected = list(current.selected)
    non_selected = [j for j in range(instance.n) if j not in current.selected]

    # Sampling parameters
    if sample_remove is None:
        sample_remove = len(selected)  # full remove scan (cheap)
    sample_remove = min(sample_remove, len(selected))
    sample_swap = min(sample_swap, len(selected), len(non_selected))

    best_neighbor, best_move, best_cost = None, None, float("inf")

    # --- REMOVE ---
    for s in random.sample(selected, sample_remove):
        if s in tabu_list and tabu_list[s] > iteration:
            if current.cost >= best.cost * alpha:
                continue
        current.remove(s)
        if current.is_feasible() and current.cost < best_cost:
            best_neighbor, best_move, best_cost = current.copy(), ("remove", s), current.cost
            # early break if we beat global best
            if current.cost < best.cost:
                current.add(s)
                return best_neighbor, best_move
        current.add(s)

    # --- ADD ---
    for s in non_selected:
        if s in tabu_list and tabu_list[s] > iteration:
            if current.cost >= best.cost * alpha:
                continue
        current.add(s)
        if current.is_feasible() and current.cost < best_cost:
            best_neighbor, best_move, best_cost = current.copy(), ("add", s), current.cost
            if current.cost < best.cost:  # early exit if global improvement found
                current.remove(s)
                return best_neighbor, best_move
        current.remove(s)

    # --- SWAP ---
    swap_out = random.sample(selected, sample_swap)
    swap_in = random.sample(non_selected, sample_swap)
    for s_out in swap_out:
        for s_in in swap_in:
            tabu_active = ((s_out in tabu_list and tabu_list[s_out] > iteration) or
                           (s_in in tabu_list and tabu_list[s_in] > iteration))
            if tabu_active and current.cost >= best.cost * alpha:
                continue
            current.remove(s_out)
            current.add(s_in)
            if current.is_feasible() and current.cost < best_cost:
                best_neighbor, best_move, best_cost = current.copy(), ("swap", s_out, s_in), current.cost
                if current.cost < best.cost:  # early exit if global improvement found
                    current.remove(s_in)
                    current.add(s_out)
                    return best_neighbor, best_move
            current.remove(s_in)
            current.add(s_out)

    return best_neighbor, best_move


# -----------------------------------------------------------
# MAIN TABU SEARCH
# -----------------------------------------------------------
def tabu_search_core(instance,
                     init_sol,
                     max_time=60.0,
                     tabu_tenure=10,
                     max_no_improve=100,
                     alpha=1.0,
                     shake_NI_max=100,
                     shake_intensity=0.1,
                     verbose=True):
    """
    Core Tabu Search logic that starts from a given initial solution.
    Used internally by both the standard tabu_search() and hybrids.
    """
    start_time = time.time()
    current = init_sol.copy()
    current.prune_by_cost()
    best = current.copy()

    # --- counters and memory ---
    no_improve_total = 0
    no_improve_since_shake = 0
    iteration = 0
    tabu_list = {}

    opt = getattr(instance, "opt_value", best.cost)
    if opt <= 0:
        opt = best.cost

    # small helper for relative deviation
    def deviation(cost):
        return max(0, 100 * (cost - opt) / opt)

    if verbose:
        print(f"\n▶️ Start Tabu Search (custom init) | "
              f"initial={best.cost:.2f} (dev={deviation(best.cost):.2f}%)\n")

    # --- main loop ---
    while time.time() - start_time < max_time and no_improve_total < max_no_improve:
        iteration += 1

        best_neighbor, best_move = _get_neighbors(current, instance, tabu_list, iteration, best, alpha)

        if best_neighbor is None:
            no_improve_total += 1
            no_improve_since_shake += 1
            if verbose:
                print(f"[Iter {iteration:3d}] No valid move | NI={no_improve_total}")
            continue

        current, tabu_list = _apply_move(best_neighbor, best_move, tabu_list, iteration, tabu_tenure)

        # --- improvement found ---
        if current.cost < best.cost:
            best = current.copy()
            no_improve_total = 0            # reset global stagnation
            no_improve_since_shake = 0      # reset local stagnation
            if verbose:
                print(f"[Iter {iteration:3d}] NEW BEST {best.cost:.2f} "
                      f"(dev={deviation(best.cost):.2f}%) | Move={_format_move(best_move)}")

            # intensify via local search
            best = best_improvement_drop_or_swap_loop(best, max_time=5.0)
            best.prune_by_cost()
            current = best.copy()

        # --- no improvement ---
        else:
            no_improve_total += 1
            no_improve_since_shake += 1

        # --- diversification trigger (relative stagnation) ---
        if no_improve_since_shake >= shake_NI_max:
            if verbose:
                print(f"⚡ Shake triggered at iter {iteration} "
                      f"(no_improve_since_shake={no_improve_since_shake})")
            current = _shake_solution(best, intensity=shake_intensity, verbose=verbose)
            current.prune_by_cost()
            no_improve_since_shake = 0   # reset local stagnation after shake

    if verbose:
        print(f"\n✅ Finished after {iteration} iters | "
              f"Best={best.cost:.2f} (dev={deviation(best.cost):.2f}%)\n")

    return best




def tabu_search(instance,
                max_time=60.0,
                tabu_tenure=10,
                max_no_improve=100,
                alpha=1.0,
                shake_NI_max=100,
                shake_intensity=0.1,
                verbose=True):
    """
    Compatibility wrapper: standard Tabu starting from a greedy solution.
    Keeps the same signature for existing code.
    """
    init_sol = greedy_cost_efficiency(instance)
    init_sol.prune_by_cost()
    return tabu_search_core(instance, init_sol,
                            max_time=max_time,
                            tabu_tenure=tabu_tenure,
                            max_no_improve=max_no_improve,
                            alpha=alpha,
                            shake_NI_max=shake_NI_max,
                            shake_intensity=shake_intensity,
                            verbose=verbose)

