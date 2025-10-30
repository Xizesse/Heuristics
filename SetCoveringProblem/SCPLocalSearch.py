
import time
from SCPDefinitions import *
from SCPConstructive import *

def best_improvement1x1(sol, max_time=10.0):
    """
    Optimized Best Improvement Local Search (1x1 swap) for SCP.
    Identical behavior to original version, but avoids excessive copying.
    """
    start = time.time()
    inst = sol.instance
    n = inst.n

    best_cost = sol.cost
    best_s, best_t = None, None
    current = sol.copy()

    selected_list = list(current.selected)
    non_selected_list = [t for t in range(n) if t not in current.selected]

    for s in selected_list:
        if time.time() - start > max_time:
            break
        current.remove(s)

        for t in non_selected_list:
            if time.time() - start > max_time:
                break
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible() and current.cost < best_cost:
                best_cost = current.cost
                best_s, best_t = s, t
            current.remove(t)

        current.add(s)

    if best_s is not None:
        improved = sol.copy()
        improved.remove(best_s)
        if best_t is not None:
            improved.add(best_t)
        return improved
    return sol.copy()



def best_improvement_1x1_loop(sol, max_time=30.0):
    """
    Repeated Best Improvement Local Search Loop.
    """
    start = time.time()
    current = sol.copy()

    while time.time() - start < max_time:
        remaining = max_time - (time.time() - start)
        new_sol = best_improvement1x1(current, max_time=remaining)

        if new_sol.cost < current.cost:
            current = new_sol
        else:
            break  # local optimum
    return current





def first_improvement1x1(sol, max_time=10.0):
    """
    Optimized First Improvement Local Search (1x1 swap).
    Stops at first improving feasible move.
    """
    start = time.time()
    inst = sol.instance
    n = inst.n

    current = sol.copy()
    base_cost = current.cost

    selected_list = list(current.selected)
    non_selected_list = [t for t in range(n) if t not in current.selected]

    for s in selected_list:
        if time.time() - start > max_time:
            break
        current.remove(s)

        for t in non_selected_list:
            if time.time() - start > max_time:
                break
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible() and current.cost < base_cost:
                improved = current.copy()
                return improved
            current.remove(t)

        current.add(s)
    return sol.copy()



def first_improvement_1x1_loop(sol, max_time=30.0):
    """
    Repeated First Improvement 1x1 swaps until local optimum or time limit.
    """
    start = time.time()
    current = sol.copy()

    while time.time() - start < max_time:
        remaining = max_time - (time.time() - start)
        new_sol = first_improvement1x1(current, max_time=remaining)

        if new_sol.cost < current.cost:
            current = new_sol
        else:
            break
    return current




def greedy_plus_RE(instance):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    return sol

def greedy_plus_FI1x1(instance, fi_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol = first_improvement_1x1_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def greedy_plus_BI1x1(instance, ls_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol = best_improvement_1x1_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol

def squared_plus_RE(instance):
    sol = greedy_cost_square_over_cover(instance)
    sol.prune_by_cost()
    return sol

def squared_plus_FI1x1(instance, fi_time=999.0):
    sol = greedy_cost_square_over_cover(instance)
    sol = first_improvement_1x1_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def squared_plus_BI1x1(instance, ls_time=999.0):
    sol = greedy_cost_square_over_cover(instance)
    sol = best_improvement_1x1_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol

def randomized_plus_RE(instance, alpha=0.1, seed=42):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol.prune_by_cost()
    return sol

def randomized_plus_FI1x1(instance, alpha=0.1, seed=42, fi_time=999.0):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol = first_improvement_1x1_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def randomized_plus_BI1x1(instance, alpha=0.1, seed=42, ls_time=999.0):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol = best_improvement_1x1_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol


def best_improvement_drop_or_swap(sol, max_time=10.0):
    """
    Best Improvement (1x0 + 1x1) neighborhood.
    Drop or swap, single pass.
    """
    start = time.time()
    inst = sol.instance
    n = inst.n

    best_cost = sol.cost
    best_s, best_t = None, None
    current = sol.copy()

    selected_list = list(current.selected)
    non_selected_list = [t for t in range(n) if t not in current.selected]

    for s in selected_list:
        if time.time() - start > max_time:
            break
        current.remove(s)

        # 1x0 removal
        if current.is_feasible() and current.cost < best_cost:
            best_cost, best_s, best_t = current.cost, s, None

        # 1x1 swaps
        for t in non_selected_list:
            if time.time() - start > max_time:
                break
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible() and current.cost < best_cost:
                best_cost, best_s, best_t = current.cost, s, t
            current.remove(t)

        current.add(s)

    if best_s is not None:
        improved = sol.copy()
        improved.remove(best_s)
        if best_t is not None:
            improved.add(best_t)
        return improved
    return sol.copy()



def best_improvement_drop_or_swap_loop(sol, max_time=30.0):
    """
    Repeated Best Improvement Local Search Loop.
    
    Calls a single-pass local search repeatedly until:
      - No improvement is found (local optimum)
      - Or time limit is reached.
    """
    start_time = time.time()
    current = sol.copy()
    best = current.copy()

    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        remaining = max_time - elapsed

        # run one neighborhood exploration
        new_sol = best_improvement_drop_or_swap(current, max_time=remaining)

        if new_sol.cost < current.cost:
            #print(f"Improvement found ({current.cost:.2f} → {new_sol.cost:.2f})")
            current = new_sol
            best = current  
            #print(f"Improved to cost {best.cost:.2f}")
        else:
            break  # no improvement found

    return best


import random

def first_improvement_drop_or_swap(sol, max_time=10.0):
    """
    First Improvement (drop or swap, randomized order).
    """
    start = time.time()
    inst = sol.instance
    n = inst.n

    current = sol.copy()
    base_cost = current.cost

    selected_list = list(current.selected)
    random.shuffle(selected_list)
    non_selected_list = [t for t in range(n) if t not in current.selected]

    for s in selected_list:
        if time.time() - start > max_time:
            break
        current.remove(s)

        # unbiased candidate order: None (drop) + swaps
        candidates = [None] + non_selected_list[:]
        random.shuffle(candidates)

        for t in candidates:
            if time.time() - start > max_time:
                break
            if t is not None:
                if t in current.selected:
                    continue
                current.add(t)

            if current.is_feasible() and current.cost < base_cost:
                improved = current.copy()
                return improved

            if t is not None:
                current.remove(t)
        current.add(s)
    return sol.copy()


def first_improvement_drop_or_swap_loop(sol, max_time=30.0):
    start = time.time()
    current = sol.copy()

    while time.time() - start < max_time:
        remaining = max_time - (time.time() - start)
        new_sol = first_improvement_drop_or_swap(current, max_time=remaining)

        if new_sol.cost < current.cost:
            current = new_sol
        else:
            break
    return current


def greedy_plus_FI_drop_or_swap(instance, fi_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol = first_improvement_drop_or_swap_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def greedy_plus_BI_drop_or_swap(instance, ls_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol = best_improvement_drop_or_swap_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol

def squared_plus_RE(instance):
    sol = greedy_cost_square_over_cover(instance)
    sol.prune_by_cost()
    return sol

def squared_plus_FI_drop_or_swap(instance, fi_time=999.0):
    sol = greedy_cost_square_over_cover(instance)
    sol = first_improvement_drop_or_swap_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def squared_plus_BI_drop_or_swap(instance, ls_time=999.0):
    sol = greedy_cost_square_over_cover(instance)
    sol = best_improvement_drop_or_swap_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol

def randomized_plus_RE(instance, alpha=0.0, seed=42):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol.prune_by_cost()
    return sol

def randomized_plus_FI_drop_or_swap(instance, alpha=0.0, seed=42, fi_time=999.0):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol = first_improvement_drop_or_swap_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def randomized_plus_BI_drop_or_swap(instance, alpha=0.0, seed=42, ls_time=999.0):
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol = best_improvement_drop_or_swap_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol

def greedy_RE_plus_FI_drop_or_swap(instance, fi_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    sol = first_improvement_drop_or_swap_loop(sol, max_time=fi_time)
    sol.prune_by_cost()
    return sol

def greedy_RE_plus_BI_drop_or_swap(instance, ls_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    sol = best_improvement_drop_or_swap_loop(sol, max_time=ls_time)
    sol.prune_by_cost()
    return sol


def greedy_RE_BI1X1(instance, ls_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    sol = best_improvement_1x1_loop(sol, max_time=ls_time)
    return sol

def greedy_RE_FI1x1(instance, fi_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    sol = first_improvement_1x1_loop(sol, max_time=fi_time)
    return sol

