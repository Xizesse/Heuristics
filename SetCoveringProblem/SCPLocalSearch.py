
import time
from SCPDefinitions import *
from SCPConstructive import *

def best_improvement1x1(sol, max_time=10.0):
    """
    Single-pass Best Improvement Local Search (1x1 swap) for the SCP.

    Explores all possible 1x1 swaps:
      - Remove one selected set (s)
      - Add one non-selected set (t)
    Keeps the best improving solution found (if any).
    Runs only one full neighborhood exploration, not iterative loops.
    """
    start_time = time.time()
    inst = sol.instance
    n = inst.n

    best_sol = sol.copy()
    best_cost = sol.cost
    current = sol.copy()

    # Explore all possible swaps
    for s in list(current.selected):
        current.remove(s)  # temporarily remove

        for t in range(n):
            if time.time() - start_time > max_time:
                break  # stop if timeout reached
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible():
                new_cost = current.cost
                if new_cost < best_cost:
                    best_sol = current.copy()
                    best_cost = new_cost
            current.remove(t)

        current.add(s)  # restore before next s

    return best_sol


def best_improvement_1x1_loop(sol, max_time=30.0):
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
        new_sol = best_improvement1x1(current, max_time=remaining)

        if new_sol.cost < current.cost:
            #print(f"Improvement found ({current.cost:.2f} → {new_sol.cost:.2f})")
            current = new_sol.copy()
            best = current.copy()
            #print(f"Improved to cost {best.cost:.2f}")
        else:
            break  # no improvement found

    return best




def first_improvement1x1(sol, max_time=10.0):
    """
    Single-pass First Improvement Local Search (1x1 swap) for the SCP.

    Explores all (s, t) swaps:
      - Remove one selected set (s)
      - Add one non-selected set (t)
    As soon as a feasible improving swap is found, it returns that new solution.
    If no improvement is found after exploring all swaps, returns the same solution.
    """
    start_time = time.time()
    inst = sol.instance
    n = inst.n

    current = sol.copy()
    base_cost = current.cost

    # Explore all possible swaps
    for s in list(current.selected):
        current.remove(s)  # temporarily remove

        for t in range(n):
            if time.time() - start_time > max_time:
                break
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible():
                new_cost = current.cost
                if new_cost < base_cost:
                    # Found first improving feasible swap
                    improved_sol = current.copy()
                    return improved_sol
            current.remove(t)

        current.add(s)  # restore before next s

    # No improvement found
    return sol.copy()


def first_improvement_1x1_loop(sol, max_time=30.0):
    """
    Repeated First Improvement Local Search Loop.
    
    Repeatedly applies single-pass First Improvement 1x1 swaps until:
      - No further improvement is found (local optimum)
      - Or time limit is reached.
    """
    start_time = time.time()
    current = sol.copy()
    best = current.copy()
    iteration = 0

    while time.time() - start_time < max_time:
        iteration += 1
        remaining = max(1e-6, max_time - (time.time() - start_time))

        # Run one neighborhood exploration (stops at first improvement)
        new_sol = first_improvement1x1(current, max_time=remaining)

        if new_sol.cost < current.cost:
            #print(f"Improvement found ({current.cost:.2f} → {new_sol.cost:.2f})")
            current = new_sol.copy()
            best = current.copy()
        else:
            break  # local optimum reached

    return best



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
    Single-pass Best Improvement Local Search (1x0 + 1x1) for the SCP.

    Neighborhood:
      - 1x0 (drop): remove one selected set if solution remains feasible.
      - 1x1 (swap): remove one selected set, add one non-selected set.

    Keeps the best improving feasible move (if any).
    Returns the improved solution, or the original if no improvement found.
    """
    start_time = time.time()
    inst = sol.instance
    n = inst.n

    best_sol = sol.copy()
    best_cost = sol.cost
    current = sol.copy()

    # Explore all possible removals and swaps
    for s in list(current.selected):
        current.remove(s)  # temporarily remove

        # --- 1x0 removal ---
        if current.is_feasible():
            new_cost = current.cost
            if new_cost < best_cost:
                best_sol = current.copy()
                best_cost = new_cost

        # --- 1x1 swaps ---
        for t in range(n):
            if time.time() - start_time > max_time:
                break
            if t in current.selected:
                continue

            current.add(t)
            if current.is_feasible():
                new_cost = current.cost
                if new_cost < best_cost:
                    best_sol = current.copy()
                    best_cost = new_cost
            current.remove(t)

        # Restore before next s
        current.add(s)

        if time.time() - start_time > max_time:
            break

    return best_sol


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
            current = new_sol.copy()
            best = current.copy()
            #print(f"Improved to cost {best.cost:.2f}")
        else:
            break  # no improvement found

    return best


def first_improvement_drop_or_swap(sol, max_time=10.0):
    """
    First Improvement (unbiased within each s):
      - Shuffle order of s
      - For each s, shuffle [None] + non_selected so drop vs swap is unbiased
      - Return at first improving feasible move
    """
    start = time.time()
    inst = sol.instance
    n = inst.n

    current = sol.copy()
    base_cost = current.cost

    selected_list = list(current.selected)
    random.shuffle(selected_list)

    # Precompute non-selected once (relative to the current solution)
    non_selected_list = [i for i in range(n) if i not in current.selected]

    for s in selected_list:
        if time.time() - start > max_time:
            break
        if s not in current.selected:   # defensive (in case of external changes)
            continue

        # Build unbiased candidate list for this s: drop (None) + all swaps
        candidates = [None] + non_selected_list[:]
        random.shuffle(candidates)

        # Temporarily remove s (we'll restore after trying candidates)
        current.remove(s)

        for t in candidates:
            if time.time() - start > max_time:
                break

            if t is not None:
                if t in current.selected:   # should not happen, but be safe
                    continue
                current.add(t)

            if current.is_feasible() and current.cost < base_cost:
                return current.copy()   # first improvement found

            if t is not None:
                current.remove(t)

        # restore s before moving to the next s
        current.add(s)

    # No improvement found in this pass
    return sol.copy()


def first_improvement_drop_or_swap_loop(sol, max_time=30.0):
    """
    Repeated randomized FI loop.
    Runs successive randomized passes until no improvement or time limit.
    """
    start_time = time.time()
    current = sol.copy()
    best = current.copy()
    iteration = 0

    while time.time() - start_time < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start_time)
        new_sol = first_improvement_drop_or_swap(current, max_time=remaining)

        if new_sol.cost < current.cost:
            current = new_sol.copy()
            best = current.copy()
        else:
            break  # local optimum

    return best


def greedy_plus_RE(instance):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    return sol

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

def greedy_plus_FI1x1(instance, fi_time=999.0):
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    sol = first_improvement_1x1_loop(sol, max_time=fi_time)
    return sol

