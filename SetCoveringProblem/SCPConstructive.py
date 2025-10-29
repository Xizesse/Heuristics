from SCPDefinitions import *
import random
import numpy as np


def greedy_first_fit(instance):
    sol = SCPSolution(instance)
    uncovered = set(range(instance.m))  # all attributes uncovered initially

    while uncovered:
        attr = next(iter(uncovered))  # first uncovered attribute
        candidates = sorted(instance.sets_of_attr[attr])
        if not candidates:
            raise ValueError(f"No airplane covers attribute {attr}!")

        chosen = candidates[0]
        sol.add(chosen)

        # update uncovered set efficiently
        newly_covered = instance.attr_of_set[chosen]
        uncovered -= newly_covered

    return sol



def greedy_cost_efficiency(instance):
    """
    Greedy constructive heuristic:
    Select airplane j minimizing (cost_j / number of newly covered attributes).
    """
    sol = SCPSolution(instance)
    costs = np.asarray(instance.costs)
    attr_of_set = instance.attr_of_set

    while not sol.is_feasible():
        best_j = None
        best_ratio = float("inf")

        for j in range(instance.n):
            if j in sol.selected:
                continue

            uncovered = sol.covered[list(attr_of_set[j])] == 0
            new_cover = np.count_nonzero(uncovered)
            if new_cover == 0:
                continue

            ratio = costs[j] / new_cover
            if ratio < best_ratio:
                best_ratio = ratio
                best_j = j

        if best_j is None:
            raise ValueError("No airplane can cover remaining attributes!")

        sol.add(best_j)

    return sol



def greedy_cost_square_over_cover(instance):
    """
    Greedy heuristic that picks sets minimizing cost^2 / new_cover.
    Slightly favors cheap wide-cover sets over purely efficient ones.
    """
    sol = SCPSolution(instance)
    costs = np.asarray(instance.costs)
    attr_of_set = instance.attr_of_set

    while not sol.is_feasible():
        best_j, best_score = None, float("inf")

        for j in range(instance.n):
            if j in sol.selected:
                continue
            uncovered_mask = sol.covered[list(attr_of_set[j])] == 0
            new_cover = np.count_nonzero(uncovered_mask)
            if new_cover == 0:
                continue

            score = (costs[j] ** 2) / new_cover
            if score < best_score:
                best_score, best_j = score, j

        if best_j is None:
            break
        sol.add(best_j)

    return sol





def greedy_randomized_adaptive(instance, alpha=0.1, seed=42):
    """
    Greedy Randomized Adaptive Constructive Heuristic (GRASP-style)
    Builds a feasible cover using a Restricted Candidate List (RCL).

    Parameters
    ----------
    instance : SCPInstance
        Problem data.
    alpha : float ∈ [0,1]
        Controls greediness vs randomness.
        alpha = 0 → purely greedy; alpha = 1 → purely random.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    sol = SCPSolution(instance)
    costs = np.asarray(instance.costs)
    attr_of_set = instance.attr_of_set

    while not sol.is_feasible():
        eff_list = []

        for j in range(instance.n):
            if j in sol.selected:
                continue

            uncovered = sol.covered[list(attr_of_set[j])] == 0
            new_cover = np.count_nonzero(uncovered)
            if new_cover == 0:
                continue

            eff = costs[j] / new_cover
            eff_list.append((eff, j))

        if not eff_list:
            raise ValueError("No remaining airplane can cover uncovered attributes!")

        eff_list.sort(key=lambda x: x[0])
        best, worst = eff_list[0][0], eff_list[-1][0]

        if alpha == 0:
            chosen = eff_list[0][1]
        else:
            threshold = best + alpha * (worst - best)
            RCL = [j for eff, j in eff_list if eff <= threshold]
            chosen = random.choice(RCL)

        sol.add(chosen)

    return sol



def eval_expected_cost(instance, sol):
    uncovered = [a for a in range(instance.m) if sol.covered[a] == 0]
    if not uncovered:
        return 0.0
    # For each remaining attribute find the cheapest set
    
    return sum  (min(instance.costs[j] for j in instance.attr_to_sets[a]) for a in uncovered) / len(uncovered)



def greedy_with_future_cost(instance, lam=0.1):
    sol = SCPSolution(instance)

    while not sol.is_feasible():
        uncovered = [a for a in range(instance.m) if sol.covered[a] == 0]
        best_j, best_score = None, float("inf")

        for j in range(instance.n):
            if j in sol.selected:
                continue

            new_elems = [a for a in instance.set_to_attrs[j] if sol.covered[a] == 0]
            if not new_elems:
                continue

            # Immediate greedy cost-efficiency
            g = instance.costs[j] / len(new_elems)

            # Evaluate future cost if we pick this set
            temp_sol = sol.copy()
            temp_sol.add(j)
            h = eval_expected_cost(instance, temp_sol)

            score = g + lam * h
            if score < best_score:
                best_score, best_j = score, j

        if best_j is None:
            raise ValueError("No remaining set can cover uncovered elements!")

        sol.add(best_j)

    return sol





def greedy_RE(instance):
    #Greedy then prune by cost
    sol = greedy_cost_efficiency(instance)
    sol.prune_by_cost()
    return sol

def randomized_RE(instance, alpha=0.1, seed=42):
    #Randomized then prune by cost
    sol = greedy_randomized_adaptive(instance, alpha=alpha, seed=seed)
    sol.prune_by_cost()
    return sol

def squared_RE(instance):
    sol = greedy_cost_square_over_cover(instance)
    sol.prune_by_cost()
    return sol

   

