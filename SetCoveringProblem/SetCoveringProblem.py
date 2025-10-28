# %% [markdown]
# 

# %% [markdown]
# # Instance Representation — `SCPInstance`
# 
# Given an SCP instance:
# 
# - $m$: number of attributes (elements to cover)  
# - $n$: number of sets (airplanes)  
# - $\mathbf{c} = [c_1, c_2, \dots, c_n]$: cost of each airplane  
# - Coverage is stored implicitly using sparse mappings:
# 
#   - `attr_of_set[j]` → attributes covered by set \( j \)  
#     (corresponds to column \( j \) of \( A \))  
# 
#   - `sets_of_attr[i]` → sets that cover attribute \( i \)  
#     (corresponds to row \( i \) of \( A \))
# 
# Thus,
# 
# $$
# a_{ij} = 1 \iff i \in \text{attr\_of\_set}[j] \iff j \in \text{sets\_of\_attr}[i].
# $$
# 
# The instance defines the data for the optimization model but does not explicitly store matrix \(A\).
# 

# %%
import os
import re
import numpy as np

class SCPInstance:
    def __init__(self, index, folder="SCP-Instances", sol_file="Solutions.txt"):
        self.folder = folder
        self.sol_file = sol_file
        self.index = index

        self.filename = self._get_filename(index)
        self.name = self.filename.replace(".txt", "")

        self.path = os.path.join(folder, self.filename)

        self.m, self.n, self.costs, self.attr_of_set, self.sets_of_attr = self._load_instance()
        self.opt_value = self._load_opt_value()

    def _get_filename(self, index):
        """
        Get the filename of the SCP instance based on the index.
        """
        files = sorted(
            [f for f in os.listdir(self.folder)
             if f.lower().startswith("scp") and f.lower().endswith(".txt")]
        )
        if not files:
            raise FileNotFoundError(f"No SCP files found in {self.folder}.")
        if isinstance(index, int):
            if index >= len(files):
                raise IndexError(f"Index {index} out of range (found {len(files)} files).")
            filename = files[index]
        else:
            raise TypeError("Index must be an integer.")
        return filename

    def _load_instance(self):
        """
        Load SCP instance from file.
        """
        with open(self.path, "r") as f:
            data = list(map(int, f.read().split()))

        m, n = data[0], data[1]
        costs = data[2:2 + n]

        attr_of_set = [set() for _ in range(n)]
        sets_of_attr = [set() for _ in range(m)]

        idx = 2 + n
        for attr in range(m):
            k_i = data[idx]
            idx += 1
            airplanes = data[idx:idx + k_i]
            idx += k_i
            for j in airplanes:
                attr_of_set[j - 1].add(attr)
                sets_of_attr[attr].add(j - 1)

        return m, n, costs, attr_of_set, sets_of_attr



    def _load_opt_value(self):
        """
        Load known optimal value from solutions file.
        """
        base = self.filename.lower().replace("scp", "").replace(".txt", "")
        sol_id = f"{base[0].upper()}.{base[1:]}" if base[0].isalpha() else f"{base[0]}.{base[1:]}"
        opt_value = None

        if os.path.exists(self.sol_file):
            with open(self.sol_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0].upper() == sol_id:
                        opt_value = float(parts[1])
                        break
        else:
            print(f"⚠️ Solutions file '{self.sol_file}' not found in current directory.")

        return opt_value
    

    def summary(self, max_show=4):
        """
        Print a summary of the SCP instance.
        """
        print("=" * 70)
        print(f"📘 Instance: {self.filename}")
        print(f"  Attributes (m): {self.m}")
        print(f"  Airplanes (n):  {self.n}")
        print(f"  Known optimal cost: {self.opt_value if self.opt_value else 'Unknown'}")
        print("Costs sample:\n", self.costs[:6], "..." if len(self.costs) > 10 else "")
        print("Example coverage:")
        for i in range(min(max_show, self.m)):
            print(f"  Attribute {i}: covered by {list(self.sets_of_attr[i])[:8]}")
        print("Example airplane coverage:")
        for j in range(min(max_show, self.n)):
            print(f"  Airplane {j}: covers {list(self.attr_of_set[j])[:8]}")
        print("=" * 70)

# %%
inst = SCPInstance(3)
inst.summary()

# %% [markdown]
# # Solution representation
# 
# A solution is represented by a binary vector of size n, where each position i indicates whether subset i is included in the cover (1) or not (0).

# %% [markdown]
# Given an SCP instance defined by:
# 
# - $m$: number of attributes (elements to cover)  
# - $n$: number of sets (airplanes)  
# - $\mathbf{c} = [c_1, c_2, \dots, c_n]$: cost vector  
# - $A \in \{0,1\}^{m \times n}$: coverage matrix with entries  
#   $$
#   a_{ij} =
#   \begin{cases}
#   1, & \text{if attribute } i \text{ is covered by set } j, \\
#   0, & \text{otherwise.}
#   \end{cases}
#   $$
# 
# ---
# 
# ### **Decision Variables**
# 
# Each airplane (set) $j$ has a binary decision variable:
# $$
# x_j =
# \begin{cases}
# 1, & \text{if airplane } j \text{ is selected,} \\
# 0, & \text{otherwise.}
# \end{cases}
# $$
# 
# In code:
# - `self.selected`  →  $\{\,j : x_j = 1\,\}$  
# - `self.covered[i]`  →  $\displaystyle \sum_{j=1}^{n} a_{ij} x_j$  
# - `self.cost`  →  $\displaystyle \sum_{j=1}^{n} c_j x_j$
# 
# ---
# 
# ### **Objective Function**
# 
# Minimize the total cost of selected airplanes:
# $$
# \min_{x \in \{0,1\}^n} Z = \sum_{j=1}^{n} c_j x_j
# $$
# 
# ---
# 
# ### **Feasibility Constraints**
# 
# Each attribute must be covered by at least one selected set:
# $$
# \sum_{j=1}^{n} a_{ij} x_j \ge 1, \quad \forall\, i = 1,\dots,m.
# $$
# 
# In code:  
# `is_feasible()` → checks `np.all(self.covered > 0)`
# 
# ---
# 
# 
# 

# %%
import numpy as np

class SCPSolution:
    def __init__(self, instance):
        """
        Initialize an empty solution for a given SCPInstance.
        """
        self.instance = instance  # reference to the problem data
        self.selected = set()     # chosen airplanes (indices)
        self.covered = np.zeros(instance.m, dtype=int)  # coverage count per attribute
        self.cost = 0.0

        # Pre-cache for speed
        self.costs = np.asarray(instance.costs)
        self.attr_of_set = instance.attr_of_set

    @classmethod
    def from_csv(cls, instance, csv_name):
        """
        Build an SCPSolution for the given instance from a CSV file
        that contains the 'solution_sets' column.

        Example:
            sol = SCPSolution.from_csv(inst, "results/greedy_BI_drop_or_swap.csv")
        """
        import pandas as pd

        df = pd.read_csv(csv_name)
        row = df.loc[df["instance_name"] == instance.name]
        if row.empty:
            raise ValueError(f"Instance '{instance.name}' not found in '{csv_name}'.")
        row = row.iloc[0]

        # Parse selected sets
        raw = str(row.get("solution_sets", "")).strip()
        if raw:
            raw = raw.strip("[]")
            tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
            selected_sets = set(map(int, tokens))
        else:
            selected_sets = set()

        # Construct and rebuild
        sol = cls(instance)
        sol.selected = selected_sets
        sol.rebuild_from_selected()

        # Optional check against CSV
        csv_cost = row.get("solution_cost", None)
        if csv_cost is not None:
            try:
                csv_cost_f = float(csv_cost)
                if abs(sol.cost - csv_cost_f) > 1e-6:
                    print(f"⚠️ Cost mismatch: recomputed={sol.cost:.6f} vs CSV={csv_cost_f:.6f}")
            except Exception:
                pass

        print(f"✅ Loaded solution for {instance.name}: "
              f"{len(sol.selected)} sets, feasible={sol.is_feasible()}, cost={sol.cost:.2f}")
        return sol

    def add(self, j: int) -> None:
        """Select airplane j and update coverage and cost."""
        if j in self.selected:
            return
        self.selected.add(j)
        self.cost += self.costs[j]
        for a in self.attr_of_set[j]:
            self.covered[a] += 1

    def remove(self, j: int) -> None:
        """Remove airplane j and update coverage and cost."""
        if j not in self.selected:
            return
        self.selected.remove(j)
        self.cost -= self.costs[j]
        for a in self.attr_of_set[j]:
            self.covered[a] -= 1

    def is_feasible(self) -> bool:
        """Return True if all attributes are covered at least once."""
        return np.all(self.covered > 0)

    def uncovered_attributes(self) -> list[int]:
        """Return list of uncovered attribute indices."""
        return np.flatnonzero(self.covered == 0).tolist()

    def prune_by_cost(self) -> None:
        """
        Try removing airplanes in decreasing cost order.
        A set is removed if the solution remains feasible.
        """
        # Sort selected sets by descending cost
        for j in sorted(self.selected, key=lambda x: self.costs[x], reverse=True):
            # Quick local check: skip if removing j would uncover something
            if any(self.covered[a] == 1 for a in self.attr_of_set[j]):
                continue
            self.remove(j)

    def rebuild_from_selected(self):
        """
        Recompute coverage and cost from the current `self.selected`.
        Use when you assign `selected` directly (e.g., after loading from CSV).
        """
        self.covered[:] = 0
        self.cost = 0.0
        for j in self.selected:
            self.cost += self.costs[j]
            for a in self.attr_of_set[j]:
                self.covered[a] += 1
        return self

    def copy(self) -> "SCPSolution":
        """Return a deep copy of the current solution."""
        new_sol = SCPSolution(self.instance)
        new_sol.selected = self.selected.copy()
        new_sol.covered = self.covered.copy()
        new_sol.cost = self.cost
        return new_sol

    def summary(self, max_show: int = 10) -> None:
        """Print concise summary of the solution."""
        feasible = self.is_feasible()
        uncovered = self.uncovered_attributes()

        print("=" * 60)
        print("✈️  SCP Solution Summary")
        print(f"  Selected airplanes: {len(self.selected)}")
        print(f"  Total cost: {self.cost:.2f}")
        print(f"  Feasible: {feasible}")
        print(f"  Uncovered attributes: {len(uncovered)}")
        print(f"Selected (sample): {sorted(self.selected)[:max_show]}")
        if uncovered:
            print(f"Uncovered (sample): {uncovered[:max_show]}")
        print("=" * 60)
        print()

    def compute_cost(self):
        """Recalculate total cost from selected sets."""
        self.cost = sum(self.instance.costs[i] for i in self.selected)
        return self.cost


# %%
inst = SCPInstance(3)
sol = SCPSolution(inst)

sol.add(5)
sol.add(10)
sol.add(200)
sol.remove(10)

sol.summary()
sol.prune_by_cost()
sol.summary()


# %%
#Load from csv
inst = SCPInstance(0, folder="SCP-Instances")
sol = SCPSolution.from_csv(inst, "results/greedy_RE.csv")

sol.summary()


# %% [markdown]
# ### ✈️ CH0: Greedy First Fit Heuristic
# 
# <span style="color:red">Not being used</span>
# 
# A simple **constructive heuristic** that builds a feasible solution step by step.
# 
# **Idea:**  
# Start with no airplanes selected. Repeatedly pick the first uncovered attribute \( i \),  
# choose the first set \( j \) that covers it, set \( x_j \leftarrow 1 \),  
# update coverage, and stop when all attributes are covered.
# 
# Formally, at each iteration:
# 
# $$
# j^{*} = \min \{\, j : a_{ij} = 1 \,\}, \qquad
# i = \text{first uncovered attribute}.
# $$
# 
# Then set \( x_{j^{*}} = 1 \).
# 
# **Characteristics:**  
# - Deterministic  
# - Greedy and myopic (covers one attribute at a time)  
# - Produces a feasible solution, not necessarily optimal  
# - A pruning step can later remove redundant sets
# 

# %%
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


# %%
inst = SCPInstance(0)  # load first file in folder
sol = greedy_first_fit(inst)
sol.summary()
sol.prune_by_cost()
sol.summary()


# %% [markdown]
# ### 💰 Ch1 : Greedy Cost-Efficient Heuristic
# 
# A **cost-aware constructive heuristic** that builds a feasible solution by balancing coverage and cost.
# 
# **Idea:**  
# At each step, select the airplane \( j \) that minimizes the ratio between its cost and  
# the number of *new* uncovered attributes it would cover.
# 
# Formally:
# 
# $$
# h(j) = \frac{c_j}{|\{\, i : a_{ij} = 1 \text{ and } \text{covered}[i] = 0 \,\}|}
# $$
# 
# At each iteration:
# 
# $$
# j^{*} = \arg\min_{j \notin \text{selected}} h(j)
# \quad \Longrightarrow \quad x_{j^{*}} = 1.
# $$
# 
# Repeat until all attributes are covered.
# 
# **Characteristics:**  
# - Deterministic  
# - Balances cost vs. coverage gain  
# - More informed than pure greedy-first-fit  
# - Still locally myopic — does not look ahead
# 

# %%
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


# %%
inst = SCPInstance(0)
sol = greedy_cost_efficiency(inst)
sol.summary()
sol.prune_by_cost()
sol.summary()


# %% [markdown]
# ### 💰 CH2: Greedy Cost Squared over Cover Heuristic
# 
# A **cost-sensitive constructive heuristic** that increases the penalty on expensive sets while still favoring wide coverage.
# 
# **Idea:**  
# At each step, select the airplane \( j \) that minimizes the ratio between the **square of its cost** and  
# the number of *new* uncovered attributes it would cover.
# 
# Formally:
# 
# $$
# h(j) = \frac{c_j^2}{|\{\, i : a_{ij} = 1 \text{ and } \text{covered}[i] = 0 \,\}|}
# $$
# 
# At each iteration:
# 
# $$
# j^{*} = \arg\min_{j \notin \text{selected}} h(j)
# \quad \Longrightarrow \quad x_{j^{*}} = 1.
# $$
# 
# Repeat until all attributes are covered.
# 
# **Characteristics:**  
# - Deterministic  
# - Applies a quadratic cost penalty (discourages expensive sets)  
# - Favors multiple cheap sets over single costly ones  
# - More conservative than the standard cost-efficiency heuristic  
# - Still locally greedy — no lookahead or randomness
# 

# %%
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


# %%
inst = SCPInstance(0)
sol = greedy_cost_square_over_cover(inst)
sol.summary()
sol.prune_by_cost()
sol.summary()


# %% [markdown]
# ### 🎲 CH3: Greedy Randomized Adaptive Heuristic (GRASP Constructive Phase)
# 
# A **stochastic extension** of the greedy cost-efficient heuristic.
# 
# At each step:
# - Compute the efficiency of each set  
#   \( h(j) = \dfrac{c_j}{|\{\, i : a_{ij} = 1 \text{ and } \text{covered}[i] = 0 \,\}|} \)
# - Sort sets by efficiency (lower is better)
# - Build a **Restricted Candidate List (RCL)** containing the best candidates:
#   $$
#   \text{RCL} = \{\, j : h(j) \le h_{\min} + \alpha (h_{\max} - h_{\min}) \,\}
#   $$
# - Randomly pick one set \( j^* \in \text{RCL} \)
# - Add that set to the solution: \( x_{j^*} \leftarrow 1 \)
# - Update covered attributes
# 
# Repeat until all attributes are covered.
# 
# **Parameter:**
# - \( \alpha \in [0,1] \): controls greediness  
#   - \( \alpha = 0 \): purely greedy  
#   - \( \alpha = 1 \): purely random  
# 
# **Characteristics:**  
# - Randomized but adaptive to current coverage  
# - Balances exploration (randomness) and exploitation (greedy choice)  
# - Provides diverse starting solutions for local search
# 

# %%
import random
import numpy as np

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


# %% [markdown]
# # Look ahead heuristics
# <span style="color:red">Not being used</span>
# 
# Non-myopic heuristics that consider future consequences of current choices.
# 
# \textcolor{orange}{To Be improved}

# %%
# Evaluate a solution

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


# %% [markdown]
# # Running all instances
# 

# %%
import os, time, pandas as pd, sys

def solve_all_instances(algorithm_name, csv_filename, folder="SCP-Instances", num_instances=0, *args, **kwargs):
    """
    Runs the given solver across all SCP instances and saves results to 'results/'.

    Adds a 'solution_sets' column listing the selected sets in the final solution.

    Minimal live feedback:
        Solver: Z
        Instance X/42
    Final line:
        Solver: Z Average deviation: +X.XX%
    """

    solver_func = globals().get(algorithm_name)
    if solver_func is None or not callable(solver_func):
        raise ValueError(f"Solver '{algorithm_name}' not found or not callable.")

    # Prepare output folder
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_filename)

    # Load instances
    files = sorted(f for f in os.listdir(folder) if f.lower().startswith("scp"))
    if num_instances > 0:
        files = files[:num_instances]
    total_instances = len(files)

    results = []
    print(f"Solver: {algorithm_name}")
    start_all = time.time()

    for i, _ in enumerate(files, start=1):
        inst = SCPInstance(i - 1, folder=folder)
        opt = inst.opt_value if inst.opt_value is not None else 0

        t0 = time.time()
        sol = solver_func(inst, *args, **kwargs)
        elapsed = time.time() - t0

        cost = getattr(sol, "cost", None)
        selected_sets = getattr(sol, "selected", None)

        # Convert selected sets to CSV-friendly string
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

        # Live progress line (overwrites same line)
        sys.stdout.write(f"\rInstance {i}/{total_instances}")
        sys.stdout.flush()

    print()  # newline after loop

    # Compute final stats
    valid_devs = [r["deviation_%"] for r in results if r["deviation_%"] is not None]
    avg_dev = round(sum(valid_devs) / len(valid_devs), 2) if valid_devs else 0.0

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print(f"Solver: {algorithm_name} Average deviation: {avg_dev:+.2f}%")
    return df


# %%

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

   

# %%
df_greedy = solve_all_instances("greedy_cost_efficiency", "greedy.csv")
df_greedy_RE = solve_all_instances("greedy_RE", "greedy_RE.csv") 

# %%
df_squared = solve_all_instances("greedy_cost_square_over_cover", "greedy_squared_results.csv")
df_squared_RE = solve_all_instances("squared_RE", "squared_RE_results.csv")

# %%
df_random = solve_all_instances("greedy_randomized_adaptive", "random.csv", alpha=0.0)
df_random_RE = solve_all_instances("randomized_RE", "random_RE.csv", alpha=0.0)

# %%


# %%


# %% [markdown]
# # Local Search 

# %% [markdown]
# Tasks:
# 
# Consider one neighbourhood
# 
# Implement first-improvement (FI) and best-improvement (BI) algorithms for the SCP. 
#  
# In these two algorithms, consider one neighborhoods of your choice. Apply redundancy elimination after each step.
# 
# Apply each of these algorithms once to an initial solution generated by CH1, CH2, CH3, and CH1+RE.  H
# 
# ence, in total eight algorithms should be tested obtained by the combinations of the four constructive heuristics with the two iterative improvement algorithms. 
# 
# As variance reduction technique for the experiments make sure that the FI and BI algorithms start from the same initial solution. This can be ensured by using for each of the executions on a same instance the same random number seed. 
# 
# As the experimental results report for each of the experiments
# the average percentage deviation from best known solutions;
# the total computation time across all instances;
# the fraction of instances that profit from the additional local search phase.
# 
# Determine by means of statistical tests (in this case, the Student t-test or the Wilcoxon test), whether there is a statistically significant difference between the solutions generated by the various algorithms. Consider the data for the Set Covering Problem (SCP) available in Moodle.
# 

# %% [markdown]
# # Best Improvement Local Search 1x1

# %%
import time

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


# %%
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


# %% [markdown]
# # First Improvement Local Search 1x1

# %%
import time

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


# %%
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


# %%
#example
inst = SCPInstance(0)
#sol.summary()

print("Squared + RE")
sol_squared_RE = greedy_cost_square_over_cover(inst)
sol_squared_RE = sol.copy()
sol_squared_RE.prune_by_cost()
sol_squared_RE.summary()

print("Squared + FI + RE")
sol_squared_FI1x1 = greedy_cost_square_over_cover(inst)
sol_squared_FI1x1 = best_improvement_1x1_loop(sol, max_time=99999)
sol_squared_FI1x1.prune_by_cost()
sol_squared_FI1x1.summary()

print("Squared + BI + RE")
sol_squared_BI1x1 = greedy_cost_square_over_cover(inst)
sol_squared_BI1x1 = best_improvement_1x1_loop(sol, max_time=99999)
sol_squared_BI1x1.prune_by_cost()
sol_squared_BI1x1.summary()



# %% [markdown]
# # Running all instances for 1x1 Search

# %%
# Running all instances for 1x1 Search
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


# %%
df_greedy_RE = solve_all_instances("greedy_plus_RE", "greedy_RE.csv")
df_greedy_FI_RE = solve_all_instances("greedy_plus_FI1x1", "greedy_FI1x1.csv", fi_time=9999.0)
df_greedy_BI_RE = solve_all_instances("greedy_plus_BI1x1", "greedy_BI1x1.csv", ls_time=9999.0)



# %%
df_squared_RE = solve_all_instances("squared_plus_RE", "squared_RE.csv")
df_greedy_FI_RE = solve_all_instances("squared_plus_FI1x1", "squared_FI1x1.csv", fi_time=9999.0)
df_greedy_BI_RE = solve_all_instances("squared_plus_BI1x1", "squared_BI1x1.csv", ls_time=9999.0)



# %%


# %% [markdown]
# # Best Improvement Local Search Drop or Swap

# %%
import time

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


# %%
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


# %% [markdown]
# # First Improvement Local Search Drop or Swap

# %%
import time, random

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


# %%
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


# %%
#example
inst = SCPInstance(0)
#sol.summary()

print("Squared + RE")
sol = greedy_cost_square_over_cover(inst)
sol_squared_RE = sol.copy()
sol_squared_RE.prune_by_cost()
sol_squared_RE.summary()

print("Squared + FI + RE")
sol_squared_FI1x1 = greedy_cost_square_over_cover(inst)
sol_squared_FI1x1 = first_improvement_drop_or_swap_loop(sol, max_time=99999)
sol_squared_FI1x1.prune_by_cost()
sol_squared_FI1x1.summary()

print("Squared + BI + RE")
sol_squared_BI1x1 = greedy_cost_square_over_cover(inst)
sol_squared_BI1x1 = best_improvement_drop_or_swap_loop(sol, max_time=99999)
sol_squared_BI1x1.prune_by_cost()
sol_squared_BI1x1.summary()



# %% [markdown]
# # Running All Instances for Drop or Swap

# %%
# Running all instances for 1x1 Search
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


# %%
df_greedy_RE = solve_all_instances("greedy_plus_RE", "greedy_RE.csv")
df_greedy_FI_RE = solve_all_instances("greedy_plus_FI_drop_or_swap", "greedy_FI_drop_or_swap.csv", fi_time=9999.0)
df_greedy_BI_RE = solve_all_instances("greedy_plus_BI_drop_or_swap", "greedy_BI_drop_or_swap.csv", ls_time=9999.0)



# %%
df_squared_RE = solve_all_instances("squared_plus_RE", "squared_RE.csv")
df_squared_FI_RE = solve_all_instances("squared_plus_FI_drop_or_swap", "squared_FI_drop_or_swap.csv", fi_time=9999.0)
df_squared_BI_RE = solve_all_instances("squared_plus_BI_drop_or_swap", "squared_BI_drop_or_swap.csv", ls_time=9999.0)

# %%
#randomized
df_randomized_RE = solve_all_instances("randomized_plus_RE", "randomized_RE.csv")
df_randomized_FI_RE = solve_all_instances("randomized_plus_FI_drop_or_swap", "randomized_FI_drop_or_swap.csv", alpha = 0.0, fi_time=9999.0)
df_randomized_BI_RE = solve_all_instances("randomized_plus_BI_drop_or_swap", "randomized_BI_drop_or_swap.csv", alpha = 0.0, ls_time=9999.0)

# %%
df_greedy_RE_FI = solve_all_instances("greedy_RE_plus_FI_drop_or_swap", "greedy_RE_FI_drop_or_swap.csv", fi_time=9999.0)
df_greedy_RE_BI = solve_all_instances("greedy_RE_plus_BI_drop_or_swap", "greedy_RE_BI_drop_or_swap.csv", ls_time=9999.0)

# %%


# %% [markdown]
# # Pruning First

# %%
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

df_greedy_RE_BI1X1 = solve_all_instances("greedy_RE_BI1X1", "greedy_RE_BI1X1_results.csv", ls_time=9999.0)
df_greedy_RE_FI1X1 = solve_all_instances("greedy_RE_FI1X1", "greedy_RE_FI1X1_results.csv", fi_time=9999.0)

# %% [markdown]
# # GRASP : Greedy Randomized Adaptive Search Procedure

# %%



