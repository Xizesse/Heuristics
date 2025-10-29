
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


