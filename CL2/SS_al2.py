import random

# class PermutationGroup:
#     def __init__(self, n):
#         self.n = n
#         self.permutations = []

#     def add_permutation(self, perm):
#         if len(perm) != self.n:
#             raise ValueError("Permutation length must be equal to n")
#         if not self.is_permutation(perm):
#             raise ValueError("Invalid permutation")
#         self.permutations.append(perm)

#     def is_permutation(self, perm):
#         if len(perm) != self.n:
#             return False
#         seen = [False] * self.n
#         for p in perm:
#             if p < 0 or p >= self.n or seen[p]:
#                 return False
#             seen[p] = True
#         return True

#     def identity(self):
#         return list(range(self.n))

# def multiply_permutations(p1, p2):
#     if len(p1) != len(p2):
#         raise ValueError("Permutations must be of same length")
#     return [p1[p2[i]] for i in range(len(p1))]

# def inverse_permutation(p):
#     n = len(p)
#     inv = [0] * n
#     for i in range(n):
#         inv[p[i]] = i
#     return inv

# class SchreierSims:
#     def __init__(self, n):
#         self.n = n
#         self.group = PermutationGroup(n)
#         self.base = []
#         self.schreier_trees = []
#         self.generators = []

#     def random_schreier_sims(self, base):
#         self.base = base
#         self.schreier_trees = []
#         for b in base:
#             self.schreier_trees.append(self.orbit_and_transversal(b))

#     def orbit_and_transversal(self, b):
#         orbit = [b]
#         transversal = {b: self.group.identity()}
#         queue = [b]
#         while queue:
#             alpha = queue.pop(0)
#             for perm in self.group.permutations:
#                 beta = perm[alpha]
#                 if beta not in orbit:
#                     orbit.append(beta)
#                     transversal[beta] = multiply_permutations(transversal[alpha], perm)
#                     queue.append(beta)
#         return (orbit, transversal)

#     def add_permutation(self, perm):
#         self.group.add_permutation(perm)
#         self.generators.append(perm)

#     def update_with_partition(self, partition):
#         partition_blocks = partition.cell_dict
#         new_permutations = []

        
#         for color_id, nodes in partition_blocks.items():
#             permutation = list(range(len(partition.cls)))
#             for i in range(len(nodes)):
#                 permutation[nodes[i]] = nodes[(i + 1) % len(nodes)]
#             new_permutations.append(permutation)

#         for perm in new_permutations:
#             self.add_permutation(perm)
#         self.random_schreier_sims(self.base)

#     def get_generators(self):
#         return self.generators

# def initialize_schreier_sims_with_partitions(partition1, partition2):
#     n = len(partition1.cls)
#     schreier_sims = SchreierSims(n)

#     # Create permutations that map elements of partition1 to partition2
#     permutation = [0] * n
#     for i in range(n):
#         permutation[partition1.cls[i]] = partition2.cls[i]

#     # Add permutation to Schreier-Sims
#     schreier_sims.add_permutation(permutation)
#     schreier_sims.random_schreier_sims(list(range(n)))
#     return schreier_sims



# Import necessary libraries
def multiply_permutations(p1, p2):
    """Compose two permutations."""
    if len(p1) != len(p2):
        raise ValueError("Permutations must be of same length")
    return [p1[p2[i]] for i in range(len(p1))]

def inverse_permutation(p):
    """Compute the inverse of a permutation."""
    n = len(p)
    inv = [0] * n
    for i in range(n):
        inv[p[i]] = i
    return inv

def compute_permutation(partition1, partition2):
    """Compute the permutation that maps partition2 to partition1."""
    mapping = {}
    for i in range(len(partition1)):
        mapping[partition2[i]] = partition1[i]
    # Build permutation list
    n = len(partition1)
    permutation = [0] * n
    for i in range(n):
        permutation[i] = mapping.get(i, i)
    return permutation

def permutation_to_cycles(perm):
    """Convert a permutation list to cycle notation."""
    n = len(perm)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            if len(cycle) > 1:
                cycles.append(cycle)
    return cycles

class PermutationGroup:
    def __init__(self, n):
        self.n = n
        self.generators = []

    def add_permutation(self, perm):
        if len(perm) != self.n:
            raise ValueError("Permutation length must be equal to n")
        if not self.is_permutation(perm):
            raise ValueError("Invalid permutation")
        self.generators.append(perm)

    def is_permutation(self, perm):
        return sorted(perm) == list(range(self.n))

    def identity(self):
        return list(range(self.n))

class SchreierSims:
    def __init__(self, n):
        self.n = n
        self.group = PermutationGroup(n)
        self.base = []
        self.strong_gens = []
        self.stabilizers = []
        self.current_partition = None  # Keep track of the current partition
        self.partition_set = []

    def add_permutation(self, perm):
        self.group.add_permutation(perm)

    def schreier_sims(self):
        """Compute a base and strong generating set using the Schreier-Sims algorithm."""
        # Initialize base (you can choose any ordering)
        self.base = list(range(self.n))
        # Initialize the stabilizer chains
        self.stabilizers = [{} for _ in range(self.n)]
        # Start with the identity permutation
        self.strong_gens = []
        for gen in self.group.generators:
            self.insert_strong_gen(gen)

    def insert_strong_gen(self, gen):
        """Insert a strong generator into the stabilizer chains."""
        g = gen
        for i in range(len(self.base)):
            b = self.base[i]
            beta = g[b]
            if beta != b:
                if beta not in self.stabilizers[i]:
                    self.stabilizers[i][beta] = g
                    self.strong_gens.append(g)
                    break
                else:
                    h = self.stabilizers[i][beta]
                    h_inv = inverse_permutation(h)
                    g = multiply_permutations(h_inv, g)
            else:
                continue

    def get_group_elements(self):
        """Generate all elements of the group."""
        # This function can be computationally intensive for large groups
        elements = set()
        queue = [self.group.identity()]
        while queue:
            g = queue.pop()
            g_tuple = tuple(g)
            if g_tuple in elements:
                continue
            elements.add(g_tuple)
            for gen in self.group.generators:
                new_g = multiply_permutations(gen, g)
                queue.append(new_g)
        return [list(elem) for elem in elements]

    def print_generators(self):
        """Print the generators in cycle notation."""
        print("\nGenerators:")
        for gen in self.group.generators:
            cycles = permutation_to_cycles(gen)
            cycle_strs = ['(' + ' '.join(map(str, cycle)) + ')' for cycle in cycles]
            print(' '.join(cycle_strs))

    def print_group_elements(self):
        """Print the group elements in cycle notation."""
        elements = self.get_group_elements()
        print("\nGroup elements:")
        for elem in elements:
            cycles = permutation_to_cycles(elem)
            if cycles:
                cycle_strs = ['(' + ' '.join(map(str, cycle)) + ')' for cycle in cycles]
                print(' '.join(cycle_strs))
            else:
                print('()')  # Identity permutation

    def update_with_new_partition(self, new_partition):
        """Update the group with a new partition by computing the permutation and updating the group."""
        if self.current_partition is None:
            # If no current partition, set the new partition as current and do nothing
            self.current_partition = new_partition
            self.partition_set.append(new_partition)
            # print("Initial partition set.")
            return

        # Compute the permutation mapping current partition to new partition
        perm = compute_permutation(new_partition, self.current_partition)
        # Add the new permutation to the group
        self.add_permutation(perm)
        # Update the current partition
        self.current_partition = new_partition
        # Recompute the Schreier-Sims structure
        self.schreier_sims()
        # print("\nUpdated with new partition.")
        # Optionally, print the updated generators and group elements
        # self.print_generators()
        # self.print_group_elements()

if __name__ == "__main__":
    # Example initial partition
    partition1 = [3, 1, 7, 5, 0, 6, 2, 8, 4]

    # Initialize SchreierSims with n elements
    n = len(partition1)
    schreier_sims = SchreierSims(n)

    # Set the initial partition
    schreier_sims.update_with_new_partition(partition1)

    # Example new partition to update with
    partition2 = [1, 5, 3, 7, 2, 0, 8, 6, 4]

    # Update the group with the new partition
    schreier_sims.update_with_new_partition(partition2)

    # You can continue to update with more partitions as needed
    # For example, updating with another partition
    partition3 = [5, 1, 7, 3, 0, 6, 2, 8, 4]
    schreier_sims.update_with_new_partition(partition3)

