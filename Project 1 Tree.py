import json
import numpy as np
import time
start = time.time()

# Load resistances between nodes
with open('node_resistances.json', 'r') as f:
    resistances_data = json.load(f)

# Build a dictionary of resistances
resistances = {}
for entry in resistances_data:
    node1 = entry["node1"]
    node2 = entry["node2"]
    resistance = entry["resistance"]
    resistances[(node1, node2)] = resistance

print(f"Resistances are {resistances}")

# Load fixed node voltages
with open('node_voltages.json', 'r') as f:
    voltages_data = json.load(f)

# Create a dictionary for nodes with fixed voltages
fixed_voltages = {}
for entry in voltages_data:
    node = entry["node"]
    voltage = entry["voltage"]
    fixed_voltages[node] = voltage

print(f"Fixed voltages are {fixed_voltages}")

# Figure out the total number of nodes
max_node = 0
for (node1, node2) in resistances.keys():
    max_node = max(max_node, node1, node2)
for node in fixed_voltages.keys():
    max_node = max(max_node, node)

n = max_node
print(f"Total number of nodes is {n}")

# Initialize A and b
A = np.zeros((n, n))
b = np.zeros(n)

def compute_A_matrix(resistances, fixed_voltages, n):
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Build neighbor lists
    neighbors = {}
    for (node1, node2), resistance in resistances.items():
        neighbors.setdefault(node1, []).append((node2, resistance))
        neighbors.setdefault(node2, []).append((node1, resistance))

    # Set up the equations
    for i in range(n):
        node = i + 1  # Adjusting for 1-based node numbering
        if node in fixed_voltages:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = fixed_voltages[node]
        else:
            # For other nodes sum up the conductances to neighbors
            sum_conductance = 0
            for neighbor, resistance in neighbors.get(node, []):
                conductance = 1 / resistance
                sum_conductance += conductance
                neighbor_index = neighbor - 1  # Adjust for zero-based indexing
                A[i, neighbor_index] -= conductance
            A[i, i] = sum_conductance
            # b[i] remains zero for these nodes

    return A, b

def lu_factorization(A):
    n = len(A)
    L = np.eye(n)  
    U = A.copy()   

    for i in range(n):
        if U[i, i] == 0:
            raise ValueError("Zero pivot :(")

        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # Store the factor in L
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y)

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

def solve(A, b):
    L, U = lu_factorization(A)  # LU factorization of A
    y = forward_substitution(L, b)  # Solve Ly = b
    x = backward_substitution(U, y)  # Solve Ux = y
    return x

# Compute A and b
A, b = compute_A_matrix(resistances, fixed_voltages, n)
print('Computed A matrix:')
print(A)
print('Computed b vector:', b)

# Solve
node_voltages = solve(A, b)

# Print the final voltages
print('Node voltages are:')
for i, voltage in enumerate(node_voltages):
    print(f'Node {i+1}: {voltage}')

end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")