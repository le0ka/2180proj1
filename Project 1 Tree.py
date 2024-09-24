import json
import numpy as np

# Load resistances between nodes from 'node_resistances.json'
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

# Load fixed node voltages from 'node_voltages.json'
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

# Initialize A (the system matrix) and b (the right-hand side vector)
A = np.zeros((n, n))
b = np.zeros(n)

# Function to compute A matrix and b vector
def compute_A_matrix(resistances, fixed_voltages, n):
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Build neighbor lists for each node
    neighbors = {}
    for (node1, node2), resistance in resistances.items():
        neighbors.setdefault(node1, []).append((node2, resistance))
        neighbors.setdefault(node2, []).append((node1, resistance))

    # Set up the equations for each node
    for i in range(n):
        node = i + 1  # Adjusting for 1-based node numbering
        if node in fixed_voltages:
            # For nodes with fixed voltages, set A[i, i] = 1 and b[i] to the voltage
            A[i, :] = 0
            A[i, i] = 1
            b[i] = fixed_voltages[node]
        else:
            # For other nodes, sum up the conductances to neighbors
            sum_conductance = 0
            for neighbor, resistance in neighbors.get(node, []):
                conductance = 1 / resistance
                sum_conductance += conductance
                neighbor_index = neighbor - 1  # Adjust for zero-based indexing
                A[i, neighbor_index] -= conductance
            A[i, i] = sum_conductance
            # b[i] remains zero for these nodes

    return A, b

# Compute A and b using the function
A, b = compute_A_matrix(resistances, fixed_voltages, n)
print('Computed A matrix:')
print(A)
print('Computed b vector:', b)

# Solve the system of equations to find node voltages
node_voltages = np.linalg.solve(A, b)

# Print the voltages at each node
print('Node voltages are:')
for i, voltage in enumerate(node_voltages):
    print(f'Node {i+1}: {voltage}')
