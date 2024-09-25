import json
import time
import numpy as np

def read_resistances_json(file_name):
    resistances = {}
    with open(file_name, 'r') as f:
        data = json.load(f)
        for entry in data:
            node1 = entry["node1"]
            node2 = entry["node2"]
            resistance = entry["resistance"]
            resistances[(node1, node2)] = resistance
    return resistances

def read_fixed_voltages_json(file_name):
    fixed_voltages = {}
    with open(file_name, 'r') as f:
        data = json.load(f)
        for entry in data:
            node = entry["node"]
            voltage = entry["voltage"]
            fixed_voltages[node] = voltage
    return fixed_voltages

#Based on the recitation method for computing the A matrix
def compute_A(resistances, fixed_voltages, num_nodes=25):
    A = np.zeros((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    for i in range(1, num_nodes + 1):
        # Check if node is in fixed voltages
        if i in fixed_voltages:
            A[i-1, i-1] = 1
            b[i-1] = fixed_voltages[i]
        else:
            # Get rows that are neighbors of current node
            json_rows = [(key, value) for key, value in resistances.items() if i in key]

            sum_resistances = 0
            for (node_pair, curr_resistance) in json_rows:
                # Find the neighboring node
                neighbor_node = node_pair[0] if node_pair[1] == i else node_pair[1]

                # Update A matrix for neighbors
                if curr_resistance > 0:
                    A[i-1][neighbor_node-1] = -1 / curr_resistance
                    sum_resistances += 1 / curr_resistance

            # Set diagonal element with sum of conductances
            A[i-1][i-1] = sum_resistances

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
            L[j, i] = factor  
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    U = U.astype(float)
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

def solve_currents(voltages, resistances):
    currents = {}
    for (node1, node2), resistance in resistances.items():
        current = (voltages[node1 - 1] - voltages[node2 - 1]) / resistance
        currents[(node1, node2)] = float(current)

    return currents


def write_output_to_file(A, L, U, node_voltages, currents, file_name):
    with open(file_name, 'w') as f:
        # Write A matrix
        f.write("A matrix:\n")
        f.write(np.array2string(A, precision=2, separator=', ') + "\n\n")
        
        # Write LU factorization
        f.write("L matrix:\n")
        f.write(np.array2string(L, precision=2, separator=', ') + "\n\n")
        
        f.write("U matrix:\n")
        f.write(np.array2string(U, precision=2, separator=', ') + "\n\n")
        
        # Write node voltages
        f.write("Node Voltages:\n")
        for i, voltage in enumerate(node_voltages, start=1):
            f.write(f"Node {i}: {voltage:.2f} V\n")
        f.write("\n")
        
        # Write currents each link
        f.write("Currents through each link:\n")
        for (node1, node2), current in currents.items():
            f.write(f"Current between Node {node1} and Node {node2}: {current:.2f} A\n")
        f.write("\n")


fixed_voltages = read_fixed_voltages_json('node_voltages.json')
resistances = read_resistances_json('node_resistances.json')
          
                
start = time.time()
A, b = compute_A(resistances, fixed_voltages, 25)

L, U = lu_factorization(A)

final_voltages = solve(A, b)

currents = solve_currents(final_voltages, resistances)

end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")

output_file_name = "project1GridResults.txt"
write_output_to_file(A, L, U, final_voltages, currents, output_file_name)
