import json

import numpy as np
import scipy.linalg as la
import time
start = time.time()


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

fixed_voltages = read_fixed_voltages_json('node_voltages.json')
resistances = read_resistances_json('node_resistances.json')



import numpy as np

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
            # No need to update b[i-1], already initialized to 0

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





            
                

A, b = compute_A(resistances, fixed_voltages, 25)
#print(A)
#print(b)

L, U = lu_factorization(A)
#print(L)
#print(U)

#P, L, U = la.lu(A)


x = np.linalg.solve(A, b)
print(x)

end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
