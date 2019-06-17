# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 00:31:33 2019

@author: ae0670
"""
import numpy as np
import sys

"""
Find a subset of specified size k whose sum is closest to some given target

n: number of items
c: capacity
k: cardinality
"""

def knapsack(n, c, k):
    for x in range(1, n+1):
        for y in range(0, c+1):
            for z in range(1, k+1):
                if weight[x-1] > y:
                    T[x][y][z] = T[x-1][y][z]
                else:
                    T[x][y][z] = max(T[x-1][y][z], T[x-1][y-weight[x-1]][z-1]+value[x-1])
    return T[n][c][k]

def reconstruct(n, c, k):
    solution = []
    for i in reversed(range(1, n+1)):
        if T[i][c][k] != T[i-1][c][k]:
            solution.append(array[i-1])
            c -= weight[i-1]
            k -= 1
    return solution

array = np.array([1,4,7,10])
#array = np.array([0.02, 0.04, 0.12, 0.14, 0.17, 0.33, 0.36, 0.38, 0.55, 0.58])
target = 13

s = 100 # scaling factor for fractional values
weight = (array*s).astype(int)
m = max(weight) # additive constant for enforcing exact-k solutions
value = (array*s+m).astype(int)

k = 2
c = sum(weight)
n = array.size
T = [[[0 for _ in range(k+1)] for _ in range(c+1)] for _ in range(n+1)]
knapsack(n, c, k)
print(T)

closest_sum = -sys.float_info.max
best_solution = None
for j in range(c+1):
    if T[n][j][k] >= k*m:
        current_sum = T[n][j][k]-k*m
        if abs(target*s-current_sum) <= abs(target*s-closest_sum):
            solution = reconstruct(n, j, k)
            closest_sum = current_sum
            best_solution = solution
            
print('best solution: {}'.format(best_solution))
print('closest sum: {}'.format(closest_sum/s))



''' ============= ADAPTATION OF KNAPSACK V.1 =============

# Works correctly as a knapsack, but somehow results in a sparse matrix,
# which is not suitable for finding the closest sum because some entries
# are missing. An iterative approach gives a dense matrix.

def knapsack(n, u, k):
    if n == 0 or k == 0:
        return 0
    if T[n][u][k] != 0:
        return T[n][u][k]
    v1 = 0
    v2 = 0
    if array[n-1] <= u:
        v1 = knapsack(n-1, u-array[n-1], k-1)+value[n-1]
    v2 = knapsack(n-1, u, k)
    T[n][u][k] = max(v1, v2)
    return T[n][u][k]
'''



''' ============= CLOSEST SUM V.1 =============

# doesn't work properly: problem with exact-k. Also, the reconstruction
# must be fixed, is it might not work correctly.

def subset_sum(items, target_sum, target_size):
    total_sum = min(np.sum(items), target_sum*2)
    T = np.zeros((len(items)+1, total_sum+1, target_size+1), dtype=int)
    #T[0,items[0],1] = 1
    T[0,items[0],1] = 1
    for i, num in enumerate(items, start=1):
        for j in range(1, total_sum+1):
            for k in range(1, target_size+1):
                if j >= num:
                    T[i,j,k] = T[i-1,j,k] or T[i-1,j-num,k-1]
                else:
                    T[i,j,k] = T[i-1,j,k]
    # find closest
    closest = 0
    best_solution = None
    for j in range(1, total_sum+1):
        if T[items.size,j,target_size] == 1 and abs(target_sum-j) < abs(target_sum-closest):
            #print(f'i={items.size}, j={j}, target_size={target_size}')
            solution = reconstruct(T, items, j, target_size)
            print('solution: {}'.format(solution))
            closest = j
            best_solution = solution
    return best_solution

def reconstruct(T, items, current_sum, target_size):
    subset = []
    n = items.size
    j = current_sum
    
    """
    for i in reversed(range(1, n+1)):
        print(T[i,j,target_size])
        if T[i,j,target_size] != T[i-1,j,target_size]:
            subset.append(items[i-1])
            j -= items[i-1]
    subset.reverse()
    """
    
    while n > 0 and j > 0:
        if not T[n-1, j, target_size]:
            subset.append(items[n-1])
            j -= items[n-1]
        n -= 1
    
    return subset
'''

''' ============= ORDINARY CLOSEST SUM =============

def subset_sum(items, target_sum, target_size):
    total_sum = min(np.sum(items), target_sum*2)
    T = np.zeros((len(items)+1, total_sum+1), dtype=int)
    T[:,0] = 1
    for i, num in enumerate(items, start=1):
        for j in range(1, total_sum+1):
            if j >= num:
                T[i,j] = T[i-1, j-num] or T[i-1, j]
            else:
                T[i,j] = T[i-1, j]
    # find closest
    closest = 0
    best_solution = None
    for j in range(1, total_sum+1):
        if T[i,j] == 1 and abs(target_sum-j) < abs(target_sum-closest):
            solution = reconstruct(T, items, j)
            closest = j
            best_solution = solution
    return best_solution

def reconstruct(T, items, current_sum):
    subset = []
    n = len(items)
    j = current_sum
    while n > 0 and j > 0:
        if not T[n-1, j]:
            subset.append(items[n-1])
            j -= items[n-1]
        n -= 1
    return subset
'''