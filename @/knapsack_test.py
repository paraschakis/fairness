# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:14:55 2019

@author: ae0670
"""

from collections import defaultdict
import numpy as np
import heapq
import time

def knapsack(n, k, l, u, v, w):
    """
    3-dimensional knapsack with cardinality and capacity constraints
    
    Inputs:
        n: number of items
        v: vector of item values
        w: vector of item weights
        k: cardinality of solution
        l: lower bound on capacity
        u: upper bound on capacity
    """
    
    if n == 0 or k == 0:
        return 0
    if T[n,u,k] != 0:
        return T[n,u,k]
    v1 = 0
    v2 = 0
    if w[n-1] <= u and \
        sum(heapq.nlargest(k-1, w[:n-1])) >= l-w[n-1]:
        v1 = knapsack(n-1, k-1, l-w[n-1], u-w[n-1], v, w)+v[n-1]
    if sum(heapq.nlargest(k, w[:n-1])) >= l:
        v2 = knapsack(n-1, k, l, u, v, w)
    #if v1 > 0 or v2 > 0:
    T[n,u,k] = max(v1, v2)         
    return T[n,u,k]

# doesn't work yet. also seems to be slower
def knapsack_iterative(n, k, l, u, v, w):
    for x in range(1, n+1):
        for y in range(0, u+1):
            for z in range(1, k+1):
                v1 = 0
                v2 = 0
                if w[x-1] <= y and \
                sum(heapq.nlargest(k-z-1, w[x+1:])) >= l-w[x-1]:
                    v1 = T[x-1,y-w[x-1],z-1]+v[x-1]
                if sum(heapq.nlargest(k-z, w[x+1:])) >= l:
                    v2 = T[x-1,y,z]
                if v1 > v2:
                    l -= w[x-1]
                    T[x,y,z] = v1
                else:
                    T[x,y,z] = v2
    return T[n,u,k]

#value = np.array([0.4,0.3,0.2,0.1,0.01,-0.01,-0.1,-0.2,-0.3,-0.4])
#weight = np.array([100,90,80,70,60,50,40,30,20,10])
value = np.array([0.9,0.8,0.7,0.6,0.51,0.49,0.40,0.30,0.20,0.10])
#value = np.array([1,2,3,7,5,6,4,8,9,10])
#value = np.array([10,9,8,7,6,5,4,3,2,1])
weight = np.array([0,0,1,0,0,1,1,0,1,1])

#value = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#value = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
#weight = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

lower_capacity = 2
upper_capacity = 3

#value = np.array([1,4,7,10])
#weight = np.array([1,4,7,10])

#value = np.array([9,8,7,6,5,4,3,2,1])
#weight = np.array([9,8,7,6,5,4,3,2,1])
#lower_capacity = 1
#upper_capacity = 10

#value = np.array([6,4,5,3,7])
#weight = np.array([4,3,5,7,7])
#lower_capacity = 7
#upper_capacity = 18

#value = np.array([0.02, 0.04, 0.12, 0.14, 0.17, 0.33, 0.36, 0.38, 0.55, 0.58])
#weight = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])



k = 5 # cardinality
n = weight.size


# transformation for exact-k v.1: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.175&rep=rep1&type=pdf
m = sum(value)
v = value + m
w = (weight*10).astype(int)
u = int(round(upper_capacity*10))
l = int(round(lower_capacity*10))


'''
# transformatio for exact-k v.2: from https://books.google.de/books?id=u5DB7gck08YC&pg=PA272&lpg=PA272&hl=de#v=onepage&q&f=false
transform_input = lambda a: a+sum(a)
transform_capacity = lambda c: int(round(k*sum(weight*100)+c)) 
l = transform_capacity(lower_capacity)
u = transform_capacity(upper_capacity)
v = transform_input((value*100).astype(int))
w = transform_input((weight*100).astype(int))     
'''

'''
T = [[[0 for _ in range(k+1)] # knapsack cardinality
            for _ in range(u+1)] # knapsack capacity
                for _ in range(n+1)] # number of items
  
'''
start = time.time()
T = np.zeros((n+1,u+1,k+1), dtype=float)

knapsack_iterative(n, k, l, u, v, w)
#print(T)

# reconstruct
solution = []
#if T[weight.size][upper_capacity][k] >= k*m: #v.1

if T[n,u,k] >= k*sum(value): #check if solution exists
    for n in reversed(range(1, weight.size+1)):
        if T[n,u,k] != T[n-1,u,k]:
            solution.append(value[n-1])
            u -= w[n-1]
            k -= 1
print(solution)
print('runtime: {}'.format(time.time()-start))


#print(T[0][capacity][k])


'''
 # ========= Testing all cardinalities with 1 table (doesn't work)  =========
def knapsack1(n, k, u, v, w):
    if n == 0 or k == 0:
        return 0
    if T[n][u][k] != 0:
        return T[n][u][k]
    if w[n-1] > u:
        T[n][u][k] = knapsack1(n-1, k, u, v, w)
    else:
        v1 = knapsack1(n-1, k-1, u-w[n-1], v, w) + v[n-1]
        v2 = knapsack1(n-1, k, u, v, w)
        T[n][u][k] = max(v1, v2)
    return T[n][u][k]

lower_capacity = 0
upper_capacity = 10
k = 10
knapsack1(n, k, u, v, w)

#if T[n][u][k] >= k*m: #check if solution exists
for n in reversed(range(1, weight.size+1)):
    if T[n][u][k] != T[n-1][u][k]:
        solution.append(value[n-1])
        u -= w[n-1]
        k -= 1
print(solution)

'''

'''
 # ========= Classical Knapsack, iterative and recursive =========

# with memoization
def knapsack_recursive(n, w):
    if n < 0:
        return 0
    if T[n][w] != 0:
        return T[n][w]
    if weight[n-1] > w:
        T[n][w] = knapsack_recursive(n-1, w)
    else:
        T[n][w] = max(knapsack_recursive(n-1, w), 
                 knapsack_recursive(n-1, w-weight[n-1]) + value[n-1])
    return T[n][w]

def knapsack_iterative(n, w):
    for i in range(1, n+1):
        for j in range(0, w+1):
            if weight[i-1] > j:
                T[i][j] = T[i-1][j]
            else:
                T[i][j] = max(T[i-1][j], T[i-1][j-weight[i-1]]+value[i-1])
    return T[n][w]
    
value = np.array([10,9,8,7,6,5,4,3,2,1])
weight = np.array([0,0,1,0,0,1,1,0,1,1])
w = 1
n = 10

T = [[0 for i in range(w+1)] for j in range(n+1)]
print (knapsack_recursive(n, w))
print (T)

# reconstruction
j = w
solution = []
for i in reversed(range(1, weight.size+1)):
    if T[i][j] != T[i-1][j]:
        solution.append(value[i-1])
        j -= weight[i-1]
print(solution)

# =======================
'''


'''
# ========= v.2 (with two recursive functions) =========
# adapted from https://ideone.com/wKzqXk
def solve(item, current_weight, items_left, lower_bound):
    if item == weight.size or items_left == 0:
        return 0
    if T[item][current_weight][items_left] != 0:
        return T[item][current_weight][items_left]
    v1 = 0
    v2 = 0
    if weight[item] <= current_weight and \
        sum(heapq.nlargest(items_left-1, weight[item+1:])) >= lower_bound-weight[item]:
        v1 = solve(item+1, current_weight-weight[item], items_left-1, 
                   lower_bound-weight[item]) + value[item]
    if sum(heapq.nlargest(items_left, weight[item+1:])) >= lower_bound:
        v2 = solve(item+1, current_weight, items_left, lower_bound)
    if v1 > 0 or v2 > 0:
        T[item][current_weight][items_left] = max(v1, v2)
    return T[item][current_weight][items_left]

def print_knapsack(item, current_weight, items_left, lower_bound):
    if item == weight.size or items_left == 0:
        return 0
    v1 = 0
    v2 = 0
    if weight[item] <= current_weight and \
        sum(heapq.nlargest(items_left-1, weight[item+1:])) >= lower_bound-weight[item]:
        v1 = solve(item+1, current_weight-weight[item], items_left-1, 
                       lower_bound-weight[item]) + value[item]
    if sum(heapq.nlargest(items_left, weight[item+1:])) >= lower_bound:
        v2 = solve(item+1, current_weight, items_left, lower_bound)      
    if v1 >= v2: #and v1 > 0:
        knapsack.append(item)
        #if T[item][current_weight][items_left] == 0:
        T[item][current_weight][items_left] = v1
        #knapsack_weight += weight[item]
        #knapsack_value += value[item]
        return print_knapsack(item+1, current_weight-weight[item], items_left-1, 
                              lower_bound-weight[item])
    else:
        #if T[item][current_weight][items_left] == 0:
        T[item][current_weight][items_left] = v2
        return print_knapsack(item+1, current_weight, items_left, 
                              lower_bound)

value += m
#knapsack = []
solve(0, upper_capacity, k, lower_capacity)
#print_knapsack(0, upper_capacity, k, lower_capacity)
#print(value[knapsack]-m) # m is the number added to values to achieve exact-k
print(T)

# reconstruct from the table (working, original)
j = upper_capacity
solution = []
for i in range(weight.size):
    if T[i][j][k] != T[i+1][j][k]:
        solution.append(value[i]-m)
        j -= weight[i]
        k -= 1
print('solution: {}'.format(solution))
'''


# ========= v.2 (without lower bound) =========
'''
# v.2: adapted from https://ideone.com/wKzqXk
def solve(item, current_weight, items_left):
    if item == weight.size or items_left == 0:
        return 0
    if T[item][current_weight][items_left] != -1:
        return T[item][current_weight][items_left]
    v1 = 0
    v2 = 0
    if weight[item] <= current_weight: # and satisfies_lower_bound(item):
        v1 = solve(item+1, current_weight-weight[item], items_left-1) + value[item]
    v2 = solve(item+1, current_weight, items_left)
    T[item][current_weight][items_left] = max(v1, v2)
    return T[item][current_weight][items_left]

def print_knapsack(item, current_weight, items_left):
    if item == weight.size or items_left == 0:
        return 0
    v1 = 0
    v2 = 0
    if weight[item] <= current_weight: #and satisfies_lower_bound(item):
        v1 = solve(item+1, current_weight-weight[item], items_left-1) + value[item]
    v2 = solve(item+1, current_weight, items_left)        
    if v1 >= v2:
        knapsack.append(item)
        #knapsack_weight += weight[item]
        #knapsack_value += value[item]
        return print_knapsack(item+1, current_weight-weight[item], items_left-1)
    else:
        return print_knapsack(item+1, current_weight, items_left)
        
'''




# ========= v.1 =========
# works well, but without adjustment for lower bound 
# adapted from https://gist.github.com/Phovox/127e5923660d60fb7924

'''
def solve3d (capacity, value, weight, maxitems):
#def solve3d (value, weight, capacity=None, maxitems=None):
    
    """
    solve the 3d-knapsack problem specified in its parameters: capacity is the
    overall capacity of the knapsack and the ith position of the arrays value
    and weight specify the value and weight of the ith item. This is called the
    3d-knapsack not because it refers to a cuboid but because it also considers
    a maximum number of items to insert which is given in the last parameter
    """
    
    #D
    if capacity is None:
        capacity = sum(weight)
    if maxitems is None:
        maxitems = weight.size
    
    # initialization - the number of items is taken from the length of any array
    # which shall have the same length
    nbitems = len(value)

    # we use dynamic programming to solve this problem. Thus, we'll need a table
    # that contains (N, M) cells where 0<=N<=capacity and 0<=M<=nbitems
    table = dict ()
    
    # initialize the contents of the dictionary for all capacities and number of
    # items to another dictionary which returns 0 by default
    for icapacity in range (0, 1+capacity):
        table[icapacity] = dict()
        for items in range (0, 1+nbitems):
            table[icapacity][items] = defaultdict(int)

    # now we are ready to start, ... for the first j items
    for j in range (0, nbitems):

        # for all capacities ranging from 1 to the maximum overall capacity
        for i in range (1, 1+capacity):

            # and for all cardinalities of items from 1 until the maximum
            # allowed
            for k in range (1, 1+maxitems):

                # if this item can not be inserted
                if (weight[j] > i):
                    table[i][1+j][k] = table[i][j][k]   # just do not add it
                    
                # otherwise, consider inserting it
                else:
                    
                    # if this is item is known to fit the knapsack (because its
                    # weight does not exceed the current capacity and adding it
                    # creates a set with a cardinality less or equal than the
                    # cardinality currently considered), then compute the
                    # optimal value as usual but considering sets of the same
                    # cardinality (k)
                    if j < k:
                        table[i][1+j][k] = max(table[i][j][k],
                                                 table[i-weight[j]][j][k]+value[j])
                    else:
                        prev = []

                        # retrieve the optimal solution for all values of
                        # (i-weight[j], kappa [0 .. j-1], k-1)
                        for kappa in range(0, 1+j):
                            prev.append(table[i-weight[j]][kappa][k-1])

                        # and consider inserting this item taking into account
                        # the optimal solution from the preceding cardinality
                        # considering all sets of items
                        table[i][1+j][k] = max(table[i][j][k],
                                                 max(prev) + value[j])
                    
    # return the table computed so far
    return table

def print_solution(table, capacity, value, weight, maxitems):
#def print_solution(table, capacity, k, m):
       
    # reconstruct
    i = capacity
    k = maxitems
    subset = []
    subset_weight = []
    
    #z_star = table[capacity][weight.size][k]
    #print ('z*: {}'.format(z_star))
    #print ('km: {}'.format(k*m))
    #print ('z*-km: {}'.format(z_star-k*m))

    z_star = table[capacity][weight.size][k] # best value
    if z_star < k*m:
        print('no solution exists')
    else:
        for j in reversed(range(1, weight.size+1)):
            if table[i][j][k] != table[i][j-1][k]:
                subset.append(value[j-1])
                subset_weight.append(weight[j-1])
                i -= weight[j-1]
                k -= 1
        subset.reverse()
        print('subset: {0}, value: {1}, weight: {2}'.format(subset, sum(subset), sum(subset_weight)))    
    return
    
table = solve3d(capacity, value, weight, k)
print_solution(table, capacity, value, weight, k)    
    
#table = solve3d(value+m, weight)
#print_solution(table, capacity, k, m)
#show3d (capacity, value, weight, table, maxitems)
'''
