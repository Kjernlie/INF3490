import numpy as np
import csv
from itertools import permutations, cycle
import time
import random

file = open('/home/johannes/Work/INF3490/oblig1/european_cities.csv', 'r+')
reader = csv.reader(file,delimiter=";")

distance = np.zeros((24,24))
cities = []

i = 0
for row in reader:
    #print row
    if i == 0:
        for j in range(len(row)):
            cities.append(row[j])
    else:
        for j in range(len(distance)):
            distance[i-1,j] = float(row[j])
    i += 1


# -------------------------------------------------------------------------------------

def dist(cities,distance_matrix):
    tour = np.zeros(len(cities))
    for i in range(len(cities)-1):
        tour[i] = distance_matrix[int(cities[i]),int(cities[i+1])]
    tour[-1] = distance_matrix[int(cities[-1]), int(cities[0])]
    return np.sum(tour)

# -----------------------------------------------------------------------------------------
# Hill climbing algorithm

def hillclimb(cities, distance_matrix):
    init = cities
    np.random.shuffle(init)
    
    best = dist(init,distance_matrix)
    count = 0
    best_sequence = init.copy()
    while count < 10:        
        number1 = np.random.randint(N)
        number2 = np.random.randint(N)
        
        while number1 == number2:
            number2 = np.random.randint(N)
        
        value1 = init[number1]
        value2 = init[number2]
        
        init[number1] = value2
        init[number2] = value1
        
        temp = dist(init,distance_matrix) 
        
        if temp < best:
            best = temp
            best_sequence = init.copy()
            count = 0
        else:
            count += 1
        
    return best, best_sequence

# ------------------------------------------------------------------------------------------------

def index_to_city_name(city_names, city_indices):
    best_cities = []
    for i in range(len(city_indices)):
        best_cities.append(city_names[int(city_indices[i])])
    return best_cities

# ------------------------------------------------------------------------------------------

N = 24 # Number of cities to include
NoR = 20 # Number of runs
city_vec = np.arange(N)
best = np.zeros(NoR)
best_sequence = np.zeros((NoR,N))

for i in range(NoR):

    best[i], best_sequence[i,:] = hillclimb(city_vec,distance)


print "The best tour is ", np.amin(best)
print "The worst tour is ", np.amax(best)
print "The mean of the tours is ", np.mean(best)
print "The standard deviaton of the tours is ", np.std(best)


# For 10 cities
"""
The best tour is  8009.86
The worst tour is  12177.84
The mean of the tours is  10136.412
The standard deviaton of the tours is  1148.87912796
"""

# For 24 cities
"""
The best tour is  23152.61
The worst tour is  31026.39
The mean of the tours is  28112.458
The standard deviaton of the tours is  2014.36933821
"""
