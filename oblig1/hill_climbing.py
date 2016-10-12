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

def hillclimb(cities, distance_matrix,neighbours):
    init = cities
    np.random.shuffle(init)
    
    best = dist(init,distance_matrix)
    best_neighbour = best.copy()
    count = 0
    best_sequence = init.copy()
    while count < 20:   
	# Check more neighbours neighbours and see if they are better 
	for i in range(neighbours):     
		number1 = np.random.randint(N)
		number2 = np.random.randint(N)
		
		while number1 == number2:
		    number2 = np.random.randint(N)
		
		value1 = init[number1]
		value2 = init[number2]
		
		init[number1] = value2
		init[number2] = value1
		
		temp_neighbour = dist(init,distance_matrix)

		if temp_neighbour < best_neighbour:
			best_neighbour = temp_neighbour
 
        
        if best_neighbour < best:
            best = best_neighbour
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

N = 10 # Number of cities to include
NoR = 20 # Number of runs
NoN = 5 # Number of neighbours to check
city_vec = np.arange(N)
best = np.zeros(NoR)
best_sequence = np.zeros((NoR,N))

start_time = time.time()

for i in range(NoR):

    best[i], best_sequence[i,:] = hillclimb(city_vec,distance,NoN)

elapsed_time = time.time() - start_time

print "The elapsed time for ", N, " cities, is ", elapsed_time, " seconds."

print "The best tour is ", np.amin(best)
print "The worst tour is ", np.amax(best)
print "The mean of the tours is ", np.mean(best)
print "The standard deviaton of the tours is ", np.std(best)


# For 10 cities
"""
Elapsed time for  10  cities, is  0.0475599765778  seconds.
The best tour is  7503.1
The worst tour is  10180.55
The mean of the tours is  8922.212
The standard deviaton of the tours is  741.616037971
"""

# For 24 cities
"""
Elapsed time for  24  cities, is  0.0690491199493  seconds.
The best tour is  21784.42
The worst tour is  27723.41
The mean of the tours is  25650.3525
The standard deviaton of the tours is  1471.6979365
"""
