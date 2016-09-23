import numpy as np
import csv
from itertools import permutations, cycle
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns


file = open('european_cities.csv', 'r+')
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

# -----------------------------------------------------------------------------------------

#start_time = time.time()

#N = 6 # number of cities
#A = distance[0:N,0:N]
#elapsed_time = np.zeros(5)
#N = np.zeros(5)


for i in range(5):
    start_time = time.time()
    
    N = 6 + i

    city_vec  = np.arange(N)

    def dist(cities,distance_matrix):
        tour = np.zeros(len(cities))
        for i in range(len(cities)-1):
            tour[i] = distance_matrix[int(cities[i]),int(cities[i+1])]
            tour[-1] = distance_matrix[int(cities[-1]), int(cities[0])]
        return np.sum(tour)
    
    #print a
    #print dist(a)
    #print "\n"
    
    best_sequence = np.zeros(len(cities))
    best = 100000000
    for i in permutations(city_vec):
        temp = dist(i,distance)
        if temp < best:
            best = temp
            best_sequence = i

        
    elapsed_time =  time.time() - start_time

    print "The elapsed time for N =", N, " is ", elapsed_time, " seconds."
    
print "\n This is the shortest tour distance ", best, "\n"
#print best_sequence
#print ("Time elapsed time is %f seconds" %(elapsed_time))


def index_to_city_name(city_names, city_indices):
    best_cities = []
    for i in range(len(city_indices)):
        best_cities.append(city_names[int(city_indices[i])])
    return best_cities

print "\n The best travel route is as follows", index_to_city_name(cities, best_sequence), "\n"



"""
The elapsed time for N = 6  is  0.00620198249817  seconds.
The elapsed time for N = 7  is  0.0489280223846  seconds.
The elapsed time for N = 8  is  0.402410984039  seconds.
The elapsed time for N = 9  is  3.91781592369  seconds.
The elapsed time for N = 10  is  44.4848008156  seconds.

 This is the shortest tour distance  7486.31 


 The best travel route is as follows ['Barcelona', 'Dublin', 'Brussels', 'Hamburg', 'Copenhagen', 'Berlin', 'Budapest', 'Bucharest', 'Istanbul', 'Belgrade'] 
"""

