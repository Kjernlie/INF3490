import numpy as np
import csv
from itertools import permutations, cycle
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns


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

# -------------------------------------------------------------------------------------------------

def index_to_city_name(city_names, city_indices):
    best_cities = []
    for i in range(len(city_indices)):
        best_cities.append(city_names[int(city_indices[i])])
    return best_cities

# -------------------------------------------------------------------------------------------------

def dist(cities,distance_matrix):
    tour = np.zeros(len(cities))
    for i in range(len(cities)-1):
        tour[i] = distance_matrix[int(cities[i]),int(cities[i+1])]
    tour[-1] = distance_matrix[int(cities[-1]), int(cities[0])]
    return np.sum(tour)

# -----------------------------------------------------------------------------------------------

# Create population
def mk_population(size,cities):
    population = np.zeros((size,len(cities)))
    for i in range(size):
        np.random.shuffle(cities) 
        population[i] = cities
    return population


# ----------------------------------------------------------------------------------------------

# Create tournament group
def mk_tournament(population,size,number_tournaments):
       
    tournaments = np.zeros((size,number_tournaments))
    population_i = np.arange(population.shape[0])
    np.random.shuffle(population_i)
    
    count = 0
    for i in range(size):
        for j in range(number_tournaments):
            tournaments[i,j] = population_i[count]
            count += 1
    
    return tournaments

# ----------------------------------------------------------------------------------------------

# create parents
# takes in the tournaments matrix from mk_tournament, the population, and which tournament you want
def mk_parents(tournaments,population,tournament,distance_matrix):


    # create the vectors with city indices of the whole tour
    actual_tournament = np.zeros((tournaments.shape[0],population.shape[1]))
    for i in range(tournaments.shape[0]):
        actual_tournament[i] = population[int(tournaments[i,int(tournament)])]

    # Find the length of the actual tour
    distances = np.zeros(tournaments.shape[0]) 
    for i in range(tournaments.shape[0]):
	distances[i] = dist(actual_tournament[i],distance_matrix) 
        
    # Find the parents
    parents = np.argpartition(distances, 2)[:2]
 
    global_coord_parents = np.zeros(len(parents))
    for i in range(len(global_coord_parents)):
        global_coord_parents[i] = tournaments[parents[i],int(tournament)]   #flipped
    outcast = np.argpartition(distances,-2)[-1]
    global_coord_outcast = tournaments[outcast,int(tournament)]


    return global_coord_parents, global_coord_outcast

    
# ----------------------------------------------------------------------------------------

# PMX (Partially mapped crossover)

def crossover(population, parents):
    parent1 = np.zeros(population.shape[1])
    parent2 = np.zeros(population.shape[1])
    parent1 = population[parents[0],:]
    parent2 = population[parents[1],:]
    N = len(parent1)
    child = np.zeros(N)
    
    
    find_zero1 = np.where(parent1 == 0.0)
    find_zero2 = np.where(parent2 == 0.0)
    
    parent1[find_zero1[0]] = N
    parent2[find_zero2[0]] = N
    
     # Creating segment size and start position
    segment_size = np.random.randint(len(parent1))  # The actual size is plus one
    segment_start_pos = np.random.randint(len(parent1)-segment_size) 
    
    # Copy segment from parent1 to child
    segment = np.zeros(segment_size+1)
    segment = parent1[segment_start_pos:segment_start_pos+segment_size+1]
    child[segment_start_pos:segment_start_pos+segment_size+1] = segment
    
    
    for i in range(len(child)):
        if not parent2[i] in child:
            mv_to = np.where(parent2 == parent2[i])
            while True:
                if child[mv_to[0]] == 0.0:  # have to make this zero
                    child[mv_to[0]] = parent2[i]
                    break
                else:
                    mv_to = np.where(parent2 == child[mv_to[0]])
    
    for i in range(N):
        if child[i] == 0.0:  # have to find a better way for this...
            child[i] = parent2[i]
            
    
    insert_zero = np.where(child == N)
    child[insert_zero[0]] = 0.0
    parent1[find_zero1[0]] = 0.0
    parent2[find_zero2[0]] = 0.0   
 
    return child
                    

# ----------------------------------------------------------------------------------------------


# Mutation

def inversion(child):
    segment_size = np.random.randint(len(child))+1
    start_pos = np.random.randint(len(child)-(segment_size-1))
    end_pos = start_pos + segment_size
    segment = child[start_pos:end_pos]
    
    reverse_segment = segment[::-1]
    
    child[start_pos:end_pos] = reverse_segment
    
    return child
       
def decision(probability):
    return random.random() < probability
    
    
def mutation(child,probability):
    if decision(probability) == True:
        child = inversion(child)
    return child
        
    
# ----------------------------------------------------------------------------------------------

# Create new population

def new_population(population, children, outcasts):
    new_pop = np.zeros(population.shape)
    new_pop = population.copy()
    outcasts = np.sort(outcasts, axis = 0)
    count = 0
    for i in range(len(outcasts)):
        new_pop = np.delete(new_pop, outcasts[i]-count, 0)
        count += 1
    for i in range(len(children)):
        new_pop = np.vstack((new_pop,children[i]))
    return new_pop

# ------------------------------------------------------------------------------------------------

# main program....

NoG = 150 # number of generations
population_size = [50, 100, 150] # size of the population
NoC = 24 # number of cities
city_vec = np.arange(NoC)
TournamentSize = 5
Pr = 0.03 # Probability of mutation
population_avg = np.zeros((3,NoG))

for m in range(len(population_size)):
	population = mk_population(population_size[m],city_vec)
	NoT = int(population_size[m]/TournamentSize) # Number of tournaments
	population_distances = np.zeros(population_size[m])
       	for i in range(NoG):
	    tournaments = mk_tournament(population,TournamentSize,NoT)
	    parents = np.zeros((NoT, 2))
	    outcast = np.zeros((NoT, 1))
	    child = np.zeros((NoT,len(city_vec)))
	 
	    for j in range(NoT):
		parents[j], outcast[j] = mk_parents(tournaments,population,j,distance)
		child[j] = crossover(population,parents[j])
		child[j] = mutation(child[j],Pr)
	  
	    population = new_population(population, child, outcast)       

	    for k in range(population_size[m]):
		population_distances[k] = dist(population[k,:],distance)
		    
		
	    population_avg[m,i] = np.average(population_distances)
	


	# print out the best result, worst result, mean and standard deviation of the results
        print "The best tour is ", np.amin(population_avg[m]), " with population size ", population_size[m]
 	print "The worst tour is ", np.amax(population_avg[m]), " with population size ", population_size[m]
	print "The mean of the tours is ", np.mean(population_avg[m]), " with population size ", population_size[m] 
	print "The standard deviaton of the tours is ", np.std(population_avg[m]), " with population size ", population_size[m]
    

# plot the results
plt.plot(population_avg[0],label='Size of population is 50')
plt.plot(population_avg[1],label='Size of population is 100')
plt.plot(population_avg[2],label='Size of population is 150')
plt.legend(loc='best')
plt.xlabel('Generations')
plt.ylabel('Route length [km]')
plt.show()


# ------------------------------------------------------------------------------------
# Terminal output
# For a population of 50 and  10 cities
"""
The best tour is  7791.86  with population size  50
The worst tour is  11867.341  with population size  50
The mean of the tours is  8044.692224  with population size  50
The standard deviaton of the tours is  749.934471998  with population size  50
"""

# For a population of 100 and  10 cities
"""
The best tour is  7486.31  with population size  100
The worst tour is  12213.2522  with population size  100
The mean of the tours is  7808.15731933  with population size  100
The standard deviaton of the tours is  899.006471627  with population size  100
"""

# For a population of 150 and  10 cities
"""
The best tour is  7486.31  with population size  150
The worst tour is  12129.9804667  with population size  150
The mean of the tours is  7840.38100356  with population size  150
The standard deviaton of the tours is  871.253094014  with population size  150
"""

# For a population of 50 and 24 cities
"""
The best tour is  18733.29  with population size  50
The worst tour is  30841.667  with population size  50
The mean of the tours is  21193.6299507  with population size  50
The standard deviaton of the tours is  2794.14361598  with population size  50
"""

# For a population of 100 and 24 cities
"""
The best tour is  18128.48  with population size  100
The worst tour is  30562.6222  with population size  100
The mean of the tours is  20304.82253  with population size  100
The standard deviaton of the tours is  2755.96455571  with population size  100
"""

# For a population of 150 and 24 cities
"""
The best tour is  17606.39  with population size  150
The worst tour is  30633.8865333  with population size  150
The mean of the tours is  19643.4409827  with population size  150
The standard deviaton of the tours is  2948.51425662  with population size  150
"""
