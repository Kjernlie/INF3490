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
def mk_tournament(population,size):
    
    tournament_size = population.shape[0]/size
    #if (float(population.shape[0])/size).is_integer()!=True:
     #   print "Change population or tournament size!"
    #print tournament_size
        
    tournaments = np.zeros((size,tournament_size))
    population_i = np.arange(population.shape[0])
    np.random.shuffle(population_i)
    
    count = 0
    for i in range(size):
        for j in range(tournament_size):
            tournaments[i,j] = population_i[count]
            count += 1
            
    
    return tournaments

# ----------------------------------------------------------------------------------------------

# create parents
# takes in the tournaments matrix from mk_tournament, the population, and which tournament you want
def mk_parents(tournaments,population,tournament,distance_matrix):
    
    # create the vectors with city indices of the whole tour
    actual_tournament = np.zeros((tournaments.shape[1],population.shape[1]))
    for i in range(tournaments.shape[0]): 
        actual_tournament[i] = population[int(tournaments[i,int(tournament)])]

    
    
    # Find the parents
    distances = np.zeros(tournaments.shape[0]) # len actual_tour
    for i in range(tournaments.shape[0]): # len actaul_tour
        distances[i] = dist(actual_tournament[:,i],distance_matrix) # acua[i]

    parents = np.argpartition(distances, 2)[:2]
    
    global_coord_parents = np.zeros(len(parents))
    for i in range(len(global_coord_parents)):
        global_coord_parents[i] = tournaments[parents[i],tournament]   #flipped
    
    outcast = np.argpartition(distances,-2)[-1]
    global_coord_outcast = tournaments[outcast,tournament]
    
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
    
    parent1[find_zero1[0]] = N+1
    parent2[find_zero2[0]] = N+1
    
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
            
    
    insert_zero = np.where(child == N+1)
    child[insert_zero[0]] = 0.0
    
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
    for i in range(len(outcasts)):
        new_pop = np.delete(population, outcasts[i], 0)
    for i in range(len(children)):
        #new_pop = np.vstack([new_pop,population[children[i]]])
        new_pop = np.vstack((new_pop,children[i]))

    return new_pop

# ------------------------------------------------------------------------------------------------

# main program....

NoG = 500 # number of generations
population_size = 50 # size of the population
NoC = 10 # number of cities
city_vec = np.arange(NoC)
TournamentSize = 5
NoT = population_size/TournamentSize # Number of tournaments 
Pr = 0.03 # Probability of mutation
population_distances = np.zeros(population_size)
population_avg = np.zeros(NoG)

population = mk_population(population_size,city_vec)
for i in range(NoG):
    tournaments = mk_tournament(population,TournamentSize)
    parents = np.zeros((NoT, 2))
    outcast = np.zeros((NoT, 1))
    child = np.zeros((NoT,len(city_vec)))
    
    for j in range(NoT):
        parents[j], outcast[j] = mk_parents(tournaments,population,j,distance)
        child[j,:] = crossover(population,parents[j])
        child[j] = mutation(child[j],Pr)
    population = new_population(population, child, outcast)
    
    for k in range(population_size):
         population_distances[k] = dist(population[k,:],distance)
            
        
    population_avg[i] = np.average(population_distances)
        
    

plt.plot(population_avg)
plt.show()
