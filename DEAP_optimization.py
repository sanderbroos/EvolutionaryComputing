import multiprocessing
import sys

sys.path.insert(0, 'evoman/')
from environment import Environment
from demo_controller import player_controller

import random
from deap import creator, tools, algorithms, base

import numpy as np
import os

# enemy_id
enemy_id = 3

run_mode = 'train'  # train or test

experiment_name = 'DEAP_optimization'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Disable the visulization for training modes, increasing training speed
if run_mode == 'train':
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# number of hidden neurons for the neural network in player_controller module
n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy_id],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# dom_u = 1
# dom_l = -1

# Evolution settings
population_number = 50
generation_count = 20
mutation_rate = 0.20

# runs evoman game simulation
def simulation(x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return simulation(x),  # fitness score returned

# create fitness fucntion and individuals
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# For the usage of numpy.ndarray
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2



if __name__ == '__main__':

    # evaluate the model
    if run_mode == 'test':
        bsol = np.loadtxt(experiment_name + f'/best_{enemy_id}.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate(bsol)
        sys.exit(0)
    # else:
    #     # Disable the visulization for training modes, increasing training speed
    #     os.environ["SDL_VIDEODRIVER"] = "dummy"

    ################## INITIALIZATION OF DEAP MODEL ################## 

    # set multiprocessing cores
    pool = multiprocessing.Pool()

    toolbox.register("map", pool.map) # multiprocessing
    toolbox.register("network_weight", random.uniform, -1, 1) # randomly initialze network weights
    # create population
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.network_weight, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # create operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # set population
    population = toolbox.population(n=population_number)

    ##################  EVOLVING DEAP MODEL ##################
    
    # evolve certain generations
    for generation in range(generation_count):
        print("###### Current generation:", generation)
        
        ''' 
        the crossover algorithm used here only applies on the variation part (crossover 
        and mutation) should be tuned in the future (assignment requirement: 2 EAs)
        '''     
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        # evaluate the offspring fitness
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # select best offsprings
        population = toolbox.select(offspring, k=len(population))
    
    # select the best solution
    top = tools.selBest(population, k=1)
    np.savetxt(experiment_name + f'/best_{enemy_id}.txt', top)