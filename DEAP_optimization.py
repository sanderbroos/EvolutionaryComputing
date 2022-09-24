from copy import copy
import multiprocessing
import sys
from tracemalloc import Statistic

sys.path.insert(0, 'evoman/')
from environment import Environment
from demo_controller import player_controller
from deap import creator, tools, base

import random
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# enemy_id
enemy_id = 1

run_mode = 'train'  # train or test

experiment_name = 'biased_mating_probobility'
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
population_number = 100
NGEN = 30
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

# Define statistics, all of the statistics will be stored in logBook
s = tools.Statistics(key=lambda ind: ind.fitness.values[0])
s.register('mean', np.mean)
s.register('std', np.std)
s.register('max', max)

logbook = tools.Logbook()

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

def main():
    pop = toolbox.population(n=population_number)
    MUTPB = 0.2

    # Evaluate the entire population
    fitnesses = np.array(toolbox.map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):

        '''select the offspring based on fitness values (biased probobility)'''
        # scales the fitness values between (0,1) to avoid negative probability
        # for every generation
        scaler = MinMaxScaler(copy=True)
        fitness_norm = scaler.fit_transform(fitnesses).flatten()

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = toolbox.map(toolbox.clone, offspring)

        for index_1, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):

            child1_prob, child2_prob = fitness_norm[index_1], \
                                   fitness_norm[index_1+1]

            # Why using OR here? Because the mating procedure should be dominated by
            # the parent with large fitness value. i.e. The good genes are more likely
            # to be passed on. (my point of view).
            if (random.random() < child1_prob) or (random.random() < child2_prob):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate all the individuals
        # Note!!! This will slow down the algorithm speed, but it is essential
        # if you want to set the probability of mating based on ftness value
        fitnesses = np.array(toolbox.map(toolbox.evaluate, offspring))

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # record statistics
        record = s.compile(pop)
        logbook.record(gen=g, mean=record['mean'], std=record['std'], max=record['max'])

    return pop, logbook


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

    ################## EVOLVING DEAP MODEL ##################
    pop, logbook = main()
    
    # select the best solution
    top = tools.selBest(pop, k=1)
    np.savetxt(experiment_name + f'/best_{enemy_id}.txt', top)

    # save logBook
    with open(f'logBook/best_{enemy_id}_{experiment_name}_logBook.pkl', 'wb') as f:
        pickle.dump(logbook, f)