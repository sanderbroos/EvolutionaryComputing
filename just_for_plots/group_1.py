import multiprocessing
import sys

from pandas import read_pickle

sys.path.insert(0, 'evoman/')
from environment import Environment
from demo_controller import player_controller
from deap import creator, tools, base

import random
import numpy as np
import os
import pickle
import pandas as pd


experiment_name = 'NSGA-II'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

run_mode = 'test'
# number of hidden neurons for the neural network in player_controller module
n_hidden_neurons = 10
n_vars = (20 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Evolution settings
population_number = 100
NGEN = 50
CXPB = 0.9
mutation_rate = 0.10
eta_SBX = 10
eta_PM = 10

group = [2,5,1]
group_id = 3

# To get individual player life and enemy life in multi-objective mode
def cons_multi(value):
    return value

def environment(group, mode):
    if mode == 'train':
        # Disable the visulization for training modes, increasing training speed
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = Environment(experiment_name=experiment_name,
                multiplemode="yes",
                enemies=group,
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=2,
                speed="fastest",
                logs='on')
    
    elif mode == 'test':
        # Disable the visulization for training modes, increasing training speed
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = Environment(experiment_name=experiment_name,
                multiplemode="yes",
                enemies=[1,2,3,4,5,6,7,8],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=2,
                speed="fastest",
                logs='on')
        env.cons_multi = cons_multi
    
    else:
        print('Wrong mode name!')
        sys.exit(0)

    return env

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


env = environment(group=group, mode=run_mode)

# runs evoman game simulation
def evaluate(x):
    f, p, e, t = env.play(pcont=x)
    return f,

def main():

    pop = toolbox.population(n=population_number)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # record statistics (for first generation)
    record = s.compile(pop)
    logbook.record(gen=0, mean=record['mean'], std=record['std'], max=record['max'])

    for g in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # Clone the selected individuals
        offspring = toolbox.map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() <= CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)

            del child1.fitness.values
            del child2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, population_number)

        # record statistics
        record = s.compile(pop)
        logbook.record(gen=g, mean=record['mean'], std=record['std'], max=record['max'])

    return pop

if __name__ == '__main__':

    runs = 10

    # # runs evoman game simulation
    # def evaluate(x):
    #     f, p, e, t = env.play(pcont=x)
    #     return f,

    # evaluate the model
    if run_mode == 'test':

        Gains = []
        stat_p = []
        stat_e = []
        
        for run in range(runs):
            success = 0

            bsol = np.loadtxt(f'group_1/NSGA-II/solution/solution_run_1.txt')
            print('\n RUNNING SAVED BEST SOLUTION \n')

            # repeat 5 times(enemy 5 is stochastic)
            gain = []
            playerlife = []
            enemylife = []
            for i in range(5):
                output = env.play(pcont=bsol)
                print(output)
                playerlife.append(output[1])
                enemylife.append(output[2])
                gain.append(sum(output[1]) - sum(output[2]))
            
            playerlife = np.mean(np.array(playerlife), axis=0)
            enemylife = np.mean(np.array(enemylife), axis=0)

            stat_p.append(np.array(playerlife))
            stat_e.append(np.array(enemylife))
        
            # take the average
            Gains.append(sum(gain)/len(gain))
        
        Gains = np.array(Gains).reshape(len(Gains), 1)

        if not os.path.exists(f'bsol/'):
            os.makedirs(f'bsol/')

        # with open(f'group_1/{experiment_name}/solution/Gain',"w") as f:
        #     for (Gains,stat_p,stat_e) in zip(Gains,stat_p,stat_e):
        #         f.write("{0},{1},{2}\n".format(Gains,stat_p,stat_e))
        np.savetxt(f'bsol/Gain.txt', Gains)
        np.savetxt(f'bsol/Plife.txt', stat_p)
        np.savetxt(f'bsol/Elife.txt', stat_e)
            
        sys.exit(0)
    
    ################## INITIALIZATION OF DEAP MODEL ################## 

    for run in range(runs):
        # set multiprocessing cores
        pool = multiprocessing.Pool()

        toolbox.register("map", pool.map) # multiprocessing
        toolbox.register("network_weight", random.uniform, -1, 1) # randomly initialze network weights
        # create population
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.network_weight, n=n_vars)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # create operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selNSGA2)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=eta_SBX, low=-1, up=1)
        # To apply adaptive mutation step size, this is defined in the main function
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta_PM, low=-1, up=1, indpb=mutation_rate)
        
        ################## EVOLVING DEAP MODEL ##################

        pop = main()

        if not os.path.exists(f'group_1/{experiment_name}/logbook'):
            os.makedirs(f'group_1/{experiment_name}/logbook')

        if not os.path.exists(f'group_1/{experiment_name}/solution'):
            os.makedirs(f'group_1/{experiment_name}/solution')

        # save logBook
        with open(f'group_1/{experiment_name}/logbook/logBook_run_{run}.pkl', 'wb') as f:
            pickle.dump(logbook, f)
        
        # select the best solution
        top = tools.selBest(pop, k=1)
        np.savetxt(f'group_1/{experiment_name}/solution/solution_run_{run}.txt', top)

        logbook.clear()

