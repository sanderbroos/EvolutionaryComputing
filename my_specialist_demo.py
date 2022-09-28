import multiprocessing
import sys
from operator import itemgetter

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import random
from deap import creator, tools, algorithms, base

import numpy as np
import os

run_mode = 'train'  # train or test

experiment_name = 'my_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if run_mode != 'test':
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
dom_u = 1
dom_l = -1
population_number = 50
generation_count = 30
mutation_rate = 0.15
mutation_sigma = 1

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x):
    return simulation(env, x),


# normalizes
def normalize(fit):
    max_fitness = max(fit, key=itemgetter(0))[0]
    min_fitness = min(fit, key=itemgetter(0))[0]

    r = max_fitness - min_fitness
    normal = map(lambda x: ((x[0] - min_fitness) / r,), fit)
    return normal


if __name__ == '__main__':

    if run_mode == 'test':
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate(bsol)
        sys.exit(0)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("network_weight", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.network_weight, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_sigma, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_number)

    for generation in range(generation_count):
        print("###### Current generation:", generation)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.9)
        fits = toolbox.map(toolbox.evaluate, offspring)
        fits = list(normalize(fits))
        # print(fits)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top = tools.selBest(population, k=1)
    np.savetxt(experiment_name + '/best.txt', top)
