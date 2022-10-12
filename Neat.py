################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
'''
Below code makes use of code from Neat package https://github.com/CodeReclaimers/neat-python
'''

import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
from neat_controller import player_controller
import neat
import visualize
import pickle
from matplotlib import pyplot as plt

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# enemy_id
enemy_id = [1,2]
NGEN = 20
runs = 1

def eval_genomes(genomes,config):
    fitnessvalues = []
    for genome_id, g in genomes:
        results = env.play(pcont=g)
        print('fitness, playerlife, enemylife, time:',results)
        g.fitness = results[0]
        fitnessvalues.append(results[0])
    genfitnessv.append(fitnessvalues)

def run():
    # Create the population, which is the top-level object for a NEAT run.
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9') # if want to begin at checkpoint
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # creates checkpoints

    # Run for up to NGEN generations.
    winner = p.run(eval_genomes, NGEN)
    
    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))
    # Save the winner.
    # with open('winner-Neat', 'wb') as f:
    #     pickle.dump(winner, f)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # initializes environment with ai player using random controller, playing against static enemy
    # create an Evoman environment
    env = Environment(experiment_name=experiment_name,
        enemies=enemy_id,
        playermode="ai",
        player_controller=player_controller(config),
        enemymode="static",
        level=2,
        speed="fastest",
        multiplemode="yes",
        logs='off')

    genfitnessv = []
    for g in range(runs):
        run()

    # genfitnessv variable currently isn't set up for multiple runs 

    fit_averages = [np.mean(gen) for gen in genfitnessv]
    fit_max = [max(gen) for gen in genfitnessv]
    plt.plot(fit_averages,label="mean")
    plt.plot(fit_max,'--',label="max")
    plt.title("Neat Results")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()