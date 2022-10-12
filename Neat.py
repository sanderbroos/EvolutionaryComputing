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
import pandas as pd

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# enemy_id
enemy_id = [1,2]
NGEN = 1
runs = 1
run_mode = 'test'  # train or test

def eval_genomes(genomes,config):
    fitnessvalues = []
    gainvalues = []
    for genome_id, g in genomes:
        results = env.play(pcont=g)
        #print('fitness, playerlife, enemylife, time:',results)
        g.fitness = results[0]
        gainvalues.append(results[1]-results[2])
        fitnessvalues.append(results[0])
    genfitnessv.append(fitnessvalues)
    gengainv.append(gainvalues)

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
    with open('winner-Neat', 'wb') as f:
        pickle.dump(winner, f)

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

    if run_mode == 'test':
        test_runs = 5
        individual_gains = []
        for en in range(2, 9):
            # Disable the visulization for training modes, increasing training speed
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            bsol = pd.read_pickle(r'winner-Neat')
            print('\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('enemies',[en])
            env.update_parameter('multiplemode','no')
            results = env.play(pcont=bsol)
            individual_gains.append(results[1]-results[2])
        print(individual_gains)
        np.savetxt(f'Neat-individual_gain',individual_gains)
        sys.exit(0)

    genfitnessv = []
    gengainv = []
    for g in range(runs):
        run()

    # genfitnessv variable currently isn't set up for multiple runs 

    fit_averages = [np.mean(gen) for gen in genfitnessv]
    fit_max = [max(gen) for gen in genfitnessv]
    gain_averages = [np.mean(gen) for gen in gengainv]
    gain_max = [max(gen) for gen in gengainv]
    plt.plot(fit_averages,'b',label="fitmean")
    plt.plot(fit_max,'b--',label="fitmax")
    plt.title("Neat Fitness Results")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()
    plt.plot(gain_averages,'r',label="gainmean")
    plt.plot(gain_max,'r--',label="gainmax")
    plt.title("Neat Gain Results")
    plt.xlabel("Generations")
    plt.ylabel("Gain")
    plt.show()

