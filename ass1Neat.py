################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
'''
Below code makes use of code from Neat package https://github.com/CodeReclaimers/neat-python
'''

# TODO;
# Do I need to give env different parameters, such as player controller?
# is the input and output of network correct? demo gives pop as action for play
# check optimization_specialist demo to see whats missing, what fitness function they use

import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
from demo_controller import player_controller
import neat
import visualize
import pickle

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name) 

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        input = env.player.sensors.get(env)
        print('input', input)
        action = net.activate(input)
        print('action',action)
        result = env.play(pcont=action)
        #print('result',result)
        fitness = result[0]
        #print('fitness',fitness)
        genome.fitness = fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9') # if want to begin at checkpoint
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # creates checkpoints

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 100)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # Save the winner.
    # with open('winner-feedforward', 'wb') as f:
    #     pickle.dump(winner, f)

    # creats plots of results
    visualize.plot_stats(stats, ylog=False, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)