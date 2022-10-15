################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
'''
Below code makes use of code from Neat package https://github.com/CodeReclaimers/neat-python
'''

import sys
from pathlib import Path

sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
import neat
import pickle
from matplotlib import pyplot as plt

# imports other libs
import numpy as np
import os
import pandas as pd

import multiprocessing
from multiprocessing import Pool

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# enemy_id
enemy_group = [1, 2, 5]
NGEN = 50
# run_id
runs = [1]  # needs to be changed to 10
run_mode = 'test'  # train or test

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
                  enemies=enemy_group,
                  playermode="ai",
                  player_controller=player_controller(config),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  multiplemode="yes",
                  logs='off')


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

        self.genfitnessv = []
        self.gengainv = []

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        fitness_values = []
        gain_values = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            results = job.get(timeout=self.timeout)
            genome.fitness = results[0]
            gain_values.append(results[1] - results[2])
            fitness_values.append(results[0])

        self.genfitnessv.append(fitness_values)
        self.gengainv.append(gain_values)

    def draw_graph_for_run(self):
        fit_averages = [np.mean(gen) for gen in self.genfitnessv]
        fit_max = [max(gen) for gen in self.genfitnessv]
        gain_averages = [np.mean(gen) for gen in self.gengainv]
        gain_max = [max(gen) for gen in self.gengainv]
        plt.plot(fit_averages, 'b', label="fitmean")
        plt.plot(fit_max, 'b--', label="fitmax")
        plt.title("Neat Fitness Results")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.show()
        plt.plot(gain_averages, 'r', label="gainmean")
        plt.plot(gain_max, 'r--', label="gainmax")
        plt.title("Neat Gain Results")
        plt.xlabel("Generations")
        plt.ylabel("Gain")
        plt.show()

    def save_result_for_run(self, path):

        with open(os.path.join(path, 'genfitnessv'), 'wb') as f:
            pickle.dump(self.genfitnessv, f)
        with open(os.path.join(path, 'gengainv'), 'wb') as f:
            pickle.dump(self.gengainv, f)


def eval_genome(genome, config):
    return env.play(pcont=genome)


def run(run_id):
    path = f'neat_result/{",".join(str(enemy) for enemy in enemy_group)}/run_{run_id}'
    Path(path).mkdir(parents=True, exist_ok=True)

    # Create the population, which is the top-level object for a NEAT run.
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9') # if want to begin at checkpoint
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # creates checkpoints

    pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Run for up to NGEN generations.
    winner = p.run(pe.evaluate, NGEN)
    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))
    # Save the winner.

    with open(os.path.join(path, 'winner'), 'wb') as f:
        pickle.dump(winner, f)

    pe.save_result_for_run(path)


if __name__ == '__main__':

    if run_mode == 'test':

        def cons_multi(value):
            return value


        env.cons_multi = cons_multi

        bsol = pd.read_pickle(r'neat_result/1,2,5/run_1/winner')
        individual_gains = []
        for en in range(1, 9):
            # Disable the visualization for training modes, increasing training speed
            print('\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('enemies', [en])
            env.update_parameter('multiplemode', 'yes')
            results = env.play(pcont=bsol)
            individual_gains.append(results[1] - results[2])
            # print(results)
        print(individual_gains)
        np.savetxt(f'Neat-individual_gain', individual_gains)
        sys.exit(0)

    for run_id in runs:
        run(run_id)

    # genfitnessv variable currently isn't set up for multiple runs
