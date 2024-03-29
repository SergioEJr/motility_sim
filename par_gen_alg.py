import numpy as np
import new_sim as sim
import math
import pickle
import matplotlib.pyplot as plt
import time


# utility methods
# takes in a population of candidates of size n. Randomly selects n cells
# weighted by their performance (distance travelled). Return this selection.
def selection(gen_size, s, pop):
    factors = [math.exp(s * cell.distance) for cell in pop]
    weights = np.asarray(factors) / np.sum(factors)
    indices = np.random.choice(pop.size, gen_size, p=weights)
    return np.asarray([pop[i] for i in indices])

# Takes in a population of parents, create new cells that inherit parameters from parents
# with mutations included. 
def offspring(settings, gen_size, parents, pool):
    children = np.asarray([sim.Cell(settings) for _ in range(gen_size)], dtype = 'O')
    new_pop = np.asarray(pool.map(mutate, zip(children, parents)))
    return new_pop

# returns an uninitialized cell object that inherits parameters
# from the given parent
def mutate(pair):
    child = pair[0]
    parent = pair[1]
    child.inherit(parent)
    return child
    
# runs a cells simulation
def cell_run(cell):
    cell.run()
    return cell

# a genetic algorithm object with fixed generation size
class Genetic:
    
    # gen_size and s are hyperparameters of the genetic algorithm
    # settings are the hyperparameters of the cells
    # initial is the initialization point
    def __init__(self, gen_size, s, settings, initial):
        self.gen_size = gen_size
        self.completions = 0
        self.s = s
        self.full_data = None
        # both of these are dictionaries
        valid_init_keys = ('k1', 'k2', 'K', 'm_dens', 'm_cov', 'dir_prob', 'imp_sens')
        valid_settings_keys = ('duration', 'FPS', 'beta', 'budget')
        self.properties = ('dist', 'pol', 'avg_k', 'mot_dens', 'mot_E',
                            'mot_cov', 'mot_dirp', 'mot_sens',
                            'mot_cent', 'prop_errors')
        for key in initial:
            assert key in valid_init_keys
        for key in settings:
            assert key in valid_settings_keys
        self.settings = settings
        self.initial = initial

    # advances the simulation by n epochs
    def parallel_gen_alg(self, n, pool):
        counter = 0
        # if no population, initialize
        if self.full_data is None:
            self.full_data = np.empty((0, self.gen_size))
            pop = np.asarray([sim.Cell(self.settings) for _ in range(self.gen_size)], dtype = 'O')
            [cell.initialize(self.initial) for cell in pop]
        else:
            parents = selection(self.gen_size, self.s, self.full_data[-1])
            pop = offspring(self.settings, self.gen_size, parents, pool)
        # initialize the population
        while(counter < n):
            start = time.time()
            # run current population simulations
            pop = np.asarray(pool.map(cell_run, pop))
            resized_pop = np.resize(pop,(1,self.gen_size))
            self.full_data = np.append(self.full_data, resized_pop, axis=0)
            # create new population
            parents = selection(self.gen_size, self.s, pop)
            new_pop = offspring(self.settings, self.gen_size, parents, pool)
            pop = new_pop
            counter += 1
            self.completions += 1  
            if (counter%1 == 0):
                end = time.time()
                print(f"{counter} of {n} (total {self.completions}) finished in {end - start : .2f} seconds...")  
        print("Done!")

    # pickles the simulation object
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def summary(self):
        print("The configuraton of this run is: ")
        print(self.settings)
        print(self.initial)
