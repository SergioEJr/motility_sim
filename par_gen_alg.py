import numpy as np
import new_sim as sim
import math
import pickle
import matplotlib.pyplot as plt
import time

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
        self.settings = settings
        self.initial = initial

    # takes in a population of candidates of size n. Randomly selects n cells
    # weighted by their performance (distance travelled). Return this selection.
    def selection(self, pop):
        factors = [math.exp(self.s * cell.distance) for cell in pop]
        weights = np.asarray(factors) / np.sum(factors)
        indices = np.random.choice(pop.size, self.gen_size, p=weights)
        return np.asarray([pop[i] for i in indices])

    # Takes in a population of parents, create new cells that inherit parameters from parents
    # with mutations included. 
    def offspring(self, parents, pool):
        children = np.asarray([sim.Cell(**self.settings) for _ in range(self.gen_size)], dtype = 'O')
        new_pop = np.asarray(pool.map(mutate, zip(children, parents)))
        return new_pop

    # advances the simulation by n epochs
    def parallel_gen_alg(self, n, pool):
        counter = 0
        # if no population, initialize
        if self.full_data is None:
            self.full_data = np.empty((0, self.gen_size))
            pop = np.asarray([sim.Cell(**self.settings) for _ in range(self.gen_size)], dtype = 'O')
            [cell.initialize(*self.initial) for cell in pop]
        else:
            parents = self.selection(self.full_data[-1])
            pop = self.offspring(parents, pool)
        # initialize the population
        while(counter < n):
            start = time.time()
            # run current population simulations
            pop = np.asarray(pool.map(cell_run, pop))
            resized_pop = np.resize(pop,(1,self.gen_size))
            self.full_data = np.append(self.full_data, resized_pop, axis=0)
            # use nondimensional distance in exponent
            parents = self.selection(pop)
            new_pop = self.offspring(parents, pool)
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

    # returns a list of the number of numerical errors per generation
    def count_errors(self):
        count = 0
        errors = []
        for gen in data.full_data:
            for cell in gen:
                if cell.ERROR is True:
                    count += 1
            errors.append(count/data.gen_size)
            count = 0
        return errors

    # returns a list of the mean of 'attribute' per generation
    def mean_pgen(self, attribute):
        means = []
        count = 0
        n_valid = 0
        for gen in self.full_data:
            for cell in gen:
                if cell.ERROR is True:
                    continue
                else:
                    n_valid += 1
                    try:
                        count += getattr(cell, attribute) 
                    except:
                        print(f"'{attribute}' not found, try one of the following")
                        print(vars(cell))
                        return -1
            means.append(count/n_valid)
            count = 0
            n_valid = 0
        return np.asarray(means)
