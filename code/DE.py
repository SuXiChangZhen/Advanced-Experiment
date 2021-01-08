# -*- coding: utf-8 -*-
# The demo of differential evolution algorithm

import numpy as np
import time

def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=200, popsize=400, tol=0.01,
                           mutation=0.5, recombination=0.7,
                           init='random', atol=0, verbose=False):

    solver = Differential_Evolution(func, bounds, args=args,
                                    strategy=strategy, maxiter=maxiter,
                                    popsize=popsize, tol=tol, mutation=mutation,
                                    recombination=recombination, init=init, 
                                    atol=atol, verbose=verbose)
    return solver.solve()

class Differential_Evolution(object):
    def __init__(self, func, bounds, args=(), strategy='best1bin', maxiter=200, popsize=400, 
    tol=0.01, mutation=0.5, recombination=0.7, init='random', 
    atol=0, verbose=False):
        self.tol = tol
        self.func = func
        self.F = mutation
        self.Cr = recombination
        self.init = init
        self.maxiter = maxiter
        self.args = args
        self.population_size = popsize
        bounds = np.asfarray(bounds)
        self.parameter_length = bounds.shape[1]
        self.population_shape = (self.population_size, self.parameter_length)
        self.tol, self.atol = tol, atol
        self.verbose = verbose
        self.output_fitness = [] if self.verbose else None

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        
        self.__scale_arg1 = (self.limits[0] + self.limits[1]) * 0.5 
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        if isinstance(init, str):
            if init == 'random':
                self.init_random
        else:
            self.init_array(init)           
        
        
    def init_random(self):
        self.population = np.random.uniform(self.population_shape)
        self.population_fitness = np.ones(self.population_size) * np.inf
    
    def init_array(self, init):
        # init from user provided arrays
        pop_init = np.asfarray(init)
        self.population = np.clip(self.unscale_parameters(pop_init), 0, 1)
        self.population_fitness = np.ones(self.population_size) * np.inf

    def mutate(self, parent):
        samples = self.select_samples(parent, 3)
        r0, r1, r2 = samples[:3]
        trial = np.ones(self.population.shape[1:]) * 2
        while np.max(trial) > 1 or np.min(trial) < 0:
            samples = self.select_samples(parent, 3)
            r0, r1, r2 = samples[:3]
            trial = self.population[r0] + self.F * (self.population[r1] - self.population[r2])
        return trial
    
    def calculate_population_fitness(self):
        candidates = self.population
        parameters = np.array([self.scale_parameters(c) for c in candidates]) # this can be vectorized
        fitness = self.func(parameters, *self.args)

        minval = np.argmin(fitness)

        # put the lowest energy into the best solution position.
        fitness[[0, minval]] = fitness[[minval, 0]]
        self.population_fitness = fitness
        self.population[[0, minval], :] = self.population[[minval, 0], :]


    def next(self):
        # if np.all(np.isinf(self.population_fitness)):
        #     self.calculate_population_energies()
        trials = np.array([self.mutate(c) for c in range(self.population_size)])
        parameters = np.array([self.scale_parameters(trial) for trial in trials])
        fitness = self.func(parameters, *self.args)  

        for candidate, (fit, trial) in enumerate(zip(fitness, trials)):
            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            if fit <= self.population_fitness[candidate]:
                self.population[candidate] = trial
                self.population_fitness[candidate] = fit

                # if the trial candidate also has a lower energy than the
                # best solution then replace that as well
                if fit <= self.population_fitness[0]:
                    self.population_fitness[0] = fit
                    self.population[0] = trial


    def solve(self):
        if np.all(np.isinf(self.population_fitness)):
            self.calculate_population_fitness()
        if self.verbose:
            self.output_fitness.append(self.population_fitness[0])

        for i in range(1, self.maxiter):
            self.next()
            if self.verbose:
                self.output_fitness.append(self.population_fitness[0])
            print(self.population_fitness[0])
        
        return self.scale_parameters(self.population[0]), self.output_fitness

    def scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
    
    def unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5
    
    def select_samples(self, parent, number_samples):

        idxs = list(range(self.population_size))
        idxs.remove(parent)
        np.random.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs
