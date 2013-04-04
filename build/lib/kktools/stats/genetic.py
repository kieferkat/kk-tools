
import os, sys, time
import numpy as np
import simplejson as sjson
import random
from pprint import pprint
import copy

from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize




class GeneticAlgorithm(object):
    
    def __init__(self, data=None):
        super(GeneticAlgorithm, self).__init__()
        
        self.verbose = True
        
        ## DATA STRUCTURE PARAMETERS
        self.folds = 0
        self.data = data
        self.cv = CVObject()
        self.unit_count = 0
        self.generation_count = 0
        self.active_units = []

        
        if self.data:
            self.X = getattr(self.data, 'X', None)
            self.Y = getattr(self.data, 'Y', None)
            self.trial_mask = getattr(self.data, 'trial_mask', None)
            self.indices_dict = getattr(self.data, 'subject_trial_indices', None)
        
        
        ## GENETIC ALGORITHM PARAMETERS
        self.population_size = 10
        self.elitism = True
        self.max_iterations = 10000
        self.base_mutation_prob = 0.005
        self.mutation_multiplier = 2.
        self.last_fitness_mean = 0.
        
        # fitness type can be either 'correct' (percent correct) or 'distance'
        # (almost the same as correct, but more continuous. ambiguity is a factor)
        self.fitness_type = 'correct'
        
        # only one of these two should be true:
        self.crossover_average = False
        self.crossover_swap = True
        
        self.children_per_generation = 5
        self.target_offset = -0.5
        
        
            
    def prepare_crossvalidation_folds(self, indices_dict=None, folds=None,
                                      leave_mod_in=True):
        
        if indices_dict:
            self.indices_dict = indices_dict
        if folds:
            self.folds = folds
            
        self.cv.prepare_folds(indices_dict=self.indices_dict, folds=self.folds,
                              leave_mod_in=leave_mod_in)
        
        
    def initialize(self, X):
        
        # the length of a genetic species "unit", one entry per voxel:
        self.unit_size = X.shape[1]
        # mutation coef is the multiplier on the random mutation value
        self.mutation_coef = self.mutation_multiplier/self.unit_size
        
        
    def preform_mutation(self, ranked_population):
        
        mutated_population = []
        mutations = 0
        
        if self.verbose:
            print 'PREFORMING MUTATION...'
        
        for i, unit in enumerate(ranked_population):
            if self.elitism and i == 0:
                mutated_population.append(unit)
            else:
                mcoefs = unit['coefs']
                rsamp = np.random.random_sample(size=(len(mcoefs)))
                repsamp = np.random.random_sample(size=(len(mcoefs)))
                for c, r, rep in zip(range(len(mcoefs)), rsamp, repsamp):
                    if r <= self.base_mutation_prob:
                        mutations += 1
                        mcoefs[c] += rep-0.5 * self.mutation_coef
                        
                unit['coefs'] = mcoefs
                mutated_population.append(unit)
        
        if self.verbose:
            print 'mutations:', mutations
            print 'fitness differential:', self.fitness_mean-self.last_fitness_mean
            
        self.last_fitness_mean = self.fitness_mean
        
        return mutated_population
        
        
    def calculate_fitness(self, Xtrial, Yvalue):
        
        aY = self.align_target(Yvalue)
        
        unit_predictions = []
        
        #print 'CALCULATING PREDICTION VALUES:'
        for unit in self.active_units:
            predictors = unit['coefs']*Xtrial
            prediction = np.sum(predictors)
            unit_predictions.append(prediction)
            
            #if self.verbose:
            #    print unit['name'], unit['predictions'][-1]
            
        #print 'CALCULATING UNIT FITNESS:'
        
        for unit, pred in zip(self.active_units, unit_predictions):
            #print pred
            if np.sign(aY) == np.sign(pred):
                unit['correct'].append(1.)
            else:
                unit['correct'].append(0.)
        
        if self.fitness_type == 'distance':
            
            prediction_offsets = [abs(aY-x) for x in unit_predictions]
            inversed_offsets = [1./x for x in prediction_offsets]
            #offset_mean = sum(inversed_offsets)/len(inversed_offsets)
            #normed_offsets = [x/offset_mean for x in inversed_offsets]
            
            for unit, no in zip(self.active_units, inversed_offsets):
                unit['fitnesses'].append(no)
            
                
        elif self.fitness_type == 'correct':
            
            pass
            
            '''
            correct_this_trial = []
            for unit in self.active_units:
                correct_this_trial.append(unit['correct'][-1])
            
            cpct_mean = float(sum(correct_this_trial))/len(correct_this_trial)
            
            if cpct_mean > 0.:
                normed_cpct = [x/cpct_mean for x in correct_this_trial]
            else:
                normed_cpct = correct_this_trial
            
            for unit, ncpct in zip(self.active_units, normed_cpct):
                unit['fitnesses'].append(ncpct)
            '''
                    
        
    
    def average_fitnesses(self):
        
        if self.verbose:
            print 'CALCULATE AVERAGE FITNESSES'
            print 'generation number:', self.generation_count
            
        if self.fitness_type == 'correct':
            self.fitness_mean = sum([sum(x['correct']) for x in self.active_units])/len(self.active_units)
        elif self.fitness_type == 'distance':
            self.fitness_mean = sum([sum(x['fitnesses']) for x in self.active_units])/len(self.active_units)
        
        if self.verbose:
            print 'fitness mean:', self.fitness_mean
        
        for unit in self.active_units:
            
            unit['percent_correct'] = sum(unit['correct'])/len(unit['correct'])
            
            if self.fitness_type == 'correct':
                unit['fitness_avg'] = sum(unit['correct'])/self.fitness_mean
                
            elif self.fitness_type == 'distance':
                unit['fitness_avg'] = sum(unit['fitnesses'])/self.fitness_mean
                        
            #if self.verbose:
            #    print unit['name'], unit['fitness_avg']
            
    
    def generate_probability_weights(self, ranked_units):
        
        gen_fitness_sum = sum([x['fitness_avg'] for x in ranked_units])
        
        for unit in ranked_units:
            if gen_fitness_sum > 0.:
                unit['selection_prob'] = unit['fitness_avg']/gen_fitness_sum
            else:
                unit['selection_prob'] = 1./float(len(ranked_units))
            
        if self.verbose:
            print 'GENERATION OF PROBABILITY SELECTION WEIGHTS:'
            for unit in ranked_units:
                print unit['name'], unit['selection_prob']
            
        return ranked_units
    
            
    
    def run_ga_generations(self, X, Y):
        
        self.initialize(X)
        self.generate_initial_population()
        
        for gen in range(self.max_iterations):
            
            for unit in self.active_units:
                unit['correct'] = []
                unit['percent_correct'] = []
                unit['fitness_avg'] = 0.
                unit['fitnesses'] = []
            
            if self.verbose:
                print 'ITERATING THROUGH TRIALS...\n'
            for tnum, (Xt, Yt) in enumerate(zip(X, Y)):
                self.calculate_fitness(Xt, Yt)
                
            self.average_fitnesses()
            
            ranked_active = sorted(self.active_units, key=lambda k: k['fitness_avg'])
            ranked_active.reverse()
            
            if self.verbose:
                for u in ranked_active:
                    print u['name'], 'fitness:', u['fitness_avg'], 'pct_correct:', u['percent_correct'], 'num correct', sum(u['correct'])
                    
            ranked_active = self.generate_probability_weights(ranked_active)
                    
            new_population = []
            
            if self.elitism:
                new_population.append(ranked_active[0])
                
            if self.verbose:
                print 'BREEDING CHILDREN...\n'
                
            for cnum in range(self.children_per_generation):
                if len(new_population) <= self.population_size:
                    new_population.append(self.create_child_unit(ranked_active))
                    
            while len(new_population) <= self.population_size:
                new_population.append(self.select_by_probability(ranked_active))
                    
            new_population = self.preform_mutation(new_population)
            
            #self.last_population = self.active_units[:]
            self.active_units = new_population
            
            self.generation_count += 1
                    
                    
    def select_by_probability(self, ranked_units):
        
        p_selection = random.random()
        cumulative_p = 0.
        
        for unit in ranked_units:
            cumulative_p += unit['selection_prob']
            if p_selection <= cumulative_p:
                return self.create_unit(clone=unit['coefs'])

    
    def create_child_unit(self, ranked_units):
        
        sel_inds = []
        parents = []
        
        while len(parents) < 2:
            
            p_selection = random.random()
            cumulative_p = 0.
            
            for i, unit in enumerate(ranked_units):
                cumulative_p += unit['selection_prob']
                if p_selection <= cumulative_p:
                    if i not in sel_inds:
                        parents.append(unit)
                        sel_inds.append(i)
                
        return self.create_unit(parent_units=parents)
        
    
    def select_random_trial(self, X, Y):
        randtrial = random.randint(0, len(X)-1)
        return X[randtiral], Y[randtrial]
        
        
    def align_target(self, Y_value):
        return float(Y_value)+float(self.target_offset)
        
        
    def initialize_unit_values(self):
        #urand = [random.random()*(1./self.unit_size) for x in range(self.unit_size)]
        #return np.array(urand)
        return np.array([x/self.unit_size for x in np.random.random_sample(size=(self.unit_size))])
    
    def breed(self, parent_units):
        p1c = parent_units[0]['coefs']
        p2c = parent_units[1]['coefs']
        rsamp = np.random.random_sample(size=(len(p1c)))
        child_coefs = np.zeros(self.unit_size)
        
        for i, (p1v, p2v, r) in enumerate(zip(p1c, p2c, rsamp)):
            if self.crossover_swap:
                if r < 0.5:
                    child_coefs[i] = p1v
                else:
                    child_coefs[i] = p2v
            elif self.crossover_average:
                child_coefs[i] = p1v+p2v/2.
                
        return child_coefs
        
        
    def create_unit(self, parent_units=None, clone=None):
        unit = {}
        unit['name'] = self.unit_count
        self.unit_count += 1
        unit['generation'] = self.generation_count
        unit['fitnesses'] = []
        unit['fitness_avg'] = 0.
        unit['correct'] = []
        unit['percent_correct'] = []
        unit['selection_prob'] = 0.
        if parent_units is not None:
            unit['coefs'] = self.breed(parent_units)
        elif clone is not None:
            unit['coefs'] = clone
        else:
            unit['coefs'] = self.initialize_unit_values()
            
        return unit
    
    
    def generate_initial_population(self):
        for i in range(self.population_size):
            self.active_units.append(self.create_unit())
            
        if self.verbose:
            for unit in self.active_units:
                print 'unit', unit['name'], 'sum coef:', sum(unit['coefs'])
    
    

    
    
    
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    