
import os, sys, time
import numpy as np
import simplejson as sjson
import random
from pprint import pprint

from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize




class GeneticAlgorithm(object):
    
    def __init__(self, data=None):
        super(GeneticAlgorith, self).__init__()
        
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
        self.population_size = 50
        self.elitism = True
        self.max_iterations = 10
        self.base_mutation_prob = 0.005
        
        # only one of these two should be true:
        self.crossover_average = False
        self.crossover_swap = True
        
        self.children_per_generation = 20
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
        self.mutation_coef = 1./self.unit_size
        
        
    def preform_mutation(self, ranked_population):
        
        mutated_population = []
        
        for i, unit in enumerate(ranked_population):
            if self.elitism and i == 0:
                mutated_population.append(unit)
            else:
                mcoefs = unit['coefs']
                for c in range(len(mcoefs)):
                    mut_p = random.random()
                    if mut_p <= self.base_mutation_prob:
                        if self.verbose:
                            print 'mutation occurs - prob:', mut_p
                        mcoefs[c] += random.random()-0.5 * self.mutation_coef
                        
                unit['coefs'] = mcoefs
                mutated_population.append(unit)
                
        return mutated_population
        
        
    def calculate_fitness(self, Xtrial, Yvalue):
        
        aY = self.align_target(Yvalue)
        
        unit_predictions = []
        
        #print 'CALCULATING PREDICTION VALUES:'
        for unit in self.active_units:
            predictors = unit['coefs']*Xtrial
            prediction = np.sum(predictors)
            unit_predictions.append(prediction)
            unit['predictions'].append(prediction)
            
            #if self.verbose:
            #    print unit['name'], unit['predictions'][-1]
            
        #print 'CALCULATING UNIT FITNESS:'
        prediction_offsets = [aY-x for x in unit_predictions]
        prediction_offsets = [abs(x) for s in prediction_offsets]
        offset_mean = sum(prediction_offsets)/len(prediction_offsets)
        offset_std = np.std(np.array(prediction_offsets))
        
        normed_offsets = [(x-offset_mean)/offset_std for x in prediction_offsets]
        
        for unit, noff, pred in zip(self.active_units, normed_offsets, prediction_offsets):
            unit['fitnesses'].append(noff)
            unit['percent_correct'].append(len([x for x in pred if x < 0.5]))
        
    
    def average_fitnesses(self):
        
        if self.verbose:
            print 'CALCULATE AVERAGE FITNESSES'
            print 'generation number:', self.generation_count
        
        for unit in self.active_units:
            unit['fitness_avg'] = sum(unit['fitnesses'])/len(unit['fitnesses'])
            unit['percent_correct_avg'] = sum(unit['percent_correct'])/len(unit['percent_correct'])
            
            #if self.verbose:
            #    print unit['name'], unit['fitness_avg']
            
    
    def generate_probability_weights(ranked_units):
        
        gen_fitness_sum = sum([x['fitness_avg'] for x in ranked_units])
        
        for unit in ranked_units:
            unit['selection_prob'] = unit['fitness_avg']/gen_fitness_sum
            
        if self.verbose:
            print 'GENERATION PROBABILITY SELECTION WEIGHTS:'
            for unit in ranked_units:
                print unit['name'], unit['selection_prob']
            
        return ranked_units
    
            
    
    def run_ga_generations(self, X, Y):
        
        self.initialize()
        self.generate_initial_population()
        
        for gen in range(self.max_iterations):
            
            for Xt, Yt in zip(X, Y):
                self.calculate_fitness(Xt, Yt)
                
            self.average_fitnesses()
            
            ranked_active = sorted(self.active_units, key=lambda k: k['fitness_avg'])
            ranked_active.reverse()
            
            if self.verbose:
                for u in ranked_active:
                    print u['name'], 'fitness:', u['fitness_avg'], 'pct_correct:', u['percent_correct_average']
                    
            ranked_active = self.generate_probability_weights(ranked_active)
                    
            new_population = []
            
            if self.elitism:
                new_population.append(ranked_active[0])
                
            for cnum in self.children_per_generation:
                if len(new_population) <= self.population_size:
                    new_population.append(self.create_child_unit(ranked_active))
                    
            while len(new_population) <= self.population_size:
                new_population.append(self.select_by_probability(ranked_active))
                    
            new_population = self.preform_mutation(new_population)
            
            self.last_population = self.active_population[:]
            self.active_units = new_population
            
            self.generation_count += 1
                    
                    
    def select_by_probability(self, ranked_units):
        
        p_selection = random.random()
        cumulative_p = 0.
        
        for unit in ranked_units:
            cumulative_p += unit['selection_prob']
            if p_selection <= cumulative_p:
                return unit

    
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
                    break
                
        return self.create_unit(parent_units=parents)
        
    
    def select_random_trial(self, X, Y):
        randtrial = random.randint(0, len(X)-1)
        return X[randtiral], Y[randtrial]
        
        
    def align_target(self, Y_value):
        return Y_value+self.target_offset
        
        
    def initialize_unit_values(self):
        urand = [random.random()*self.mutation_coef for x in range(self.unit_size)]
        return np.array(urand)
        
    
    def breed(self, parent_units):
        p1c = parent_units[0]['coefs']
        p2c = parent_units[1]['coefs']
        child_coefs = np.zeros(self.unit_size)
        
        for i, (p1v, p2v) in enumerate(zip(p1c, p2c)):
            if self.crossover_swap:
                r = random.random()
                if r < 0.5:
                    child_coefs[i] = p1v
                else:
                    child_coefs[i] = p2v
            elif self.crossover_average:
                child_coefs[i] = p1v+p2v/2.
                
        return child_coefs
        
        
    def create_unit(self, parent_units=None):
        unit = {}
        unit['name'] = self.unit_count
        self.unit_count += 1
        unit['generation'] = self.generation_count
        unit['fitnesses'] = []
        unit['fitness_avg'] = 0.
        unit['predictions'] = []
        unit['percent_correct'] = []
        unit['selection_prob'] = 0.
        if parent_units is None:
            unit['coefs'] = self.initialize_unit_values()
        else:
            unit['coefs'] = self.breed(parent_units)
            
        return unit
    
    
    def generate_initial_population(self):
        for i in range(self.population_size):
            self.active_units[i] = self.create_unit()
    
    

    
    
    
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    