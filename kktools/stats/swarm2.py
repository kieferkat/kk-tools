
import numpy as np
import os
import random
import itertools
import time
import json
import nibabel as nib
import scipy.stats as stats
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from linearsvm import ScikitsSVM
from normalize import simple_normalize
from threshold import threshold_by_pvalue



class Particle(object):
    
    def __init__(self):
        super(Particle, self).__init__()
        
        self.id = 0
        self.neighbors = []
        self.best_accuracy = 0.
        self.best_fitness = 0.
        self.accuracies = []
        self.fitnesses = []
        
    
    def initialize_coefs(self, dimensions, particle_type='weight', multiplier=1.):
        
        if particle_type == 'weight':
            random_range = 2.*np.random.random_sample(size=dimensions)-1.
            self.coefs = multiplier*random_range
            
        elif particle_type == 'binary':
            self.coefs = 1.*np.random.randint(2, size=dimensions)
            
        elif particle_type == 'probability':
            self.coefs = np.random.random_sample(size=dimensions)
            
            
    def initialize_velocity(self, dimensions, velocity_start='zeros'):
    
        if velocity_start == 'zeros':
            self.velocity = np.zeros(dimensions)
            


class Swarm(object):
    
    def __init__(self, id=0, vmax=None, xmax=None, particle_type='weight',
                 particle_multiplier=1., neighborhood_type='circle'):
        super(Swarm, self).__init__()
        
        self.id = id
        self.vmax = vmax
        self.xmax = xmax
        self.particle_type = particle_type
        self.particle_multiplier = particle_multiplier
        self.neighborhood_type = neighborhood_type
        
        
        
    def initialize_particles(self, dimensions, neighborhood_type=):
        
        






















