
import numpy as np
import random
import itertools
import nibabel as nib
import scipy.stats as stats
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from normalize import simple_normalize
from threshold import threshold_by_pvalue





class ParticleSwarm(object):
    
    
    def __init__(self, X, Y):
        
        super(ParticleSwarm, self).__init__()
        self.data = data
        self.X = X
        self.Y = Y
        self.verbose = True
        
        # maximum iterations to run
        self.max_iterations = 1000
        
        # particle length; how many dimenstions/coefficients? Should always be
        # one per voxel, so the column length of X
        self.particle_length = self.X.shape[1]
        
        # the particle initialization coefficient is the "size penalty" put on
        # each coefficients initial random coefficient value. The coefficients
        # will start at random [0,1] * self.particle_initialization_coef
        self.particle_initialization_coef = 1./self.particle_length
        
        # the number of particles to use, typically in range 10-50, though
        # there is no recommended "range" for brain data (that i know of).
        # population size must obviously be greater than 1, and should be
        # at the very least greater than 2
        self.population_size = 30
        
        # the neighborhood type specifies the type of connections between population
        # members.
        #
        # 'circle'  :   circle is the default and classic mode. with the circle
        #               each particle is connected to two neighbors, one on the
        #               left size and one on the right. the last and first particles
        #               are connected to complete the circle.
        #
        # 'wheel'   :   The wheel has all particles connected only to a single
        #               particle in the middle. the middle particle is the
        #               communication buffer that the other particles use to
        #               influence each other.
        #
        self.neighborhood_type = 'circle'
        
        # initial velocity function indicates the function that initializes velocity
        # the default is to start the velocities at zero.
        # [more options forthcoming...]
        self.initialize_velocity = 'zero'
        
        # the velocity function determines which components to use when calculating
        # the velocity for each particle. there are pros/cons to each
        #
        # 'accelerated' :   accelerated velocity function forgoes the inclusion
        #                   of local best (Xbest - Xcurrent) component. instead
        #                   the velocity uses only (Gbest - Xcurrent) global best
        #                   component
        #
        # 'global'      :   global uses the conventional (Gbest - Xcurrent) and
        #                   (Xbest - Xcurrent) components in the velocity calculation
        #
        # 'local'       :   local is similar to the global function, except the
        #                   global best is replaced by just the local best in the
        #                   particle's neighborhood
        #
        self.velocity_function = 'local'
        
        # acceleration parameters are subject to stochastic changes as the program
        # iterates. you can change the sum of the accelerations here:
        self.acceleration_sum = 4.1
        self.acceleration_nb = self.acceleration_sum - 2.*random.random()
        self.acceleration_ind = self.acceleration_sum - self.acceleration_nb
        
        # various ways to control the velocity...
        # v_max sets a maximum (absolute) velocity
        self.v_max = 1.0
        
        # inertia constant: keeps the velocity from going out of control as long
        # as some conditions are met
        # in the future this inertia constant will be able to move as the program
        # progresses
        self.inertia_weight = True
        self.inertia_constant = 0.8
        
        ## these variables determine how fitness testing is done...
        
        # fitness type:
        #   'full'  :   this uses all of the training X/Y matrices to test fitness.
        #               slower but more fitness information
        #
        #   'subsample' :   takes a subsample of training X/Y (with replacement) and tests
        #                   the fitness on that. faster but less fitness information
        self.fitness_type = 'subsample'
        
        # subsample size determines how many trials to use if fitness_type is set
        # to subsample. keep in mind that the smaller the size the less fitness info:
        self.subsample_size = 50
        
        # balanced trials ensures that the subsample is comprised equally of
        # positive and negative choices
        # note: if this is set true you probably do not need to downsample the
        # original X matrix
        self.balanced_trials = False
        
        
        
        
    def normalize_xset(self, Xset):
        if self.verbose:
            print 'normalizing X...'
        Xnormed = simple_normalize(Xset)
        return Xnormed
    
    
    def subselect_x(self, Xset, inds):
        if self.verbose:
            print 'subselecting from X...'
        return Xset[:,inds]
        
    
    def subselect_y(self, Yset, inds):
        if self.verbose:
            print 'subselecting from Y...'
        return Yset[inds]
        
        
        
    def split_trials(self, subject_indices, subjects_out):
        if self.verbose:
            print 'splitting trials into trainX/Y, testX/Y...'
        subjects = subject_indices.keys()
        subjects_in = [x for x in subjects if not x in subjects_out]
        train_trials = []
        test_trials = []
        for si in subjects_in:
            train_trials.extend(subject_indices[si])
        for so in subjects_out:
            test_trials.extend(subject_indices[so])
        self.trainX = self.subselect_x(self.X, train_trials)
        self.testX = self.subselect_x(self.X, test_trials)
        self.trainY = self.subselect_y(self.Y, train_trials)
        self.testY = self.subselect_y(self.Y, test_trials)
        


    def create_particle(self, pid, neighbor_ids):
        
        if self.verbose:
            print 'initializing particle w. neighbors: ', pid, neighbor_ids
        particle = {'id':pid, 'neighbors':neighbor_ids}
        
        initial_coefs = np.random.random_sample(size=self.particle_length)
        initial_coefs = initial_coefs * self.particle_initialization_coef
        
        particle['coefs'] = initial_coefs
        particle['best_coefs'] = initial_coefs
        particle['best_fitness'] = 0.0
        particle['distances'] = []
        particle['fitnesses'] = []
        
        if self.initialize_velocity == 'zero':
            particle['velocity'] = np.zeros(self.particle_length)
        
        
        
    def instantiate_particles(self):
        
        self.particles = {}
        self.global_best = np.zeros(self.particle_length)
        self.global_fitness = 0.5
        
        if self.neighborhood_type == 'circle':
            
            if self.verbose:
                print 'initializing particles in type:', self.neighborhood_type
                print 'population size:', self.population_size
            
            for i in range(self.population_size):
                if i == 0:
                    self.particles[i] = self.create_particle(i, [self.population_size-1,i+1])
                elif i == self.population_size-1:
                    self.particles[i] = self.create_particle(i, [i-1, 0])
                else:
                    self.particles[i] = self.create_particle(i, [i-1, i+1])
                    
        elif self.neighborhood_type == 'wheel':
            # not implemented yet
            pass
        
        
        
        
    def subselect_trials(self, X, Y):
        
        if self.verbose:
            print 'setting useable X/Y according to fitness testing type:', self.fitness_type
        
        if self.fitness_type == 'subsample':
            
            if self.balanced_trials:
                pass
            else:
                trials = self.subselect_x(self.trainX, np.random.random_integers(0, len(self.trainX), self.subsample_size))
            
            self.useX = self.subselect_x(self.trainX, trials)
            self.useY = self.subselect_y(self.trainY, trials)
            
        else:
            
            self.useX = self.X
            self.useY = self.Y
            
    
    
    def fitness(self, coefs):
        
        correct = []
        distances = []
        
        for tX, tY in zip(self.useX, self.useY):
            predictors = coefs*tX
            prediction = np.sum(predictors)
            if np.sign(prediction) == np.sign(tY):
                correct.append(1.)
            else:
                correct.append(0.)
            
            distances.append(float(tY)-prediction)
            
        avg_correct = float(sum(correct))/len(correct)
        avg_distance = float(sum(distances))/len(distances)
        
        return avg_correct, avg_distance
        
        
        
    def local_velocity(self, p, nb):
        
        past_v = self.particles[p]['velocity']
        
        if p == nb:
            nb_d = np.zeros(self.particle_length)
        else:
            nb_d = self.particles[nb]['coefs'] - self.particles[p]['coefs']
        
        ind_d = self.particles[p]['best_coefs'] - self.particles[p]['coefs']
        
        nb_d *= self.acceleration_nb
        ind_d *= self.acceleration_ind
        
        if self.inertia_weight:
            inertia = self.inertia_constant
        else:
            inertia = 1.0
            
        return inertia*past_v + nb_d + ind_d
        
        
    def update(self):
        
        # assess current fitness of all particles:
        for p in range(self.population_size):
            
            particle_fitness, particle_distance = self.fitness(self.particles[p]['coefs'])
            self.particles[p]['current_fitness'] = particle_fitness
            self.particles[p]['distances'].append(particle_distance)
            self.particles[p]['fitnesses'].append(particle_fitness)
            
            if particle_fitness > self.particles[p]['best_fitness']:
                self.particles[p]['best_fitness'] = particle_fitness
                self.particles[p]['best_coefs'] = self.particles[p]['coefs']
            
            if particle_fitness > self.global_fitness:
                self.global_fitness = particle_fitness
                self.global_best = self.particles[p]['coefs']
                
        self.fitness_ranking = [[p['id'], p['current_fitness']] for p in self.particles]
        self.fitness_ranking = sorted(self.fitness_ranking, key=lambda k: k[1])
        self.fitness_ranking.reverse()
        
        if self.verbose:
            for id, fit in self.fitness_ranking:
                print 'ID:', id, 'ACC:', fit

        # update the velocities of all particles:
        for p in range(self.population_size):
            
            if self.velocity_function == 'local':
                
                neighborhood_best = p
                for np in self.particles[p]['neighbors']:
                    if self.fitness(self.particles[np]['coefs']) > particle_fitness:
                        neighborhood_best = np
                    
                self.particles[p]['velocity'] = self.local_velocity(p, neighborhood_best)
            
                
        # update the positions of all particles:
        for p in range(self.population_size):
            self.particles[p]['coefs'] += self.particles[p]['velocity']
    
    
    def run(self):
        
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    