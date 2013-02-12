
import numpy as np
import os
import random
import itertools
import time
import nibabel as nib
import scipy.stats as stats
from pprint import pprint
from ..base.crossvalidation import CVObject
from ..base.nifti import NiftiTools
from linearsvm import ScikitsSVM
from normalize import simple_normalize
from threshold import threshold_by_pvalue





class ParticleSwarm(object):
    
    
    def __init__(self, X, Y):
        
        super(ParticleSwarm, self).__init__()
        self.X = X
        self.Y = Y
        self.verbose = True
        
        # maximum iterations to run
        self.max_iterations = 1000
        
        
        # particle length; how many dimenstions/coefficients? Should always be
        # one per voxel, so the column length of X
        self.particle_length = self.X.shape[1]
        
        # dummy negative 1 and positive 1 vectors:
        self.negones = -1.*np.ones(self.particle_length)
        self.posones = np.ones(self.particle_length)
        
        # the 
        
        # the particle initialization coefficient is the "size penalty" put on
        # each coefficients initial random coefficient value. The coefficients
        # will start at random [-1,1] * self.particle_initialization_coef
        self.particle_initialization_coef = 1.0     #1./self.particle_length
        
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
        # 'axle'   :   The axle has all particles connected only to a single
        #               particle in the middle. the middle particle is the
        #               communication buffer that the other particles use to
        #               influence each other.
        #
        # 'wheel'   :   The wheel is a combination of the cirlce and the axle.
        #
        # 'scatter' :   connects to 4 other group members at random
        #
        #
        self.neighborhood_type = 'circle'
        
        
        # 'cascade' :   particles are only affected by the global best. when
        #               their velocity passes a lower threshold they are
        #               re-randomized to be somewhere on the grid and start
        #               'falling' back towards the global best
        self.use_cascade = True
        
        self.cascade_velocity_threshold = 100000.
        self.cascade_narrowing = True
        self.cascade_narrowing_coef = 0.95
        
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
        self.acceleration_sum = 4.2
        
        
        self.use_constriction_coefficient = True
        self.constriction_k = 1.0
        self.constriction = None
        
        # various ways to control the velocity...
        # v_max sets a maximum (absolute) velocity per dimension
        self.xmax = 100.0
        self.vmax = self.xmax*0.25
        
        
        # inertia constant: keeps the velocity from going out of control as long
        # as some conditions are met
        # in the future this inertia constant will be able to move as the program
        # progresses
        self.inertia_constant = 0.7
        
        

        self.accuracy_lookback = 20.
        
        # fitness type:
        #   'full'  :   this uses all of the training X/Y matrices to test fitness.
        #               slower but more fitness information
        #
        #   'subsample' :   takes a subsample of training X/Y (with replacement) and tests
        #                   the fitness on that. faster but less fitness information
        #
        self.fitness_type = 'full'
        
        # subsample size determines how many trials to use if fitness_type is set
        # to subsample. keep in mind that the smaller the size the less fitness info:
        self.subsample_size = 10
        
        # balanced trials ensures that the subsample is comprised equally of
        # positive and negative choices
        # note: if this is set true you probably do not need to downsample the
        # original X matrix
        self.balanced_trials = False
        
        
        #self.use_variable_momentum = True
        self.momentum_lowpass = 0.5
        self.momentum_power = 1.0
        self.last_momentum = np.ones(self.particle_length)
        self.applied_momentum = np.ones(self.particle_length)
        
        
        
    def set_stochastic_accuracy_weights(self):
        self.acceleration_a = self.acceleration_sum * random.random()
        self.acceleration_b = self.acceleration_sum - self.acceleration_a
        
    
        
    def normalize_xset(self, Xset):
        if self.verbose:
            print 'normalizing X...'
        Xnormed = simple_normalize(Xset)
        return Xnormed
    
    
    def subselect_x(self, Xset, inds):
        print inds
        inds = [int(x) for x in inds]
        if self.verbose:
            print 'subselecting from X...'
        return Xset[inds,:]
        
    
    def subselect_y(self, Yset, inds):
        inds = [int(x) for x in inds]
        if self.verbose:
            print 'subselecting from Y...'
        return Yset[inds]
        

        
    def subselect_trials(self, X, Y):
        
        if self.verbose:
            print 'setting useable X/Y according to fitness testing type:', self.fitness_type
        
        if self.fitness_type == 'subsample':
            
            if self.balanced_trials:
                pass
            else:
                trials = np.random.random_integers(0, X.shape[0]-1, self.subsample_size)
            
            self.useX = self.subselect_x(X, trials)
            self.useY = self.subselect_y(Y, trials)
            
        else:
            
            self.useX = self.X
            self.useY = self.Y
            
        
        
        


    def create_particle(self, pid, neighbor_ids, particle_type='continuous',
                        prepared_coefs=None):
        
        if self.verbose:
            print 'initializing particle w. neighbors: ', pid, neighbor_ids
        particle = {'id':pid, 'neighbors':neighbor_ids}
        
        if prepared_coefs is None:
            if particle_type == 'continuous':
                initial_coefs = (2.*self.xmax*np.random.random_sample(size=self.particle_length))-self.xmax
                initial_coefs = initial_coefs * self.particle_initialization_coef
            elif particle_type == 'binary':
                initial_coefs = 1.*np.random.randint(2, size=self.particle_length)
            elif particle_type == 'probability':
                initial_coefs = np.random.random_sample(size=self.particle_length)
        else:
            print 'USING PREPARED COEFS FOR:', pid
            print prepared_coefs
            assert len(prepared_coefs) == self.particle_length
            initial_coefs = np.array(prepared_coefs)
        
        particle['coefs'] = initial_coefs
        particle['best_coefs'] = initial_coefs
        particle['best_distance'] = 0.0
        particle['best_accuracy'] = 50.0
        particle['distances'] = []
        particle['accuracies'] = []
        particle['sum_distances'] = []
        particle['sum_accuracies'] = []
        
        if self.initialize_velocity == 'zero':
            particle['velocity'] = np.zeros(self.particle_length)
        
        return particle
        
    
    
    def fill_neighborhood(self, particles, population_size, neighborhood_type,
                          particle_type='continuous', prepared_coefs=None):
        
        if neighborhood_type == 'circle':
            for i in range(population_size):
                if i == 0:
                    particles[i] = self.create_particle(i, [population_size-1,i+1],
                                                        particle_type, prepared_coefs=prepared_coefs)
                elif i == self.population_size-1:
                    particles[i] = self.create_particle(i, [i-1, 0],
                                                        particle_type)
                else:
                    particles[i] = self.create_particle(i, [i-1, i+1],
                                                        particle_type)
                    
        elif neighborhood_type == 'wheel':
            
            for i in range(population_size-1):
                if i == 0:
                    particles[i] = self.create_particle(i, [population_size-2, i+1, population_size-1],
                                                        particle_type)
                elif i == self.population_size-2:
                    particles[i] = self.create_particle(i, [i-1, 0, population_size-1],
                                                        particle_type)
                else:
                    particles[i] = self.create_particle(i, [i-1, i+1, population_size-1],
                                                        particle_type)
                    
            particles[population_size-1] = self.create_particle(population_size-1, range(population_size-1),
                                                                particle_type, prepared_coefs=prepared_coefs)
            
        
        elif neighborhood_type == 'scatter':
            # you must have a decent amount of particles to use scatter
            
            for i in range(population_size):
                neighbors = []
                while len(neighbors) < 4:
                    rp = np.random.randint(population_size)
                    if rp != i:
                        neighbors.append(rp)
                if i == 0:
                    particles[i] = self.create_particle(i, neighbors, particle_type,
                                                        prepared_coefs=prepared_coefs)
                else:
                    particles[i] = self.create_particle(i, neighbors, particle_type)
                    
                    
                
        return particles
        
    
    
    def instantiate_particles(self, neighborhood_type, neighborhood_size,
                              prepared_coefs=None, particle_type='continuous'):
        
        self.particles = {}
        self.global_best_coefs = -1. * np.ones(self.particle_length)
        self.global_best_distance = -1.0
        self.global_best_accuracy = 0.00
            
        self.vmax = np.array([self.vmax for x in range(self.particle_length)])
        self.xmax = np.array([self.xmax for x in range(self.particle_length)])
        
        self.particles = self.fill_neighborhood(self.particles, neighborhood_size,
                                                neighborhood_type, particle_type=particle_type,
                                                prepared_coefs=prepared_coefs)

            
            
    
    def calculate_log_distance(self, Y, prediction):
        abs_pred = np.absolute(prediction)
        correct = np.sign(Y) == np.sign(prediction)
        justified_pred = abs_pred * correct
        
        score = 1. / (1. + np.exp(-1. * justified_pred))
        
        return (score - 0.5) * 2.
        
    
    def fitness(self, coefs, testX=None, testY=None, accuracy_bonus=0.0):
        
        correct = []
        distances = []
        
        if testX is None:
            cX = self.useX
        else:
            cX = testX
        if testY is None:
            cY = self.useY
        else:
            cY = testY
            
            
        for iter, (tX, tY) in enumerate(zip(cX, cY)):
            
            predictors = coefs*tX
            prediction = np.sum(predictors)
            
            if np.sign(prediction) == np.sign(tY):
                correct.append(1.)
            else:
                correct.append(0.)
                
            distances.append(self.calculate_log_distance(float(tY), prediction))
            
        avg_correct = float(sum(correct))/len(correct)
        avg_distance = sum(distances)/len(distances)
        
        return avg_correct, avg_distance
    
    
    
    def svm_crossvalidated_fitness(self, X, Y, subject_indices, voxel_p, folds=5):
        
        r_vector = np.random.random_sample(size=self.particle_length)
        inclusion_inds = np.where(voxel_p > r_vector)[0]
        
        svm = ScikitsSVM()
        svm.X = X[:,inclusion_inds]
        svm.Y = Y
        svm.subject_indices = subject_indices
        
        cv_accuracy = svm.crossvalidate(folds=folds)
        
        return cv_accuracy
    
    
    
    def validate_outofsample(self, testX, testY):
        avg_correct, avg_distance = self.fitness(self.global_best_coefs, testX, testY)
        print '\nTEST SAMPLE AVG CORRECT:', avg_correct
        print 'TEST SAMPLE AVG DISTANCE:', avg_distance
        
        
        
    def calculate_variable_momentum(self, particles):
        
        self.average_velocity = np.zeros(self.particle_length)
        self.average_speed = np.zeros(self.particle_length)
        self.convergence_distance = np.zeros(self.particle_length)
        
        for p in particles:
            self.average_velocity += particles[p]['velocity']
            self.average_speed += np.absolute(particles[p]['velocity'])
            
        self.average_velocity = np.absolute(self.average_velocity) / self.population_size
        self.average_speed = self.average_speed / self.population_size
        
        print 'average velocity sum:', np.sum(self.average_velocity)
        print 'average speed sum:', np.sum(self.average_speed)
        
        self.convergence_distance = np.square(self.average_velocity) + np.square(self.average_speed)
        self.convergence_distance = np.sqrt(self.convergence_distance)
        
        print 'convergence sum:', np.sum(self.convergence_distance)
        
        convergence_max = np.max(self.convergence_distance)
        
        variable_momentum_weight = convergence_max / (self.convergence_distance + convergence_max)
        
        variable_momentum_weight = variable_momentum_weight * self.inertia_constant
        
        variable_momentum = self.particle_length * variable_momentum_weight/np.sum(variable_momentum_weight)
        
        self.applied_momentum = (1. - self.momentum_lowpass) * variable_momentum + (self.momentum_lowpass*self.last_momentum)
        
        self.applied_momentum = np.power(self.applied_momentum, np.array([self.momentum_power for x in range(self.particle_length)]))
        
        print 'momentum sum:', np.sum(self.applied_momentum)
        
        self.last_momentum = self.applied_momentum
        return self.applied_momentum
        
        
        
    def local_velocity(self, particles, p, nb):
        
        past_v = particles[p]['velocity']
        
        vel_coef = past_v*self.inertia_constant
        
        # multiply the change stochastically with uniform random vectors:
        self.acceleration_a = self.acceleration_sum * np.random.random()
        self.acceleration_b = self.acceleration_sum - self.acceleration_a
        

        if self.use_constriction_coefficient and self.constriction is None:
            a_ = (2. * self.constriction_k)
            b_ = 2. - self.acceleration_sum
            c_ = self.acceleration_sum**2. - 4.*self.acceleration_sum
            d_ = np.absolute(b_ - np.sqrt(c_))
            self.constriction = a_ / d_
            print 'constriction, ', self.constriction
        
        
        #gb_d = self.global_best_coefs - particles[p]['coefs']

        if p == nb:
            nb_d = np.zeros(self.particle_length)
        else:
            nb_d = particles[nb]['best_coefs'] - particles[p]['coefs']
            
        ind_d = particles[p]['best_coefs'] - particles[p]['coefs']

        nb_d *= self.acceleration_a
        ind_d *= self.acceleration_b

        next_velocity = vel_coef + nb_d + ind_d
            
            
        if self.use_constriction_coefficient:
            next_velocity *= self.constriction
                            
        next_velocity = np.minimum(self.vmax, np.absolute(next_velocity)) * np.sign(next_velocity)
            
        return next_velocity
    
    
    def metric_over_recent(self, metric, iterations, average=False):
        
        if len(metric) < iterations:
            metric_sum = sum(metric)
            avg_len = len(metric)
        else:
            metric_sum = sum(metric[-iterations:])
            avg_len = iterations
    
        if average:
            return metric_sum/avg_len
        else:
            return metric_sum
    
    
    def svm_update(self, particles, X, Y, subject_indices, folds=5, prior_iters=25):
        
        for p in particles:
            
            probs = particles[p]['coefs']
            cv_accuracy = self.svm_crossvalidated_fitness(X, Y, subject_indices, probs, folds=folds)
            
            particles[p]['current_accuracy'] = cv_accuracy
            particles[p]['accuracies'].append(cv_accuracy)
            
            acc_iters = self.metric_over_recent(particles[p]['accuracies'], prior_iters,
                                                average=True)
            
            particles[p]['sum_accuracies'].append(acc_iters)
            
            if particles[p]['best_coefs'] is None:
                particles[p]['best_coefs'] = probs.copy()
            
            if cv_accuracy > particles[p]['best_accuracy']:
                particles[p]['best_accuracy'] = cv_accuracy
                
            if cv_accuracy > self.global_best_accuracy:
                self.global_best_accuracy = cv_accuracy
                self.global_best_coefs = probs.copy()
            
            neighborhood_best = p
            for npr in particles[p]['neighbors']:
                if particles[npr]['best_accuracy'] > cv_accuracy:
                    neighborhood_best = npr
                    
            particles[p]['velocity'] = self.local_velocity(particles, p, neighborhood_best)
            particles[p]['coefs'] += particles[p]['velocity']
            particles[p]['coefs'] = np.maximum(np.zeros(self.particle_length),particles[p]['coefs'])
            particles[p]['coefs'] = np.minimum(np.ones(self.particle_length),particles[p]['coefs'])
            
            
    def save_best(self, output_dir=os.path.join(os.getcwd(), 'coefsave'),
                  ):
        
        try:
            os.makedirs(output_dir)
        except:
            pass
        
        name = '_'.join(['I'+str(self.current_iteration), '.npy'])
        
        
        
    
    def asynchronous_update(self, particles, prior_iters=100):
        
        for p in particles:
            
            pcoefs = particles[p]['coefs']
            pacc, pdist = self.fitness(pcoefs)
            
            particles[p]['current_distance'] = pdist
            particles[p]['current_accuracy'] = pacc
            particles[p]['distances'].append(pdist)
            particles[p]['accuracies'].append(pacc)
            
            dist_iters = self.metric_over_recent(particles[p]['distances'], prior_iters)
            acc_iters = self.metric_over_recent(particles[p]['accuracies'], prior_iters,
                                                average=True)
            
            particles[p]['sum_distances'].append(dist_iters)
            particles[p]['sum_accuracies'].append(acc_iters)
            
            if particles[p]['best_coefs'] is None:
                particles[p]['best_coefs'] = pcoefs.copy()
                
            if pdist > particles[p]['best_distance']:
                particles[p]['best_distance'] = pdist
                particles[p]['best_coefs'] = pcoefs.copy()
                
            if pacc > particles[p]['best_accuracy']:
                particles[p]['best_accuracy'] = pacc
                
            if pdist > self.global_best_distance:
                self.global_best_distance = pdist
                self.global_best_accuracy = pacc
                self.global_best_coefs = pcoefs.copy()
                self.save_best()
                    
            neighborhood_best = p
            for npr in particles[p]['neighbors']:
                if particles[npr]['best_distance'] > pdist:
                    neighborhood_best = npr
                    
            particles[p]['velocity'] = self.local_velocity(particles, p, neighborhood_best)
            particles[p]['coefs'] += particles[p]['velocity']
            particles[p]['coefs'] = np.maximum(-1.*self.xmax*np.ones(self.particle_length),particles[p]['coefs'])
            particles[p]['coefs'] = np.minimum(self.xmax*np.ones(self.particle_length),particles[p]['coefs'])
            
            if self.current_iteration > 1:
                if self.use_cascade:
                    if np.sum(np.absolute(particles[p]['velocity'])) < self.cascade_velocity_threshold:
                        print 'CASCADE RESET OF PARTICLE: ', p
                        particles[p]['coefs'] = (2.*self.xmax*np.random.random_sample(size=self.particle_length))-self.xmax
                        particles[p]['best_distance'] = 0.0
                        particles[p]['best_accuracy'] = 0.5
                        particles[p]['distances'] = []
                        particles[p]['accuracies'] = []

                        
                        if self.cascade_narrowing:
                            self.cascade_velocity_threshold *= self.cascade_narrowing_coef
                            print 'NEW CASCADE VELOCITY THRESHOLD:', self.cascade_velocity_threshold



    def print_rank(self, particles, lookback=100):
        self.fitness_ranking = [[p['id'], p['sum_distances'][-1], p['current_distance'], p['current_accuracy'],
                                p['sum_accuracies'][-1], np.sum(np.absolute(p['velocity']))] for p in particles.values()]
        self.fitness_ranking = sorted(self.fitness_ranking, key=lambda k: k[1])
        self.fitness_ranking.reverse()
        for id, adist, dist, acc, accs, vel in self.fitness_ranking:
            print 'ID:', id, '\tAVG FIT:', adist, '\tCUR FIT:', dist, '\tCUR ACC:', acc, '\tAVG ACC :', accs, '\tVEL:', vel
                
                
    def print_svm_rank(self, particles, lookback=25):
        self.fitness_ranking = [[p['id'], p['current_accuracy'], p['sum_accuracies'][-1],
                                 np.sum(np.absolute(p['velocity']))] for p in particles.values()]
        self.fitness_ranking = sorted(self.fitness_ranking, key=lambda k: k[1])
        self.fitness_ranking.reverse()
        for id, acc, accs, vel in self.fitness_ranking:
            print 'ID:', id, '\tCUR ACC:', acc, '\tAVG ACC :', accs, '\tVEL:', vel
              
                
                
    def prepare_testsample(self, X, Y):
        self.testX = X
        self.testY = Y
        self.testX = self.normalize_xset(self.testX)
                
    
    def run(self, test=True, prepared_coefs=None):
                
        self.instantiate_particles(self.neighborhood_type, self.population_size,
                                   prepared_coefs=prepared_coefs)
        self.X = self.normalize_xset(self.X)
        
        for i in range(self.max_iterations):
            self.current_iteration = i
            
            print '\nITERATION: %s\n' % (str(i))
            

            self.subselect_trials(self.X, self.Y)
            #if self.current_iteration > 0:
            #    self.calculate_variable_momentum(self.particles)
            self.asynchronous_update(self.particles)
            
            #if i % 10 == 0:
            self.print_rank(self.particles)
            
            if self.use_cascade:
                if self.current_iteration > 1:
                    accs = []
                    for p in self.particles:
                        accs.append(self.particles[p]['current_distance'])
                    if len(np.unique(accs)) == 1:
                        print 'RESETTING ALL PARTICLES, SAME ACCURACY...'
                        for p in self.particles.values():
                            p['coefs'] = (2.*self.xmax*np.random.random_sample(size=self.particle_length))-self.xmax
                            p['best_distance'] = 0.0
                            p['best_accuracy'] = 0.5
                            p['distances'] = []
                            p['accuracies'] = []
                            
            
            if test:
                self.validate_outofsample(self.testX, self.testY)
            
            
            
    def run_svm_featureselection(self, subject_indices, folds=5):
        
        self.instantiate_particles(self.neighborhood_type, self.population_size,
                                   particle_type='probability')
        #self.X = self.normalize_xset(self.X)
        
        for i in range(self.max_iterations):
            self.current_iteration = i
            print '\nITERATION: %s\n' % (str(i))
            
            self.svm_update(self.particles, self.X, self.Y, subject_indices, folds=folds)
            
            self.print_svm_rank(self.particles)
    
    
    
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    