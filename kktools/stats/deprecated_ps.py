
class DepPS(object):
    
    def __init__(self):
        super(DepPS, self).__init__()
        
    def synchronus_update(self):
        
        # assess current fitness of all particles:
        for p in range(self.population_size):
            
            #print self.particles[p]
            
            particle_accuracy, particle_distance = self.fitness(self.particles[p]['coefs'])
            
            if self.fitness_metric == 'accuracy':
                particle_fitness = particle_accuracy
            elif self.fitness_metric == 'distance':
                particle_fitness = particle_distance
                
            if self.accuracy_fitness_compensation:
                accuracy_compensation = self.accuracy_compensator(self.particles[p]['accuracies'])
                self.particles[p]['last_accuracy_comp'] = accuracy_compensation
                particle_fitness += accuracy_compensation
            
            self.particles[p]['current_fitness'] = particle_fitness
            self.particles[p]['current_accuracy'] = particle_accuracy
            self.particles[p]['current_distance'] = particle_distance
            self.particles[p]['distances'].append(particle_distance)
            self.particles[p]['accuracies'].append(particle_accuracy)
            
            if self.particles[p]['best_fitness'] is None:
                #if self.verbose:
                #    print 'setting first best fitness to current fitness.'
                self.particles[p]['best_fitness'] = particle_fitness
                            
            if particle_fitness > self.particles[p]['best_fitness']:
                self.particles[p]['best_fitness'] = particle_fitness
                self.particles[p]['best_coefs'] = self.particles[p]['coefs'].copy()
            
            if particle_fitness > self.global_fitness:
                self.global_fitness = particle_fitness
                self.global_best = self.particles[p]['coefs'].copy()
                    
                
        self.fitness_ranking = [[part['id'], part['current_fitness'], part['accuracies'],
                                 part['distances'], part['current_accuracy']] for part in self.particles.values()]
        self.fitness_ranking = sorted(self.fitness_ranking, key=lambda k: k[1])
        
        self.fitness_ranking.reverse()
        
        if self.verbose:
            for id, fit, accs, dists, acc in self.fitness_ranking:
                if len(accs) > 20:
                    short_accs = accs[-20:]
                else:
                    short_accs = [-1]
                print 'ID:', id, '\tFIT:', fit, '\tCUR ACC:', acc, '\tAVG ACC -20:', sum(short_accs)/len(short_accs)
                #print 'ID:', id, '\tCUR ACC:', acc, '\tAVG ACC:', sum(accs)/len(accs), '\tAVG -20:', sum(short_accs)/len(short_accs)
                
        
        if self.verbose:
            print 'caclulating variable momentum...'
            
        if self.use_variable_momentum:
            if self.current_iteration > 0:
                self.calculate_variable_momentum()
                
                
        # update the velocities of all particles:
        if self.verbose:
            print 'updating particle velocities...'
            
        for p in range(self.population_size):
            current_fitness = self.particles[p]['current_fitness']
            
            if self.velocity_function == 'local':
                neighborhood_best = p
                
                for npr in self.particles[p]['neighbors']:
                    if self.particles[npr]['best_fitness'] > current_fitness:
                        neighborhood_best = npr

                self.particles[p]['velocity'] = self.local_velocity(p, neighborhood_best)
                
                
            
                
        # update the positions of all particles:
        for p in range(self.population_size):
            self.particles[p]['coefs'] += self.particles[p]['velocity']
            
            if self.fitness_calculation == 'probability':
                self.particles[p]['coefs'] = np.maximum(self.negones, self.particles[p]['coefs'])
                self.particles[p]['coefs'] = np.minimum(self.posones, self.particles[p]['coefs'])
                
            elif self.fitness_calculation == 'importance':
                self.particles[p]['coefs'] = np.maximum(self.negones, self.particles[p]['coefs'])
                self.particles[p]['coefs'] = np.minimum(self.posones, self.particles[p]['coefs'])
            
            #print 'DIFF:', self.particles[p]['coefs'] - self.particles[p]['best_coefs']

    def fitness(self, coefs):
        
        correct = []
        distances = []
        
        if self.fitness_calculation == 'probability':
            coef_sign = np.sign(coefs)
            coef_abs = np.absolute(coefs)
            rands = np.random.random_sample(size=(len(self.useX), self.particle_length))
        
        for iter, (tX, tY) in enumerate(zip(self.useX, self.useY)):
            
            if self.fitness_calculation == 'multiply':
                predictors = coefs*tX
                prediction = np.sum(predictors)
                
            elif self.fitness_calculation == 'probability':  
                predictors = tX*(rands[iter] < coef_abs)*coef_sign
                prediction = np.sum(predictors)
                
            elif self.fitness_calculation == 'importance':
                predictors = coefs*tX
                prediction = np.sum(predictors) / np.sum(coefs)

                
            if np.sign(prediction) == np.sign(tY):
                correct.append(1.)
            else:
                correct.append(0.)
                
            
            if self.fitness_calculation == 'multiply':
                distances.append(self.calculate_multiply_distance(float(tY), prediction))
            
            elif self.fitness_calculation == 'probability':
                distances.append(self.calculate_probability_distance(float(tY), prediction))
                
            elif self.fitness_calculation == 'importance':
                distances.append(self.calculate_importance_distance(float(tY), prediction))


        
        avg_correct = float(sum(correct))/len(correct)
        avg_distances = sum(distances)/len(distances)
            
        return avg_correct, avg_distances
    
        def calculate_multiply_distance(self, Y, prediction):
        
        if self.harsh_scoring:
            if np.sign(Y) != np.sign(prediction):
                return 0.
        
        # calculate the absolute distance from the target:
        absolute_dist = abs(Y - prediction)
        
        # perfect distance is 0., ambivalence is 0.5, wrong is beyond that
        # use a hyperbolic function on dat distance for inverse fitness:
        justified_dist = absolute_dist*2.
        inverse_dist = 1./justified_dist
        
        
        #square_dist = absolute_dist*absolute_dist
        #return 1./square_dist
        
        return inverse_dist
    
    
    def calculate_probability_distance(self, Y, prediction):
        
        abs_pred = abs(prediction)
        if abs_pred > 7.:
            abs_pred = 7.
        
        if np.sign(Y) == np.sign(prediction):
            return 1. / (1. + np.exp(-1.*abs_pred))
        else:
            return 1. / (1. + np.exp(1.*abs_pred))
            
        
            
    
    def calculate_importance_distance(self, Y, prediction):
        
        abs_pred = abs(prediction)
        if abs_pred > 7.:
            abs_pred = 7.
            
        correct = np.sign(Y) == np.sign(prediction)
        
        if correct:
            imp = 1. / (1. + np.exp(-1.*abs_pred))
        else:
            imp = 1. / (1. + np.exp(1.*abs_pred))
            
        if self.harsh_scoring:
            if correct:
                return 1. + (imp - 0.5)
            else:
                return 0. - (0.5 - imp)
        else:
            if correct:
                return imp - 0.5
            else:
                return 0. - (0.5 - imp)
                
                
    def accuracy_compensator(self, accuracies):
        
        if len(accuracies) < self.accuracy_lookback:
            return 0.
        else:
            avg_acc = sum(accuracies[-int(self.accuracy_lookback):])/self.accuracy_lookback
            avg_acc -= 0.50 # chance justified
            acc_comp = avg_acc * self.accuracy_compensation_coef
            return acc_comp
    
    
        