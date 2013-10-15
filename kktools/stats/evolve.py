import numpy as np 
import scipy.stats

from pyevolve import GSimpleGA, G1DList, Selectors, Initializators, Mutators, Consts, Crossovers
from threshold import threshold_by_pvalue, threshold_by_rawrange






class EvolutionarySolver(object):

	def __init__(self, dataX, dataY):
		self.dataX = np.array(dataX)
		self.dataY = np.array(dataY)
		self.beta_length = self.dataX.shape[1]
		self.n_generations = 1000
		self.evolve_printouts = 1


	def evaluation_function(self, betas, sample_range=25):
		fitness = 0.0
		betas_ = np.array([x for x in betas])

		randinds = [np.random.randint(0,len(self.dataY)) for x in range(sample_range)]

		#randind = np.random.randint(0,len(self.dataY))

		'''
		for X, Y in zip(self.dataX, self.dataY):
			prediction = np.sum(np.array(X)*betas_)
			pred_sign = np.sign(prediction)
			if pred_sign == Y:
				fitness += 1.
		'''

		for rind in randinds:
			X = self.dataX[rind]
			Y = self.dataY[rind]
			prediction = np.sum(np.array(X)*betas_)
			pred_sign = np.sign(prediction)
			if pred_sign == Y:
				fitness += 1.

		#X = self.dataX[randind]
		#Y = self.dataY[randind]

		#prediction = np.sum(np.array(X)*betas_)
		#fitness = np.abs(Y - prediction)

			
		
		fitness = fitness / sample_range
		#fitness = fitness / len(self.dataY)
		#print 'fitness', fitness
		return fitness


	def initialize_genome(self, mutation_rate=0.02, population_size=20, elitism=True):

		genome = G1DList.G1DList(self.beta_length)
		genome.setParams(rangemin=-1.0, rangemax=1.0)

		genome.initializator.set(Initializators.G1DListInitializatorReal)

		genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

		genome.evaluator.set(self.evaluation_function)

		#genome.crossover.set(Crossovers.G1DListCrossoverOX)

		self.ga = GSimpleGA.GSimpleGA(genome)
		self.ga.selector.set(Selectors.GRouletteWheel)

		#self.ga.minimax = Consts.minimaxType["minimize"]

		self.ga.setMutationRate(mutation_rate)

		self.ga.setPopulationSize(population_size)

		self.ga.setGenerations(self.n_generations)

		self.ga.setElitism(elitism)


	def evolve(self):

		#print 'population size:', self.ga.size
		self.ga.evolve(freq_stats=self.evolve_printouts)

	
	def output_maps(self, data_object, nifti_filepath, time_points, threshold=0.01, two_tail=True):

		best_coefs = self.ga.bestIndividual()
		best_coefs = np.array([x for x in best_coefs])

		#print np.sum(best_coefs)
		#thresholded_coefs = threshold_by_pvalue(best_coefs, threshold, two_tail=two_tail)
		#print np.sum(thresholded_coefs)

		#unmasked = data_object.unmask_Xcoefs(thresholded_coefs, time_points)
		unmasked = data_object.unmask_Xcoefs(best_coefs, time_points)

		data_object.save_unmasked_coefs(unmasked, nifti_filepath)







