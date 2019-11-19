import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []   # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		result = np.zeros(len(features))
		for ii in range(0,len(self.betas)):
			prd= self.betas[ii] * (np.array(self.clfs_picked[ii].predict(features)))
			result += prd
		for ii in range(0,len(result)):
			if result[ii] >= 0:
				result[ii] = 1
			else:
				result[ii] = -1
		return(list(result))		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		wt = np.full((N, 1), 1/N)
		list_clfs = list(self.clfs)
		for iter in range(self.T):
			min_err,min_clf,min_diff = 0, None, None
			for H in list_clfs:
				diff = (np.array(H.predict(features))!=labels)
				#print (diff)
				error = float(np.matmul(np.transpose(wt),diff))
				#print (weight)
				#print(error)
				if (min_clf == None) or (error < min_err):
					min_err,min_clf,min_diff = error, H, diff

			min_beta = 0.5 * np.log((1-min_err)/min_err)
			self.clfs_picked.append(min_clf)
			self.betas.append(min_beta)

			for ii in range(N):
				if min_diff[ii] == True:
					wt[ii] = wt[ii]*np.exp(min_beta)
				else:
					wt[ii] = wt[ii] * np.exp(-1.0 * min_beta)
			wt = wt / np.sum(wt)

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	
