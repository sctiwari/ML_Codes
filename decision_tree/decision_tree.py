import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return

	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
			print(name + '{')

		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
			print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
			print(indent+'}')

class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted
		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
			C is the number of classes,
			B is the number of branches
			it stores the number of 
			corresponding training samples 
			e.g.
			○ ○ ○ ○
			● ● ● ●
			┏━━━━┻━━━━┓
			○ ○       ○ ○
			● ● ● ●
                                              
			branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branch_total = np.sum(branches,axis=0)
			branch_entropy = []
			for i in range(0,len(branches[0])):
				entropy = 0.0
				for j in range(0,len(branches)):
					value= float(branches[j][i])/branch_total[i]
					if (value != 0):
						entropy += -1*value * np.log(value)
				branch_entropy.append(entropy)

			total = np.sum(branch_total)
			cond_entropy = 0.0
			for i in range(0,len(branch_entropy)):
				cond_entropy += float(branch_total[i]/total) * branch_entropy[i]
			return cond_entropy


		min_entropy = 0
		min_map_branches = {}
		for idx_dim in range(len(self.features[0])):
			############################################################
			# TODO: compare each split using conditional entropy
			#       find the
			############################################################
			map_branches={}
			for branch_idx in range(0,len(self.features)):
				if self.features[branch_idx][idx_dim] not in map_branches:
					map_branches[self.features[branch_idx][idx_dim]] = len(map_branches)
           
			branches = [[0 for i in range(len(map_branches))]for j in range(self.num_cls)]

			for branch_idx in range(0, len(self.features)):
				branches[self.labels[branch_idx]][map_branches[self.features[branch_idx][idx_dim]]] += 1

			cond_entropy = conditional_entropy(branches)

			if (idx_dim == 0) or (cond_entropy < min_entropy):
				min_entropy = cond_entropy
				self.dim_split = idx_dim
				min_map_branches = map_branches

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		#print("min_entropy: ",min_entropy)
		self.feature_uniq_split = [0 for i in range(len(min_map_branches))]
		for key in min_map_branches:
			self.feature_uniq_split[min_map_branches[key]]  = key

		min_labels = [row[self.dim_split] for row in self.features]
		child_features = [None]*len(min_map_branches)
		child_labels = [None]*len(min_map_branches)
		new_features = np.delete(self.features,self.dim_split,1)


		for idx in range(0,len(min_labels)):
			if child_features[min_map_branches[min_labels[idx]]] == None:
				child_features[min_map_branches[min_labels[idx]]] = [list(new_features[idx])]
			else:
				child_features[min_map_branches[min_labels[idx]]].append(list(new_features[idx]))

			if child_labels[min_map_branches[min_labels[idx]]] == None:
				child_labels[min_map_branches[min_labels[idx]]] = [self.labels[idx]]
			else:
				child_labels[min_map_branches[min_labels[idx]]].append(self.labels[idx])

		for idx in range(0,len(min_map_branches)):
			node = TreeNode(child_features[idx],child_labels[idx],max(child_labels[idx])+1)
			if len(child_features[idx][0]) == 0:
				node.splittable = False
			self.children.append(node)


		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(int(feature[self.dim_split]))
			feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max
