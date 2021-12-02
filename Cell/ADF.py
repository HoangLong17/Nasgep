import random
import math
from Tree import *
ADF_function = ['sum']
ADF_terminal = ['i-1', 'i-2']
conv = ['dep 3x3', 'dep 5x5', 'dep 3x5', 'dep 5x3', 'dep 1x7', 'dep 7x1',
        'sep 3x3', 'sep 5x5', 'sep 3x5', 'sep 5x3', 'sep 1x7', 'sep 7x1',
        'isep 3x3', 'isep 5x5', 'isep 3x5', 'isep 5x3', 'isep 1x7', 'isep 7x1']
class ADF:
	def __init__(self, gen_size):
		self.num_function = int((gen_size-1)/2)
		self.num_terminal = int((gen_size+1)/2)
		self.data = self.init_data()
		self.tree = buildTree(self.data)

	def init_data(self):
		function = [*ADF_function, *conv]
		terminal = [*ADF_terminal, *conv]
		data = []
		height = int(math.log(self.num_function+self.num_terminal,2))+1
		#print('Height: ', height)
		for i in range(self.num_function):
			if i>0:
				if data[int((i-1)/2)].data==None or (data[int((i-1)/2)].data in conv and i%2==0):    #decrease num of functions
					data.append(Node(None))
					continue
			index = random.randint(0, len(function)-1)
			n = Node(function[index])
			data.append(n)
			if function[index] in conv:    #conv has only one child
				#print('floor: ', int(math.log(i+1,2))+1)
				#a = int(math.pow(2, height - int(math.log(i+1,2))-2)-1) 
				b = int(math.pow(2, height - int(math.log(i+1,2))-2)) 
				#print('a: ',a)
				#print('b: ',b)
				#self.num_function -= a  #cut branch of functions
				self.num_terminal -= b     #cut branch of terminals
		#print('num function: ', self.num_function)
		#print('num terminal: ', self.num_terminal)
		res = []
		for i in range(len(data)):
			if data[i].data != None:
				res.append(data[i])
		for i in range(self.num_function, self.num_function+self.num_terminal):
			index = random.randint(0, len(terminal)-1)
			if terminal[index] in conv:  #conv needs an input
				n = Node(terminal[index])
				ind = random.randint(0, len(ADF_terminal)-1)
				n.left = Node(ADF_terminal[ind])
				res.append(n)
			else:
				res.append(Node(terminal[index]))
		return res


	#def init_function(self):
	#	tmp = [*ADF_function, *conv]
	#	function = []
	#	for i in range(self.num_function):
	#		index = random.randint(0, len(tmp)-1)
	#		function.append(tmp[index])
	#	return function

	#def init_terminal(self):
	#	tmp = [*ADF_terminal, *conv]
	#	terminal = []
	#	for i in range(self.num_terminal):
	#		index = random.randint(0, len(tmp)-1)
	#		terminal.append(tmp[index])
	#		if tmp[index] in conv:
	#			index = random.randint(0, len(ADF_terminal)-1)
	#			terminal.append(ADF_terminal[index])
	#	return terminal

	#def init_leaves(self):
	#	leaves = getLeaves(self.tree)
	#	ter = iter(self.terminal)
	#	for leaf in leaves:
	#		n = next(ter)
	#		if n == None: break
	#		leaf.left = Node(n)
	#		if n in conv:
	#			n = next(ter)
	#			leaf.left.left = Node(n)
	#		n = next(ter)
	#		if n == None: break
	#		leaf.right = Node(n)
	#		if n in conv:
	#			n = next(ter)
	#			leaf.right.left = Node(n)

	def __str__(self):
		return f'<{self.tree.data} \n {self.tree.left} \n {self.tree.right}>'


if __name__=="__main__":
	adf = ADF(15)
	#print(adf)
	drawTree(adf.tree)
