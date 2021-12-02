import random
from ADF import *
from Tree import *
Cell_function = ['sum', 'concat']


class Cell:
	def __init__(self, cell_size, ADF_population):
		self.num_function = int((cell_size-1)/2)
		self.num_terminal = int((cell_size+1)/2)
		self.data = self.init_data(ADF_population)
		self.tree = buildTree(self.data)
	
	def init_data(self, ADF_population):
		data = []
		for i in range(self.num_function): 
			if i != 0 and data[int((i-1)/2)].data == 'sum':  #trick =)))
				data.append(Node('sum'))
				continue
			index = random.randint(0, len(Cell_function)-1)
			data.append(Node(Cell_function[index]))
		for i in range(self.num_function, self.num_function+self.num_terminal):
			index = random.randint(0, len(ADF_population)-1)
			data.append(ADF_population[index].tree)
		return data

	#def init_function(self):
	#	function = []
	#	for i in range(self.num_function): 
	#		if i != 0 and function[int((i-1)/2)] == 'add':
	#			function.append('add')
	#			continue
	#		index = random.randint(0, len(Cell_function)-1)
	#		function.append(Cell_function[index])
	#	return function

	#def init_terminal(self, ADF_population):
	#	terminal = []
	#	for i in range(self.num_terminal):
	#		index = random.randint(0, len(ADF_population)-1)
	#		terminal.append(ADF_population[index])
	#	return terminal

	#def init_leaves(self):
	#	leaves = getLeaves(self.tree)
	#	ter = iter(self.terminal)
	#	for leaf in leaves:
	#		n = next(ter)
	#		if n == None: break
	#		leaf.left = Node(n)
	#		n = next(ter)
	#		if n == None: break
	#		leaf.right = Node(n)



	def __str__(self):
		return f'<{self.tree.data} \n {self.tree.left} \n {self.tree.right}>'

if __name__=="__main__":
	ADF_population = [ADF(3), ADF(3), ADF(3)]
	cell = Cell(9, ADF_population)
	layers = getConv(cell.tree)
	print(layers)
	for layer in layers:
		print(layer)
	#for adf in ADF_population:
	#	print('ADF: ', adf)
	#print('Cell: ',cell)
	drawTree(cell.tree)