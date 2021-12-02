from collections import deque
conv = ['dep 3x3', 'dep 5x5', 'dep 3x5', 'dep 5x3', 'dep 1x7', 'dep 7x1',
        'sep 3x3', 'sep 5x5', 'sep 3x5', 'sep 5x3', 'sep 1x7', 'sep 7x1',
        'isep 3x3', 'isep 5x5', 'isep 3x5', 'isep 5x3', 'isep 1x7', 'isep 7x1']
class Node(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def __str__(self):
        return f'<{self.data} \n {self.left} \n {self.right}>'

def buildTree(data):
    n = iter(data)
    tree = next(n)
    fringe = deque([tree])
    while(True):
        head = fringe.popleft()
        try:
            head.left = next(n)
            fringe.append(head.left)
            if head.data not in conv:
                head.right = next(n)
                fringe.append(head.right)
        except StopIteration:
            break
    return tree

def getLeaves(tree):
	leaves = []
	if tree == None: return leaves
	if tree.left == None:
		leaves.append(tree)
		return leaves
	leftLeaves = getLeaves(tree.left)
	rightLeaves = getLeaves(tree.right)
	leaves = [*leftLeaves, *rightLeaves]
	return leaves

def getConv(tree):
    conv_layer = set()
    if tree == None: return conv_layer
    if tree.data in conv: 
        conv_layer.add(tree.data)
    left_conv = getConv(tree.left)
    right_conv = getConv(tree.right)
    conv_layer.update(left_conv)
    conv_layer.update(right_conv)
    return conv_layer

def drawTree(root):
    def height(root):
        return 1 + max(height(root.left), height(root.right)) if root else -1
    def jumpto(x, y):
        t.penup()
        t.goto(x, y)
        t.pendown()
    def draw(node, x, y, dx):
        if node:
            t.goto(x, y)
            jumpto(x, y-20)
            t.write(node.data, align='center', font=('Arial', 12, 'normal'))
            draw(node.left, x-dx, y-60, dx/2)
            jumpto(x, y-20)
            draw(node.right, x+dx, y-60, dx/2)
    import turtle
    t = turtle.Turtle()
    t.speed(0); turtle.delay(0)
    h = height(root)
    jumpto(0, 30*h)
    draw(root, 0, 30*h, 40*h)
    t.hideturtle()
    turtle.mainloop()