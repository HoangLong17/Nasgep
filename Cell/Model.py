import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from NasConv import *
from Cell import *
from Tree import *

function = ['sum', 'concat']
terminal = ['i-1', 'i-2']
conv = ['dep 3x3', 'dep 5x5', 'dep 3x5', 'dep 5x3', 'dep 1x7', 'dep 7x1',
        'sep 3x3', 'sep 5x5', 'sep 3x5', 'sep 5x3', 'sep 1x7', 'sep 7x1',
        'isep 3x3', 'isep 5x5', 'isep 3x5', 'isep 5x3', 'isep 1x7', 'isep 7x1']


class Nasgep(nn.Module):
    def __init__(self, cell, outputChannel):
        super(Nasgep, self).__init__()
        self.tree = cell.tree
        self.outputChannel = outputChannel
        self.conv_layers = getConv(cell.tree)
        self.layers = nn.ModuleList([NasConv(layer,3,3) for layer in self.conv_layers])
        self.pointWises = nn.ModuleList([nn.Conv2d(3, outputChannel, 1), nn.Conv2d(6, 3, 1)])
        #self.layers = nn.ModuleList([])
        #self.pointWises = nn.ModuleList([nn.LazyConv2d(outputChannel, 1)])
        self.fc1 = nn.Linear(3072, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        tree = self.tree
        x = self.browseTree(tree, x)
        x = self.pointWises[0](x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def browseTree(self, tree, x):
        if tree.data in function:
            if tree.data == 'sum':
                y1 = self.browseTree(tree.left, x)
                y2 = self.browseTree(tree.right, x)
                #if y1.shape[1] < y2.shape[2]:
                #    inputChannel = y1.shape[2]
                #    self.pointWises.append(NasConv("point 1x1", inputChannel, y2.shape[1]))
                #    id = len(self.layers) - 1
                #    y2 = self.layers[id](y2)
                #if y1.shape[1] > y2.shape[2]:
                #    inputChannel = y2.shape[2]
                #    self.pointWises.append(NasConv("point 1x1", inputChannel, y1.shape[1]))
                #    id = len(self.layers) - 1
                #    y1 = self.layers[id](y1)
                return y1 + y2
            if tree.data == 'concat':
                y1 = self.browseTree(tree.left, x)
                y2 = self.browseTree(tree.right, x)
                y =  torch.cat((y1, y2), dim=2)
                y = self.pointWises[1](y)
                return y
        if tree.data in conv:
            #layer = tree.data
            #inputChannel = x.shape[1]
            #self.layers.append(NasConv(layer, inputChannel, self.outputChannel))
            #id = len(self.layers) - 1
            #return self.layers[id](x)
            for id,val in enumerate(self.conv_layers):
                if tree.data == val: 
                    index = id
            return self.layers[id](self.browseTree(tree.left, x))
        if tree.data in terminal:
            return x

#def getLayer(layer):
#    if layer[0:4] == 'dep ':
#       x_kernel = int(layer[-3])
#       y_kernel = int(layer[-1])
#       return dep(1, x_kernel, y_kernel)
#    elif layer[0:4] == 'sep ':
#       x_kernel = int(layer[-3])
#       y_kernel = int(layer[-1])
#       return sep(1, x_kernel, y_kernel)
#    elif layer[0:4] == 'isep':
#       x_kernel = int(layer[-3])
#       y_kernel = int(layer[-1])
#       return isep(1, x_kernel, y_kernel)
#def get_input():
#    img = cv2.imread("./images/mach.png", cv2.imread_color)
#    x = np.asarray(img)
#    x = np.transpose(x, (2,0,1))
#    x = np.expand_dims(x, axis=0)
#    x = torch.tensor(x)
#    return x

#def save_output(y):
#    print(y.shape)
#    y = torch.squeeze(y, 0)
#    #y = torch.cat((y, y, y), dim=0)
#    print(y.shape)
#    y = y.detach().numpy()
#    y = np.transpose(y, (1,2,0))
#    print(y.shape)
#    image = image.fromarray((y * 255).astype(np.uint8))
#    image.save("./images/test.png")
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def imshow(img):
    print(img.shape)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ADF_population = [ADF(3), ADF(3), ADF(3)]
    cell = Cell(9, ADF_population)
    save_object(cell, './cell.pkl')

    model = Nasgep(cell, 3)
    model.to(device)
    
    #x = get_input()
    #y = model(x)
    #save_output(y)
    #drawTree(cell.tree)

    optimizer = torch.optim.SGD(model.parameters(), lr= 1e-4)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat',
    #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()

    ## show images
    #imshow(torchvision.utils.make_grid(images))
    ## print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_model.pth'
    torch.save(model.state_dict(), PATH)

    print(model)