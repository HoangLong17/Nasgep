import torch

class NasConv(torch.nn.Module):
    def __init__(self, layer, inputChannel, outputChannel=1):
        super(NasConv, self).__init__()
        x_kernel = int(layer[-3])
        y_kernel = int(layer[-1])
        x_pad = int((x_kernel-1)/2)
        y_pad = int((y_kernel-1)/2)
        if layer[0:4] == 'dep ':
            self.layer = torch.nn.Conv2d(inputChannel, inputChannel, (x_kernel, y_kernel), padding=(x_pad, y_pad), groups=inputChannel)
        elif layer[0:4] == 'sep ':
            depth_conv = torch.nn.Conv2d(inputChannel, inputChannel, (x_kernel, y_kernel), padding=(x_pad, y_pad), groups=inputChannel)
            point_conv = torch.nn.Conv2d(inputChannel, outputChannel, (1, 1))
            self.layer = torch.nn.Sequential(depth_conv, point_conv)
        elif layer[0:4] == 'isep':
            depth_conv = torch.nn.Conv2d(inputChannel, inputChannel, (x_kernel, y_kernel), padding=(x_pad, y_pad), groups=inputChannel)
            point_conv = torch.nn.Conv2d(inputChannel, outputChannel, (1, 1))
            self.layer = torch.nn.Sequential(point_conv, depth_conv)
        else:
            self.layer = torch.nn.Conv2d(inputChannel, outputChannel, (1, 1))
    
    def forward(self, x):
        y = self.layer(x)
        return y


#def dep(inputChannel, x_kernel, y_kernel):
#    x_pad = int((x_kernel-1)/2)
#    y_pad = int((y_kernel-1)/2)
#    #depth_conv = torch.nn.Conv2d(inputChannel, inputChannel, (x_kernel, y_kernel), padding=(x_pad, y_pad), group=inputChannel)
#    depth_conv = torch.nn.Conv2d(inputChannel, 1, (x_kernel, y_kernel), padding=(x_pad, y_pad), groups=1)
#    return depth_conv

#def pointWise(inputChannel, ouputChannel=1):
#    point_conv = torch.nn.Conv2d(inputChannel, ouputChannel, (1, 1))
#    return point_conv

#def sep(inputChannel, x_kernel, y_kernel):
#    depth_conv = dep(inputChannel, x_kernel, y_kernel)
#    point_conv = pointWise(inputChannel)
#    seperate_conv = torch.nn.Sequential(depth_conv, point_conv)
#    return seperate_conv

#def isep(inputChannel, x_kernel, y_kernel):
#    depth_conv = dep(inputChannel, x_kernel, y_kernel)
#    point_conv = pointWise(inputChannel)
#    inverse_seperate_conv = torch.nn.Sequential(point_conv, depth_conv)
#    return inverse_seperate_conv

