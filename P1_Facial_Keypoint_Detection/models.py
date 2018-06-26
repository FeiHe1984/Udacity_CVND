## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        # 2nd conv layer:
        self.conv2 = nn.Conv2d(32, 48, 5,  stride=2)
        # 3rd conv layer:
        self.conv3 = nn.Conv2d(48, 64, 3,  stride=1)
        # 4nd conv layer:
        self.conv4 = nn.Conv2d(64, 64, 3,  stride=1)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # MaxPool layer
        self.pool = nn.MaxPool2d(2,2)       
        # FC layer 1:
        self.fc1 = nn.Linear(64*1*1, 500)       
        # Dropout layer:
        self.fc1_drop = nn.Dropout(p = 0.5)
        # Dropout layer:
        self.fc2_drop = nn.Dropout(p = 0.3)
        # FC layer 2:
        self.fc2 = nn.Linear(500, 256) 
        # FC layer 3:
        self.fc3 = nn.Linear(256, 136) 
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(48) 
        self.bn2 = nn.BatchNorm2d(64)  
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(self.bn1(F.leaky_relu(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(self.bn2(F.leaky_relu(self.conv4(x))))
        
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layer with dropout layer
        x = F.leaky_relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
