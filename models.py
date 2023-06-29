import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MetaLearner(nn.Module): # stacking
    def __init__(self, num_classes, num_modules, num_layers=2, activation='relu'):
        super(MetaLearner, self).__init__()
        self.num_classes = num_classes
        self.num_modules = num_modules
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':    
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh

        layers = []

        if num_layers == 2:
            layers.append(nn.Linear(num_modules * num_classes, num_modules * num_classes))
            layers.append(nn.Linear(num_modules * num_classes, num_classes))
        elif num_layers == 1:
            layers.append(nn.Linear(num_modules * num_classes, num_classes))
        else:
            raise NotImplementedError
        
        for layer in layers[:-1]: # start with zero bias
            nn.init.zeros_(layer.bias)
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x.reshape((-1, self.num_modules * self.num_classes))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        x = F.softmax(x, dim=1) 
        return x



class Ensemble(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10, initial=16, num_modules=20):
        super(Ensemble, self).__init__()
        self.num_classes = num_classes
        self.num_modules = num_modules

        layers = []
        self.num_classes = num_classes
        self.initial = initial
        # block 1
        layers.append(nn.Conv2d(num_modules * input_shape[0], num_modules*initial, 3, padding=1, groups=num_modules))
        layers.append(nn.Conv2d(num_modules*initial, num_modules*initial, 3, padding=1, groups=num_modules))

        # block 2
        layers.append(nn.Conv2d(num_modules*initial, num_modules*initial*2, 3, padding=1, groups=num_modules))
        layers.append(nn.Conv2d(num_modules*initial*2, num_modules*initial*2, 3, padding=1, groups=num_modules))

        # block 3
        layers.append(nn.Conv2d(num_modules*initial*2, num_modules*initial*4, 3, padding=1, groups=num_modules))
        layers.append(nn.Conv2d(num_modules*initial*4, num_modules*initial*4, 3, padding=1, groups=num_modules))

        self.layers = nn.ModuleList(layers)
        self.fc1 = ModularLinear(initial*4*4*4, initial*4*2, num_modules)
        self.fc2 = ModularLinear(initial*4*2, num_classes, num_modules)

    def get_weights(self):
        return self.layers, self.fc1, self.fc2


    def forward(self, x, eval=False):
        x = x.repeat(1, self.num_modules, 1, 1) # duplicate input for each module
        n_examples = x.shape[0]
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if i % 2 == 1: # maxpool every other layer
                x = F.max_pool3d(x, (1, 2, 2)) # equivalent to groupwise maxpool2d(3,3)
                if not eval:
                    x = F.dropout(x, 0.2)

        # Turning from 4D: N, M, K x K into 3D tensor M x N x K
        # N examples, M modules, K classes / features
        conv2_3D = x.view(n_examples, self.num_modules, -1).permute(1, 0, 2)
        x = self.fc1(conv2_3D)
        x = self.fc2(x)
        x = F.softmax(x, dim=2)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class ModularLinear(nn.Module):
    """Custom linear layer that allows for multiple modules (groups)"""
    def __init__(self, in_features, out_features, num_modules):
        super(ModularLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules

        self.weight = nn.Parameter(torch.Tensor(self.num_modules, self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.Tensor(self.num_modules, 1, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        in_features = self.weight.size(1)
        std_dev = 1. / math.sqrt(in_features)
        self.weight.data.uniform_(-std_dev, std_dev)
        self.bias.data.uniform_(-std_dev, std_dev)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
    
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Ensemble_single(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, initial=16):
        super(Ensemble_single, self).__init__()
        self.num_classes = num_classes
        self.num_modules = 1

        layers = []
        self.num_classes = num_classes
        self.initial = initial
        # block 1
        layers.append(nn.Conv2d(input_shape[0], initial, 3, padding=1))
        layers.append(nn.Conv2d(initial, initial, 3, padding=1))

        # block 2
        layers.append(nn.Conv2d(initial, initial*2, 3, padding=1))
        layers.append(nn.Conv2d(initial*2, initial*2, 3, padding=1))

        # block 3
        layers.append(nn.Conv2d(initial*2, initial*4, 3, padding=1))
        layers.append(nn.Conv2d(initial*4, initial*4, 3, padding=1))

        # block 4
        self.layers = nn.ModuleList(layers)
        self.fc1 = ModularLinear(initial*4*4*4, num_classes, 1)
        self.fc2 = ModularLinear(num_classes, num_classes, 1)
        

    def set_weights(self, weight_dict, idx):
        # could be done better... without hardcoding
        self.layers[0].weight.data = weight_dict['layers.0.weight'][16*idx:16*(1+idx),:,:,:] # 128 x 3 x 3 x 3
        self.layers[0].bias.data = weight_dict['layers.0.bias'][16*idx:16*(1+idx)]
        self.layers[1].weight.data = weight_dict['layers.1.weight'][16*idx:16*(1+idx),:,:,:] # 128 x 16 x 3 x 3
        self.layers[1].bias.data = weight_dict['layers.1.bias'][16*idx:16*(1+idx)]
        self.layers[2].weight.data = weight_dict['layers.2.weight'][32*idx:32*(1+idx),:,:,:] # 256 x 16 x 3 x 3
        self.layers[2].bias.data = weight_dict['layers.2.bias'][32*idx:32*(1+idx)]
        self.layers[3].weight.data = weight_dict['layers.3.weight'][32*idx:32*(1+idx),:,:,:] # 256 x 32 x 3 x 3
        self.layers[3].bias.data = weight_dict['layers.3.bias'][32*idx:32*(1+idx)]
        self.layers[4].weight.data = weight_dict['layers.4.weight'][64*idx:64*(1+idx),:,:,:] # 512 x 32 x 3 x 3
        self.layers[4].bias.data = weight_dict['layers.4.bias'][64*idx:64*(1+idx)]
        self.layers[5].weight.data = weight_dict['layers.5.weight'][64*idx:64*(1+idx),:,:,:] # 512 x 64 x 3 x 3
        self.layers[5].bias.data = weight_dict['layers.5.bias'][64*idx:64*(1+idx)]

        self.fc1.weight.data = weight_dict['fc1.weight'][idx]
        self.fc1.bias.data = weight_dict['fc1.bias'][idx]
        self.fc2.weight.data = weight_dict['fc2.weight'][idx]
        self.fc2.bias.data = weight_dict['fc2.bias'][idx]

    def forward(self, x, eval=False):
        n_examples = x.shape[0]
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if i % 2 == 1: # maxpool after every other layer
                x = F.max_pool3d(x, (1, 2, 2)) # equivalent to groupwise maxpool2d((2,2))
                if not eval:
                    x = F.dropout(x, 0.2)

        # Turning from 4D: N, M, K x K into 3D tensor M x N x K
        # N examples, M modules, K classes / features
        conv2_3D = x.view(n_examples, self.num_modules, -1).permute(1, 0, 2)
        x = self.fc1(conv2_3D)
        x = self.fc2(x)
        x = F.softmax(x, dim=2)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

