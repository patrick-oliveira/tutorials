import torch
import torch.nn as nn

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, X):
        return self.activation(self.batch_norm(self.conv(X)))
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        #__Adaptive average pooling__ is simply an average pooling operation that, given an input and output dimensionality, calculates the correct kernel size necessary to produce an output of the given dimensionality from the given input.

        self.conv = nn.Conv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, X):
        X = self.pool(X)
        X = self.conv(X)
        X = self.act(X)
        X = torch.flatten(X, 1)
        X = self.fc1(X)
        X = self.act(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pp):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_1x1, kernel_size = 1, stride = 1, padding = 0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size = 1, stride=  1, padding = 0),
            ConvBlock(f_3x3_r, f_3x3, kernel_size = 3, stride = 1, padding = 1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_5x5_r, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(f_5x5_r, f_5x5, kernel_size = 5, stride = 1, padding = 2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride = 1, padding = 1, ceil_mode = True),
            ConvBlock(in_channels, f_pp, kernel_size = 1, stride = 1, padding = 0)
        )
    
    def forward(self, X):
        return torch.cat([
            self.branch1(X),
            self.branch2(X),
            self.branch3(X),
            self.branch4(X)
        ], 1)

class Inception(nn.Module):
    def __init__(self, num_classes = 10):
        super(Inception, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.inception3A = InceptionModule(in_channels=192,
                                           f_1x1=64,
                                           f_3x3_r=96,
                                           f_3x3=128,
                                           f_5x5_r=16,
                                           f_5x5=32,
                                           f_pp=32)
        self.inception3B = InceptionModule(in_channels=256,
                                           f_1x1=128,
                                           f_3x3_r=128,
                                           f_3x3=192,
                                           f_5x5_r=32,
                                           f_5x5=96,
                                           f_pp=64)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.inception4A = InceptionModule(in_channels=480,
                                           f_1x1=192,
                                           f_3x3_r=96,
                                           f_3x3=208,
                                           f_5x5_r=16,
                                           f_5x5=48,
                                           f_pp=64)
        self.inception4B = InceptionModule(in_channels=512,
                                           f_1x1=160,
                                           f_3x3_r=112,
                                           f_3x3=224,
                                           f_5x5_r=24,
                                           f_5x5=64,
                                           f_pp=64)
        self.inception4C = InceptionModule(in_channels=512,
                                           f_1x1=128,
                                           f_3x3_r=128,
                                           f_3x3=256,
                                           f_5x5_r=24,
                                           f_5x5=64,
                                           f_pp=64)
        self.inception4D = InceptionModule(in_channels=512,
                                           f_1x1=112,
                                           f_3x3_r=144,
                                           f_3x3=288,
                                           f_5x5_r=32,
                                           f_5x5=64,
                                           f_pp=64)
        self.inception4E = InceptionModule(in_channels=528,
                                           f_1x1=256,
                                           f_3x3_r=160,
                                           f_3x3=320,
                                           f_5x5_r=32,
                                           f_5x5=128,
                                           f_pp=128)
        self.pool5 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.inception5A = InceptionModule(in_channels=832,
                                           f_1x1=256,
                                           f_3x3_r=160,
                                           f_3x3=320,
                                           f_5x5_r=32,
                                           f_5x5=128,
                                           f_pp=128)
        self.inception5B = InceptionModule(in_channels=832,
                                           f_1x1=384,
                                           f_3x3_r=192,
                                           f_3x3=384,
                                           f_5x5_r=48,
                                           f_5x5=128,
                                           f_pp=128)
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        self.aux4A = InceptionAux(512, num_classes) 
        self.aux4D = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.inception3A(x)
        x = self.inception3B(x)
        x = self.pool4(x)
        x = self.inception4A(x)
        aux1 = self.aux4A(x)
        x = self.inception4B(x)
        x = self.inception4C(x)
        x = self.inception4D(x)
        aux2 = self.aux4D(x)
        x = self.inception4E(x)
        x = self.pool5(x)
        x = self.inception5A(x)
        x = self.inception5B(x)
        x = self.pool6(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x, aux1, aux2
    
def googlenet_training_routine(model, inputs, labels, criterion, optimizer, scheduler = None):
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs, aux1, aux2 = model(inputs)

        # Compute the loss.
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux1, labels)
        loss3 = criterion(aux2, labels)
        loss  = loss1 + 0.3 * loss2 + 0.3 * loss3 
        cumulative_loss = loss.data.item() * inputs.size(0)

        # Backward pass.
        loss.backward()

        # Optimize.
        optimizer.step()

        # Decrease learning rate.
        if scheduler != None:
            scheduler.step()

        # Compute training accuracy.
        _, predictions = torch.max(outputs, 1)
        cumulative_hits = (predictions == labels.data).float().sum().item()

        del(inputs); del(labels)
        return cumulative_loss, cumulative_hits
    
def googlenet_validation_routine(model, inputs, labels, criterion):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, aux1, aux2 = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux1, labels)
        loss3 = criterion(aux2, labels)
        loss  = loss1 + 0.3 * loss2 + 0.3 * loss3 
        cumulative_loss = loss.data.item() * inputs.size(0)
        _, predictions = torch.max(outputs, 1)
        cumulative_hits = (predictions == labels.data).float().sum().item()
        del(inputs); del(labels)
        return cumulative_loss, cumulative_hits