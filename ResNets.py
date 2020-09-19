import torch
from torch import nn
from torchvision import datasets, transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 3e-3

# Image preprocessing
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# CIFAR-10 dataset
trainset = datasets.CIFAR10(
    root="../Data/",
    train=True,
    transform=transform,
    download=True
)

testset = datasets.CIFAR10(
    root="../Data/",
    train=False,
    transform=transform,
    download=True
)

# Data loader
trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)

# Residual block
class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, 
                 downsample=False, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               3, strides, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1)
        if downsample:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   1, strides)
        else:
            self.conv3 = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        res = x
        x = self.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        if self.conv3:
            res = self.conv3(res)
        x += res
        return self.relu(x)

# ResNets
class ResNet(nn.Module):
    
    def __init__(self, num_residuals, layer_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, layer_channels[0], 7, 2, 3),
            nn.BatchNorm2d(layer_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.blocks = self.make_layers(num_residuals, layer_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[-1], num_classes)
    
    def make_layers(self, num_residuals, layer_channels):
        layers = []
        for idx, value in enumerate(num_residuals):
            for b in range(value):
                if b == 0 and idx != 0:
                    layers.append(ResBlock(layer_channels[idx-1], layer_channels[idx],
                                           downsample=True, strides=2))
                else:
                    layers.append(ResBlock(layer_channels[idx], layer_channels[idx]))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Model architecture    
num_residuals = [2, 2, 2, 2]
layer_channels = [64, 128, 256, 512]
num_classes = 10

# Model
model = ResNet(num_residuals, layer_channels, num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss after {}th epoch: {:.3f}'.format(epoch, loss.item()))

    # Learning rate decay
    if (epoch+1)%10 == 0:

        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy: {}%'.format(100 * correct/total))