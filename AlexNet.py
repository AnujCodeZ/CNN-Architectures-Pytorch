import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

IMAGES_PATH = 'image_path'

transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(227),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor()])

trainset = datasets.ImageFolder(IMAGES_PATH+str('train/'), transform=transform)
testset = datasets.ImageFolder(IMAGES_PATH+str('test/'), transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.MaxPool2d(3, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2),
            nn.MaxPool2d(3, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.layer4 = nn.Conv2d(384, 384, 3, padding=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.layer6 = nn.Linear(256*6*6, 4096)
        self.layer7 = nn.Linear(4096, 4096)
        self.layer8 = nn.Linear(4096, 1000)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.dropout(x)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.dropout(x)
        x = F.relu(self.layer5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer6(x))
        x = F.dropout(x)
        x = F.relu(self.layer7(x))
        x = self.layer8(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(10):
    for images, labels in trainloader:
        
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch: {e+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.4f}')
        