# encoding=utf8

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as st

# data transformer
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# prepare train set
train_set = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    download=False,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
# prepare test set
test_set = torchvision.datasets.CIFAR10(
    root='./data/',
    train=False,
    download=False,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

# prepare classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5
    npimp = img.numpy()
    plt.imshow(np.transpose(npimp, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# st(context=21)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s ' % classes[labels[j]] for j in range(4)))


# define convolution nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3@32*32 → 6@28*28 (via 5*5 square convolution)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6@14*14 → 16@10*10
        self.conv2 = nn.Conv2d(6, 16, 5)
        # full connection
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # sub sampling pooling
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# define a Loss function and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train the network
st(context=21)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs = Variable(inputs)
        labels = Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        # print tmp statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')
