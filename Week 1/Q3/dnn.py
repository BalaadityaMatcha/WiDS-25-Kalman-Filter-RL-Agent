import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


# train = datasets.MNIST("", train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ]))
# test = datasets.MNIST("", train=False, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ]))

# trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
# testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

class DigitClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.i = nn.Linear(784, 256)
    self.l1 = nn.Linear(256, 128)
    self.l2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.i(x))
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return x

dc = DigitClassifier()
optimizer = optim.Adam(dc.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    dc.train()
    # for data in trainset:
    for data in train_loader:
        x, y = data
        dc.zero_grad()
        out = dc(x.view(-1, 784))
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    print(loss.item())


correct = 0
total = 0

dc.eval()
with torch.no_grad():
  # for data in trainset:
  for data in train_loader:
    x, y = data
    out = dc(x.view(-1, 784))
    
    for idx, i in enumerate(out):
      if torch.argmax(i) == y[idx]:
        correct += 1
      total += 1
print("Train accuracy:", correct/total)

correct = 0
total = 0

with torch.no_grad():
  # for data in testset:
  for data in val_loader:
    x, y = data
    out = dc(x.view(-1, 784))
    
    for idx, i in enumerate(out):
      if torch.argmax(i) == y[idx]:
        correct += 1
      total += 1
print("Test accuracy:", correct/total)