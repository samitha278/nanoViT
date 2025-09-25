import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dataclasses import dataclass

from vit import ViT,ViTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'




#_______________________________________________________________________________



torch.manual_seed(278)
if torch.cuda.is_available():
  torch.cuda.manual_seed(278)


max_iter = 100
lr = 3e-4


config = ViTConfig(img_size=28,im_channels=1,patch_size=4)
vit = ViT()
vit = vit.to(device)
vit = torch.compile(vit)




#_______________________________________________________________________________

#Get data

transform_train = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform_train)


train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)


#_______________________________________________________________________________





#Trianer

optimizer = torch.optim.AdamW(vit.parameters(), lr = lr )

losses = torch.zeros((max_iter,))

train_iter = iter(train_data)

for i in range(max_iter):

  xb,yb = next(train_iter)
  xb , yb = xb.to(device),yb.to(device)

  logits , loss = vit(xb,yb)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  losses[i] = loss.item()

  if i%10==0 : print(f'{i}/{max_iter}   {loss.item()}')

