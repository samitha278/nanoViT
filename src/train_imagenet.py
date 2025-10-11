import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset




# Data pre


# Load dataset (dataset object : "train" : 126689 , "val" : 5000 splits , 100 classes) 
dataset = load_dataset("/home/samitha/Projects/datasets/imagenet100")


# Image Transformation with Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(num_ops=2, magnitude=9),   # ViT style
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class DataLoader(nn.Module):
    def __init__(self,data,transform,batch_size):
        super().__init__()
        
        self.data = data
        self.transform = transform
        self.batch_size = batch_size
        
        self.count = 0
        
    def forward(self):
        
        batch = self.data[self.count : self.count + self.batch_size]  # taking batch
        self.count += self.batch_size   # update for next batch
        
        
        xb = [self.transform(img) for img in batch['image']]
        yb = batch['label']
        
        
        
        
        
        
        
        
    





# Train 