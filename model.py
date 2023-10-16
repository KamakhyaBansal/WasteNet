import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

train_path = '/content/drive/MyDrive/SatelliteImages/Data/UAV/images/'
test_path = train_path
train_mask = '/content/drive/MyDrive/SatelliteImages/Data/UAV/Mask/'
test_mask = train_mask

train_list = os.listdir(train_path)[:617]
test_list = os.listdir(train_path)[617:771]
train_dir = train_path
val_dir = test_path
train_fns = train_list
val_fns = test_list
print(len(train_fns), len(val_fns))

sample_image_fp = os.path.join(train_dir, train_fns[4])
sample_image = Image.open(sample_image_fp).convert("RGB")
sample_image = sample_image.resize((256,256))
sample_mask = Image.open(train_mask+train_fns[4]).convert("RGB")
sample_mask = sample_mask.resize((256,256))
plt.axis('off')
plt.imshow(sample_image)
plt.imshow(sample_mask,alpha=0.3)
print(sample_image_fp)


num_items = len(train_fns)
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)
print((color_array[0]))

num_classes = 2
label_model1 = KMeans(n_clusters=num_classes)
label_model1.fit(color_array)

label_class = label_model1.predict(np.asarray(sample_mask).reshape(-1, 3)).reshape(256, 256)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sample_image)
axes[1].imshow(sample_mask)
axes[2].imshow(label_class)

class CleanWasteDataset(Dataset):

    def __init__(self, image_list, mask_dir, label_model):
        self.image_dir = train_dir
        self.image_fns = image_list
        self.label_model = label_model
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')

        image = image.resize((256,256))
        image = np.asarray(image)
        image = self.transform(image)

        mask = Image.open(self.mask_dir+image_fn).convert("RGB")
        mask = mask.resize((256,256))
        mask = np.asarray(mask)
        label_class = self.label_model.predict(mask.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        return image, label_class

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform_ops(image)
dataset = CleanWasteDataset(train_fns, train_mask, label_model1)
print(len(dataset))

image,label_class = dataset[0]
print(image.shape, label_class.shape)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image.permute(1,2,0))
axes[1].imshow(label_class)

class WasteNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.scaleconv_1 = (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0))
        self.scaleconv_2 = (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.scaleconv_3 = (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2))
        self.scaleconv_4 = (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3))
        self.weighted_sum = nn.Linear(4,1)
        self.contracting_01 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_11 = self.conv_block(in_channels=64, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.scaleconv_21 = (nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0))
        self.scaleconv_22 = (nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.scaleconv_23 = (nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2))
        self.scaleconv_24 = (nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3))

        self.contracting_30 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_31 = self.conv_block(in_channels=256, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):
        #print(X.shape)
        scaleconv_1_out =  self.scaleconv_1(X)
        scaleconv_2_out =  self.scaleconv_2(X)
        scaleconv_3_out =  self.scaleconv_3(X)
        scaleconv_4_out =  self.scaleconv_4(X)
        #print(scaleconv_1_out.shape,scaleconv_2_out.shape,scaleconv_3_out.shape)
        temp = torch.stack((scaleconv_1_out,scaleconv_2_out,scaleconv_3_out,scaleconv_4_out),dim=0)
        #print(temp.shape)
        weighted_sum_out = self.weighted_sum(temp.permute(1,2,3,4,0))
        #print(weighted_sum_out.shape)
        weighted_sum_out = weighted_sum_out.squeeze(4)
        #print(weighted_sum_out.shape)

        #print(contracting_01_out.shape)
        skip_out = weighted_sum_out + self.contracting_01(X)
        #print(skip_out.shape)
        contracting_11_out = self.contracting_11(skip_out) # [-1, 64, 256, 256]
        #print(contracting_11_out.shape)
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        #print(contracting_12_out.shape)
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        #print(contracting_21_out.shape)
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        #print(contracting_22_out.shape)

        scaleconv_21_out =  self.scaleconv_21(contracting_22_out)
        scaleconv_22_out =  self.scaleconv_22(contracting_22_out)
        scaleconv_23_out =  self.scaleconv_23(contracting_22_out)
        scaleconv_24_out =  self.scaleconv_24(contracting_22_out)
        #print(scaleconv_1_out.shape,scaleconv_2_out.shape,scaleconv_3_out.shape)
        temp1 = torch.stack((scaleconv_21_out,scaleconv_22_out,scaleconv_23_out,scaleconv_24_out),dim=0)
        #print(temp.shape)
        weighted_sum_out1 = self.weighted_sum(temp1.permute(1,2,3,4,0))
        #print(weighted_sum_out.shape)
        weighted_sum_out1 = weighted_sum_out1.squeeze(4)
        #print(weighted_sum_out.shape)

        #print(contracting_01_out.shape)
        skip_out1 = weighted_sum_out1 + self.contracting_30(contracting_22_out)

        contracting_31_out = self.contracting_31(skip_out1) # [-1, 256, 64, 64]
        #print(contracting_31_out.shape)
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        #print(contracting_32_out.shape)
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        #print(contracting_41_out.shape)
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        #print(contracting_42_out.shape)
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        #print(middle_out.shape)
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        #print(expansive_11_out.shape)
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        #print(expansive_12_out.shape)
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        #print(expansive_21_out.shape)
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        #print(expansive_22_out.shape)
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        #print(expansive_31_out.shape)
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        #print(expansive_32_out.shape)
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        #print(expansive_41_out.shape)
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]

        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out

model = WasteNet(num_classes=num_classes)


data_loader = DataLoader(dataset, batch_size=4)
#print(len(dataset), len(data_loader))
model = model.to(device)
X, Y = next(iter(data_loader))
X = X.to(device)
Y = Y.to(device)
#print(X.shape, Y.shape)
Y_pred = model(X)
print(X.shape, Y.shape, Y_pred.shape)

#Initial predictions
out = np.asarray(torch.argmax(Y_pred[1],dim=0).unsqueeze(2).repeat(1,1,3).reshape(-1, 3).cpu())
out *=255
label_pred = label_model1.predict(out).reshape(256, 256)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(X[1].squeeze(0).permute(1,2,0).cpu())
axes[1].imshow(Y[1].squeeze(0).cpu())
axes[2].imshow(label_pred)



batch_size = 16

epochs = 15
lr = 0.01
dataset = CleanWasteDataset(train_fns, train_mask,label_model1)
data_loader = DataLoader(dataset, batch_size=batch_size)
#model = UNet(num_classes=num_classes).to(device)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
step_losses = []
epoch_losses = []
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        Y_pred = model(X)

        loss = criterion(Y_pred, Y)
        print(loss.item())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    epoch_losses.append(epoch_loss/len(data_loader))
    print(epoch_loss/len(data_loader))

torch.save(model.state_dict(), '/content/drive/MyDrive/SatelliteImages/M6U-Net.pth')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)


test_batch_size = 10
val_data = CleanWasteDataset(test_list, test_mask,label_model1)
val_loader = DataLoader(val_data, batch_size=test_batch_size)
X, Y = next(iter(val_loader))
X, Y = X.to(device), Y.to(device)

Y_pred = model(X)
print(Y_pred.shape)
for i in range(test_batch_size):
  out = np.asarray(torch.argmax(Y_pred[i],dim=0).unsqueeze(2).repeat(1,1,3).reshape(-1, 3).cpu())
  out *=255
  label_pred = label_model1.predict(out).reshape(256, 256)
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  axes[0].set_title("Input image")
  axes[0].imshow(X[i].squeeze(0).permute(1,2,0).cpu())
  axes[1].set_title("Actual")
  axes[1].imshow(Y[i].squeeze(0).cpu())
  axes[2].set_title("Predicted")
  axes[2].imshow(label_pred)
