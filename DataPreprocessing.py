!git clone https://github.com/UAVVaste/UAVVaste.git

!python3 'UAVVaste/main.py'
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import PIL.ImageOps
import numpy as np
import os
import torchvision
from torchvision import transforms
import json

with open("/content/drive/MyDrive/SatelliteImages/Data/annotations.json") as json_data:
    data = json.load(json_data)
files = []
for i in range(len(data['images'])):
   files.append( data['images'][i]['file_name'])

small = 0
medium = 0
large = 0

j=0
k=0
for k in range(len(files)):
    image = Image.open("/content/drive/MyDrive/SatelliteImages/Data/UAV/images/"+files[k]).convert('RGB')
    draw = ImageDraw.Draw(image)
    i=j
    square = np.zeros(image.size)
    while i>=0:
            if data['annotations'][i]['image_id']==k:
                x,y,w,h = data['annotations'][i]['bbox']
                draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                if (abs(x-y)*abs(w-h)<3200):
                  #Small
                  small+=1
                  square[x:x+w, y:y+h]=253
                elif (abs(x-y)*abs(w-h)<9600):
                  #Medium
                  medium+=1
                  square[x:x+w, y:y+h]=255
                else:
                  #Large
                  large+=1
                  square[x:x+w, y:y+h]=254
                i+=1
            else:
                j=i
                i=-1

    square_img = Image.fromarray(square.transpose())
    square_img=square_img.convert("L")
    inverted_image = PIL.ImageOps.invert(square_img)

    if(os.path.exists('/content/drive/MyDrive/SatelliteImages/Data/UAV/Mask/'+files[k])):
        os.remove('/content/drive/MyDrive/SatelliteImages/Data/UAV/Mask/'+files[k])
    inverted_image.save('/content/drive/MyDrive/SatelliteImages/Data/UAV/Mask/'+files[k],bbox_inches='tight', transparent=True)
    print("Done ",files[k])
