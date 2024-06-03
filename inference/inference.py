#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils import paths
import os
import cv2
from tqdm import tqdm
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,help="path with the files")

args = vars(ap.parse_args())


p = args["path"]
m = "config-patchcore-resnet50.yaml"

shutil.rmtree(p+'/temp',ignore_errors=True)
os.makedirs(p+'/temp',exist_ok=True)

def rreplace(s, old, new):
    return (s[::-1].replace(old[::-1],new[::-1], 1))[::-1]


print("[Splitting the images]")

images = list(paths.list_images(p))

for imPath in tqdm(images):
    image = cv2.imread(imPath)
    height, width, _ = image.shape

    # Calculate the dimensions for the four parts
    half_width = width // 2
    half_height = height // 2

    # Split the image into 4 parts
    top_left = image[0:half_height, 0:half_width]
    top_right = image[0:half_height, half_width:width]
    bottom_left = image[half_height:height, 0:half_width]
    bottom_right = image[half_height:height, half_width:width]
    
    cv2.imwrite(rreplace(imPath[:-4]+"_1.tif",'/','/temp/'),top_left)
    cv2.imwrite(rreplace(imPath[:-4]+"_2.tif",'/','/temp/'),top_right)
    cv2.imwrite(rreplace(imPath[:-4]+"_3.tif",'/','/temp/'),bottom_left)
    cv2.imwrite(rreplace(imPath[:-4]+"_4.tif",'/','/temp/'),bottom_right)
    # os.remove(imPath)
    
images = list(paths.list_images(p+'/temp/'))

for imPath in tqdm(images):
    image = cv2.imread(imPath)
    height, width, _ = image.shape

    # Calculate the dimensions for the four parts
    half_width = width // 2
    half_height = height // 2

    # Split the image into 4 parts
    top_left = image[0:half_height, 0:half_width]
    top_right = image[0:half_height, half_width:width]
    bottom_left = image[half_height:height, 0:half_width]
    bottom_right = image[half_height:height, half_width:width]
    
    cv2.imwrite(imPath[:-4]+"_1.tif",top_left)
    cv2.imwrite(imPath[:-4]+"_2.tif",top_right)
    cv2.imwrite(imPath[:-4]+"_3.tif",bottom_left)
    cv2.imwrite(imPath[:-4]+"_4.tif",bottom_right)
    os.remove(imPath)



# In[2]:


from anomalib.data import Folder
from anomalib.deploy import OpenVINOInferencer


# In[3]:




# In[7]:


from anomalib.config import get_configurable_parameters
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from pytorch_lightning import Trainer

config = get_configurable_parameters(config_path=m)
# config.optimization.export_mode = "openvino"


# In[8]:


from anomalib.models import Patchcore
from anomalib.models import get_model
import torch


# Create the model and engine
model = torch.load(m.replace('.yaml','.pth'))
model.eval()
trainer = Trainer(**config.trainer)


# In[ ]:


from anomalib.data.utils import InputNormalizationMethod, get_transforms
from torch.utils.data import DataLoader
from anomalib.data.inference import InferenceDataset


# In[ ]:



imagefolder = p+'/temp/' 

normalization = InputNormalizationMethod('imagenet')

image_size = 128

transform = get_transforms(
    config=None, image_size=image_size, center_crop=None, normalization=normalization
)

dataset = InferenceDataset(
        imagefolder, image_size=image_size, transform=transform  # type: ignore
    )
dataloader = DataLoader(dataset)

preds = trainer.predict(model=model, dataloaders=[dataloader])


# In[ ]:



predScore = [x['pred_scores'] for x in preds]
predLabels = [x['pred_labels'] for x in preds]


# In[ ]:


from anomalib.data.utils import get_image_filenames
names = get_image_filenames(p+'/temp/')




# In[ ]:


import pandas as pd
df = pd.DataFrame(list(zip(names,predScore,predLabels)),columns=['name','score','label'])


# In[ ]:


df['parent'] = df['name'].apply(lambda x: str(x)[:-10])

df.to_csv(p+'/result.csv',index=None)



images = list(paths.list_images(p+'/temp/'))

for imPath in tqdm(images):
    image = cv2.imread(imPath)
    os.remove(imPath)
    
os.rmdir(p+'/temp/')