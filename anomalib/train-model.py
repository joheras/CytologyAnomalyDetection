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
ap.add_argument("-i", "--iteration", required=True,help="name of the user")
ap.add_argument("-m", "--model", required=True,help="name of the user")

args = vars(ap.parse_args())


i = int(args["iteration"])
m = args["model"]


shutil.rmtree('dataset/train/benignas',ignore_errors=True)
shutil.rmtree('dataset/valid/benignas',ignore_errors=True)
shutil.rmtree('dataset/test/malignas',ignore_errors=True)
shutil.rmtree('dataset/test/benignas',ignore_errors=True)


os.makedirs('dataset/train/benignas',exist_ok=True)
os.makedirs('dataset/valid/benignas',exist_ok=True)
os.makedirs('dataset/test/malignas',exist_ok=True)
os.makedirs('dataset/test/benignas',exist_ok=True)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--iteration", required=True,help="name of the user")
# args = vars(ap.parse_args())


# i = int(args["iteration"])

with open('../datasets/training-'+str(i)+'.txt') as f:
    lines = f.readlines()
    
namesTrainBenignas = [line.replace('\n','') for line in lines]

with open('../datasets/valid-'+str(i)+'.txt') as f:
    lines = f.readlines()
namesValidBenignas = [line.replace('\n','') for line in lines]

with open('../datasets/valid-'+str(i)+'.txt') as f:
    lines = f.readlines()
namesTestBenignas = [line.replace('\n','') for line in lines]



import cv2
from tqdm import tqdm 

for imPath in tqdm(namesTrainBenignas):
    try:
        image = cv2.imread('../../segundaTanda/S3/Benignas/' + imPath)
        height, width, _ = image.shape

        # Calculate the dimensions for the four parts
        half_width = width // 2
        half_height = height // 2

        # Split the image into 4 parts
        top_left = image[0:half_height, 0:half_width]
        top_right = image[0:half_height, half_width:width]
        bottom_left = image[half_height:height, 0:half_width]
        bottom_right = image[half_height:height, half_width:width]

        cv2.imwrite('dataset/train/benignas/'+imPath[:-4]+"_1.tif",top_left)
        cv2.imwrite('dataset/train/benignas/'+imPath[:-4]+"_2.tif",top_right)
        cv2.imwrite('dataset/train/benignas/'+imPath[:-4]+"_3.tif",bottom_left)
        cv2.imwrite('dataset/train/benignas/'+imPath[:-4]+"_4.tif",bottom_right)
    except:
        print(imPath)

        
for imPath in tqdm(namesValidBenignas):
    try:
        image = cv2.imread('../../segundaTanda/S3/Benignas/' + imPath)
        height, width, _ = image.shape

        # Calculate the dimensions for the four parts
        half_width = width // 2
        half_height = height // 2

        # Split the image into 4 parts
        top_left = image[0:half_height, 0:half_width]
        top_right = image[0:half_height, half_width:width]
        bottom_left = image[half_height:height, 0:half_width]
        bottom_right = image[half_height:height, half_width:width]

        cv2.imwrite('dataset/valid/benignas/'+imPath[:-4]+"_1.tif",top_left)
        cv2.imwrite('dataset/valid/benignas/'+imPath[:-4]+"_2.tif",top_right)
        cv2.imwrite('dataset/valid/benignas/'+imPath[:-4]+"_3.tif",bottom_left)
        cv2.imwrite('dataset/valid/benignas/'+imPath[:-4]+"_4.tif",bottom_right)
    except:
        print(imPath)
        

for imPath in tqdm(namesTrainBenignas[0:5]):
    try:
        image = cv2.imread('../../segundaTanda/S3/Benignas/' + imPath)
        height, width, _ = image.shape

        # Calculate the dimensions for the four parts
        half_width = width // 2
        half_height = height // 2

        # Split the image into 4 parts
        top_left = image[0:half_height, 0:half_width]
        top_right = image[0:half_height, half_width:width]
        bottom_left = image[half_height:height, 0:half_width]
        bottom_right = image[half_height:height, half_width:width]

        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_1.tif",top_left)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_2.tif",top_right)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_3.tif",bottom_left)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_4.tif",bottom_right)
    except:
        print(imPath)
        
for imPath in tqdm(namesTestBenignas):
    try:
        image = cv2.imread('../../segundaTanda/S3/Benignas/' + imPath)
        height, width, _ = image.shape

        # Calculate the dimensions for the four parts
        half_width = width // 2
        half_height = height // 2

        # Split the image into 4 parts
        top_left = image[0:half_height, 0:half_width]
        top_right = image[0:half_height, half_width:width]
        bottom_left = image[half_height:height, 0:half_width]
        bottom_right = image[half_height:height, half_width:width]

        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_1.tif",top_left)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_2.tif",top_right)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_3.tif",bottom_left)
        cv2.imwrite('dataset/test/benignas/'+imPath[:-4]+"_4.tif",bottom_right)
    except:
        print(imPath)

for imPath in tqdm(paths.list_images('../../segundaTanda/S3/Malignas/')):
    imPath = imPath.split('/')[-1]
    image = cv2.imread('../../segundaTanda/S3/Malignas/' + imPath)
    height, width, _ = image.shape

    # Calculate the dimensions for the four parts
    half_width = width // 2
    half_height = height // 2

    # Split the image into 4 parts
    top_left = image[0:half_height, 0:half_width]
    top_right = image[0:half_height, half_width:width]
    bottom_left = image[half_height:height, 0:half_width]
    bottom_right = image[half_height:height, half_width:width]
    
    cv2.imwrite('dataset/test/malignas/'+imPath[:-4]+"_1.tif",top_left)
    cv2.imwrite('dataset/test/malignas/'+imPath[:-4]+"_2.tif",top_right)
    cv2.imwrite('dataset/test/malignas/'+imPath[:-4]+"_3.tif",bottom_left)
    cv2.imwrite('dataset/test/malignas/'+imPath[:-4]+"_4.tif",bottom_right)
    
    
images = list(paths.list_images('dataset/'))

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
    
images = list(paths.list_images('dataset/'))

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


datamodule = Folder(
    # name="citologia",
    root="dataset/",
    normal_dir="valid/benignas",
    normal_test_dir = "test/benignas",
    abnormal_dir="test/malignas",
    task="classification",
    image_size=128,
    train_batch_size=64,
    eval_batch_size=64,
)

# Setup the datamodule
datamodule.setup()


# In[4]:


i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image'])


# In[5]:


train_data['image_path'][0]


# In[7]:


from anomalib.config import get_configurable_parameters
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from pytorch_lightning import Trainer

config = get_configurable_parameters(config_path=m)
# config.optimization.export_mode = "openvino"


# In[8]:


from anomalib.models import Patchcore
from anomalib.models import get_model

# Create the model and engine
model = get_model(config)
callbacks = get_callbacks(config)
trainer = Trainer(**config.trainer, callbacks=callbacks)
trainer.fit(model=model, datamodule=datamodule)


# In[ ]:


from anomalib.data.utils import InputNormalizationMethod, get_transforms
from torch.utils.data import DataLoader
from anomalib.data.inference import InferenceDataset


# In[ ]:


datasetbenigntest = 'dataset/test/benignas/' 

normalization = InputNormalizationMethod('imagenet')

image_size = 128

transform = get_transforms(
    config=None, image_size=image_size, center_crop=None, normalization=normalization
)

dataset = InferenceDataset(
        datasetbenigntest, image_size=image_size, transform=transform  # type: ignore
    )
dataloader = DataLoader(dataset)

predsbenign = trainer.predict(model=model, dataloaders=[dataloader])


# In[ ]:


imagefolder = 'dataset/test/malignas/' 

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


predsMalignas = [x['pred_labels'] for x in preds]
predsBenign = [x['pred_labels'] for x in predsbenign]


# In[ ]:


from anomalib.data.utils import get_image_filenames
benignnames = get_image_filenames('dataset/test/benignas/')


# In[ ]:


malignnames = get_image_filenames('dataset/test/malignas/')


# In[ ]:


import pandas as pd
df = pd.DataFrame(list(zip(benignnames+malignnames,predsBenign+predsMalignas)),columns=['name','error'])


# In[ ]:


df['parent'] = df['name'].apply(lambda x: str(x)[:-10])


# In[ ]:


df2 = df[df.error==True]


# In[ ]:


tn = 0
fn = 0
fp = 0
tp = 0

for x in df.groupby('parent').size().keys():
    if('benignas' in x):
        tn+=1
    else:
        fn+=1

for x in  df2.groupby('parent').size().keys():
    if('benignas' in x):
        tn-=1
        fp+=1
    else:
        tp+=1
        fn-=1
        
print(tp,fn,fp,tn)
accuracy = (tp+tn)/(tp+fn+fp+tn)
precision = (tp)/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * precision*recall/(precision+recall)




# In[ ]:


predsMalignas = [x['pred_scores'] for x in preds]
predsBenign = [x['pred_scores'] for x in predsbenign]

from sklearn.metrics import RocCurveDisplay, roc_curve,auc
fprs = [0]
tprs = [0]

for t in range(100,0,-5): 
    thresh = t/100

    df = pd.DataFrame(list(zip(benignnames+malignnames,predsBenign+predsMalignas)),columns=['name','error'])
    df['parent'] = df['name'].apply(lambda x: str(x)[:-10])
    df2 = df[df.error>thresh]
    tn = 0
    fn = 0
    fp = 0
    tp = 0

    for x in df.groupby('parent').size().keys():
        if('benignas' in x):
            tn+=1
        else:
            fn+=1
    
    for x in  df2.groupby('parent').size().keys():
        if('benignas' in x):
            tn-=1
            fp+=1
        else:
            tp+=1
            fn-=1
    
    
    
    # print(tp,fn,tn,fp)
    tprs.append(tp/(tp+fn))
    fprs.append((fp/(fp+tn)))

with open(m.replace('.yaml','.txt'),mode="a") as f:
    f.write(str((accuracy,precision,recall,f1_score,auc(fprs,tprs)))+"\n")


# In[ ]:


# fpr,tpr,_=roc_curve([0 for x in benignnames]+[1  for x in malignnames],predsBenign+predsMalignas)
# auc(fpr,tpr)


# In[ ]:




