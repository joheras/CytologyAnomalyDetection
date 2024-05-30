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
args = vars(ap.parse_args())


i = int(args["iteration"])
# i = 0


shutil.rmtree('dataset/train/benignas',ignore_errors=True)
shutil.rmtree('dataset/valid/benignas',ignore_errors=True)
shutil.rmtree('dataset/test/malignas',ignore_errors=True)
shutil.rmtree('dataset/test/benignas',ignore_errors=True)
shutil.rmtree('dataset/gen',ignore_errors=True)

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
        image = cv2.imread('../dataset/Benignas/' + imPath)
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
        image = cv2.imread('../dataset/Benignas/' + imPath)
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
        

for imPath in tqdm(namesTestBenignas):
    try:
        image = cv2.imread('../dataset/Benignas/' + imPath)
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

for imPath in tqdm(paths.list_images('../dataset/Malignas/')):
    imPath = imPath.split('/')[-1]
    image = cv2.imread('../dataset/Malignas/' + imPath)
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


from fastai.vision.all import *
from fastai.vision.gan import *


# The generator

# In[3]:


path = Path('dataset')
pathcrops = Path('dataset/')


# In[4]:


import torch
torch.cuda.set_device(0)


# In[5]:


def get_dls(bs:int, size:int):
  "Generates two `GAN` DataLoaders"
  dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=partial(get_image_files, folders = ['train', 'valid']),
                   get_y = lambda x: x.parent/x.name,
                   splitter=GrandparentSplitter(),
                   item_tfms=Resize(size),
                   batch_tfms=[*aug_transforms(max_zoom=2.),
                               Normalize.from_stats(*imagenet_stats)])
  dls = dblock.dataloaders(path, bs=bs, path=path,num_workers=0)
  dls.c = 3 # For 3 channel image
  return dls


# In[6]:


dls_gen = get_dls(8, 64)


# In[7]:


wd, y_range, loss_gen = 1e-3, (-3., 3.), MSELossFlat()


# In[8]:


bbone = wide_resnet50_2

def create_gen_learner():
  return unet_learner(dls_gen, bbone, loss_func=loss_gen,blur=True, norm_type=NormType.Weight, self_attention=True,
                  y_range=y_range)


# In[9]:


learn_gen = create_gen_learner()


# In[10]:


learn_gen.fit_one_cycle(2, pct_start=0.8, wd=wd)


# In[11]:


learn_gen.unfreeze()


# In[12]:


learn_gen.fit_one_cycle(3, slice(1e-6,1e-3), wd=wd)


# In[13]:


learn_gen.save('gen-model-'+str(i)+'-wide-resnet50')


# In[14]:


name_gen = 'gen'
name_gen1 = 'image_gen_wide_resnet50'
path_gen = pathcrops/name_gen/name_gen1
shutil.rmtree(str(path_gen),ignore_errors=True)
(pathcrops/name_gen).mkdir(exist_ok=True)
path_gen.mkdir(exist_ok=True)


# In[15]:


def save_preds(dl, learn):
  "Save away predictions"
  names = dl.dataset.items
  
  preds,_ = learn.get_preds(dl=dl)
  for i,pred in enumerate(preds):
      dec = dl.after_batch.decode((TensorImage(pred[None]),))[0][0]
      arr = dec.numpy().transpose(1,2,0).astype(np.uint8)
      Image.fromarray(arr).save(path_gen/names[i].name)


# In[16]:


dl = dls_gen.train.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])


# In[17]:


save_preds(dl, learn_gen)


# The critic

# In[18]:


path_g = get_image_files(pathcrops/name_gen)
path_i = get_image_files(path/'train')
fnames = path_g + path_i


# In[19]:


def grand_parent_label(o):
    "Label `item` with the parent folder name."
    return Path(o).parent.parent.name


# In[20]:


def get_crit_dls(fnames, bs:int, size:int):
  "Generate two `Critic` DataLoaders"
  splits = RandomSplitter(0.1)(fnames)
  dsrc = Datasets(fnames, tfms=[[PILImage.create], [grand_parent_label, Categorize]],
                 splits=splits)
  tfms = [ToTensor(), Resize(size)]
  gpu_tfms = [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
  return dsrc.dataloaders(bs=bs, after_item=tfms, after_batch=gpu_tfms,num_workers=0)


# In[21]:


dls_crit = get_crit_dls(fnames, bs=8, size=64)


# In[22]:


loss_crit = AdaptiveLoss(nn.BCEWithLogitsLoss())


# In[23]:


def create_crit_learner(dls, metrics):
  return Learner(dls, gan_critic(), metrics=metrics, loss_func=loss_crit)


# In[24]:


learn_crit = create_crit_learner(dls_crit, accuracy_thresh_expand)


# In[25]:


learn_crit.fit_one_cycle(6, 1e-3, wd=wd)


# In[26]:


learn_crit.save('critic-model-'+str(i)+'-wide-resnet50')


# The GAN

# In[27]:


ls_crit = get_crit_dls(fnames, bs=8, size=64)


# In[28]:


learn_crit = create_crit_learner(dls_crit, metrics=None).load('critic-model-'+str(i)+'-wide-resnet50').to_fp16()


# In[29]:


learn_gen = create_gen_learner().load('gen-model-'+str(i)+'-wide-resnet50').to_fp16()


# In[30]:


class GANDiscriminativeLR(Callback):
    "`Callback` that handles multiplying the learning rate by `mult_lr` for the critic."
    def __init__(self, mult_lr=5.): self.mult_lr = mult_lr

    def begin_batch(self):
        "Multiply the current lr if necessary."
        if not self.learn.gan_trainer.gen_mode and self.training: 
            self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr']*self.mult_lr)

    def after_batch(self):
        "Put the LR back to its value if necessary."
        if not self.learn.gan_trainer.gen_mode: self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr']/self.mult_lr)


# In[31]:


switcher = AdaptiveGANSwitcher(critic_thresh=.65)


# In[32]:


learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(Adam, mom=0.), cbs=GANDiscriminativeLR(mult_lr=5.))


# In[33]:


learn.to_fp16()


# In[34]:


lr = 1e-4


# In[35]:


learn.fit(10, lr, wd=wd)


# In[36]:


learn.save('gan-model-'+str(i)+'-wide-resnet50')


# In[37]:


dlTrain = dls_gen.train.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
dlValid = dls_gen.valid.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])


# In[38]:


# Computing error for training dataset
errors = []
    
preds,real = learn.get_preds(dl=dlValid)

for (image, recon) in zip(real, preds):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean(np.array((image - recon) ** 2))
    errors.append(mse)


# In[ ]:





# In[39]:


dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=partial(get_image_files, folders = ['train', 'test']),
                   get_y = lambda x: x.parent/x.name,
                   splitter=GrandparentSplitter(valid_name='test'),
                   item_tfms=Resize(64),
                   batch_tfms=[*aug_transforms(max_zoom=2.),
                               Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path, bs=128, path=path,num_workers=0)
dls.c = 3 # For 3 channel image


# In[40]:


dlTrain = dls.train.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
dlTest = dls.valid.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])


# In[41]:


# Computing error for test dataset
errorsTest = []
    
preds,real = learn.get_preds(dl=dlTest)

for (image, recon) in zip(real, preds):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean(np.array((image - recon) ** 2))
    errorsTest.append(mse)


# In[44]:


from sklearn.metrics import RocCurveDisplay, roc_curve,auc
fprs = [0]
tprs = [0]

for t in range(100,50,-1): 
    thresh = np.quantile(errors, t/100)

    idxs = np.where(np.array(errorsTest) >= thresh)[0]

    images = dlTest.dataset.items

    res = []
    for idx in idxs:
        res.append([images[idx],errorsTest[idx]])
        
    df = pd.DataFrame(res,columns=['name','error'])
    df['parent'] = df['name'].apply(lambda x: str(x)[:-10])
    df2 = pd.DataFrame(dlTest.dataset.items,columns=['name'])
    df2['parent'] = df2['name'].apply(lambda x: str(x)[:-10])
    
    tn = 0
    fn = 0
    fp = 0
    tp = 0

    for x in df2.groupby('parent').size().keys():
        if('benignas' in x):
            tn+=1
        else:
            fn+=1
    
    for x in  df.groupby('parent').size().keys():
        if('benignas' in x):
            tn-=1
            fp+=1
        else:
            tp+=1
            fn-=1
    
    
    
    # print(tp,fn,tn,fp)
    tprs.append(tp/(tp+fn))
    fprs.append((fp/(fp+tn)))

fprs.append(1)
tprs.append(1)
print("AUC: " +str(auc(fprs,tprs)))
roc_display = RocCurveDisplay(fpr=fprs, tpr=tprs).plot()


# In[45]:


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]


# In[50]:


thresh = np.quantile(errors, cutoff_youdens_j(np.array(fprs),np.array(tprs),range(100,70,-1))/100)
idxs = np.where(np.array(errorsTest) >= thresh/100)[0]

images = dlTest.dataset.items

res = []
for idx in idxs:
    res.append([images[idx],errorsTest[idx]])

df = pd.DataFrame(res,columns=['name','error'])
df['parent'] = df['name'].apply(lambda x: str(x)[:-10])
df2 = pd.DataFrame(dlTest.dataset.items,columns=['name'])
df2['parent'] = df2['name'].apply(lambda x: str(x)[:-10])

tn = 0
fn = 0
fp = 0
tp = 0

for x in df2.groupby('parent').size().keys():
    if('benignas' in x):
        tn+=1
    else:
        fn+=1

for x in  df.groupby('parent').size().keys():
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
kappa = 2 * (tp * tn - fn * fp) / (tp * fn + tp * fp + 2 * tp * tn + fn**2 + fn * tn + fp**2 + fp * tn)

with open('wide-resnet50.txt',mode="a") as f:
    f.write("Iteration " + str(i)+"\n")
    f.write(str((accuracy,precision,recall,f1_score,kappa,auc(fprs,tprs)))+"\n")


# In[ ]:




