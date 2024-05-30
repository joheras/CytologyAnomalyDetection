#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
from fastai.vision.gan import *
torch.cuda.set_device(2)

# The generator

# In[2]:


path = Path('../segundaTanda/crops/dataset/')
pathcrops = Path('../segundaTanda/crops/')


# In[20]:


def get_dls(bs:int, size:int):
  "Generates two `GAN` DataLoaders"
  dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=partial(get_image_files, folders = ['train', 'valid']),
                   get_y = lambda x: x.parent/x.name,
                   splitter=GrandparentSplitter(),
                   item_tfms=Resize(size),
                   batch_tfms=[*aug_transforms(max_zoom=2.),
                               Normalize.from_stats(*imagenet_stats)])
  dls = dblock.dataloaders(path, bs=bs, path=path)
  dls.c = 3 # For 3 channel image
  return dls


# In[21]:


dls_gen = get_dls(128, 64)


# In[4]:


wd, y_range, loss_gen = 1e-3, (-3., 3.), MSELossFlat()


# In[18]:


bbone = resnet34

def create_gen_learner():
  return unet_learner(dls_gen, bbone, loss_func=loss_gen,blur=True, norm_type=NormType.Weight, self_attention=True,
                  y_range=y_range)


# In[29]:


learn_gen = create_gen_learner()


# In[8]:


learn_gen.fit_one_cycle(2, pct_start=0.8, wd=wd)


# In[9]:


learn_gen.unfreeze()


# In[10]:


learn_gen.fit_one_cycle(3, slice(1e-6,1e-3), wd=wd)


# In[11]:


learn_gen.save('gen-model0-resnet34')


# In[5]:


name_gen = 'gen'
name_gen1 = 'image_gen_2'
path_gen = pathcrops/name_gen/name_gen1
path_gen.mkdir(exist_ok=True)


# In[13]:


def save_preds(dl, learn):
  "Save away predictions"
  names = dl.dataset.items
  
  preds,_ = learn.get_preds(dl=dl)
  for i,pred in enumerate(preds):
      dec = dl.after_batch.decode((TensorImage(pred[None]),))[0][0]
      arr = dec.numpy().transpose(1,2,0).astype(np.uint8)
      Image.fromarray(arr).save(path_gen/names[i].name)


# In[14]:


dl = dls_gen.train.new(shuffle=False, drop_last=False, 
                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])


# In[15]:


save_preds(dl, learn_gen)


# The critic

# In[6]:


path_g = get_image_files(pathcrops/name_gen/name_gen1)
path_i = get_image_files(path/'train')
fnames = path_g + path_i


# In[7]:


def grand_parent_label(o):
    "Label `item` with the parent folder name."
    return Path(o).parent.parent.name


# In[8]:


def get_crit_dls(fnames, bs:int, size:int):
  "Generate two `Critic` DataLoaders"
  splits = RandomSplitter(0.1)(fnames)
  dsrc = Datasets(fnames, tfms=[[PILImage.create], [grand_parent_label, Categorize]],
                 splits=splits)
  tfms = [ToTensor(), Resize(size)]
  gpu_tfms = [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
  return dsrc.dataloaders(bs=bs, after_item=tfms, after_batch=gpu_tfms)


# In[9]:


dls_crit = get_crit_dls(fnames, bs=128, size=64)


# In[10]:


loss_crit = AdaptiveLoss(nn.BCEWithLogitsLoss())


# In[11]:


def create_crit_learner(dls, metrics):
  return Learner(dls, gan_critic(), metrics=metrics, loss_func=loss_crit)


# In[12]:


learn_crit = create_crit_learner(dls_crit, accuracy_thresh_expand)


# In[13]:


learn_crit.fit_one_cycle(6, 1e-3, wd=wd)


# In[14]:


learn_crit.save('critic-model0-resnet34')


# The GAN

# In[15]:


ls_crit = get_crit_dls(fnames, bs=128, size=64)


# In[16]:


learn_crit = create_crit_learner(dls_crit, metrics=None).load('critic-model0')


# In[22]:


learn_gen = create_gen_learner().load('gen-model0-resnet34')


# In[23]:


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


# In[24]:


switcher = AdaptiveGANSwitcher(critic_thresh=.65)


# In[25]:


learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(Adam, mom=0.), cbs=GANDiscriminativeLR(mult_lr=5.))


# In[26]:


lr = 1e-4


# In[27]:


learn.fit(10, lr, wd=wd)


# In[31]:


learn.save('gan-model0-resnet34')


# In[ ]:




