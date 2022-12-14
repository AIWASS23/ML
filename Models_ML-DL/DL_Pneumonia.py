# download de dataset https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download

from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np

x  = 'Path'
path = Path(x)
path.ls()

np.random.seed(40)
data = ImageDataBunch.from_folder(
    path, train = '.', 
    valid_pct = 0.2,
    ds_tfms = get_transforms(),
    size=224,
    num_workers = 4).normalize(imagenet_stats)

data.show_batch(rows = 3, figsize = (7,6),recompute_scale_factor = True)

learn = cnn_learner(
    data,
    models.resnet50,
    metrics=[accuracy],
    model_dir = Path('Path'),
    path = Path("."))

learn.lr_find()
learn.recorder.plot(suggestions = True)

lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(4,slice(lr1,lr2))

# lr1 = 1e-3
lr = 1e-1
learn.fit_one_cycle(20,slice(lr))

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,slice(1e-4,1e-3))

learn.recorder.plot_losses()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
img = open_image('IM-0001-0001.jpeg')
print(learn.predict(img)[0])