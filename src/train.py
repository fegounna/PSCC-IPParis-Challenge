import os
import torch
import monai
from monai.data import Dataset, DataLoader
from torch.utils.data import Subset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn.model_selection import KFold

from monai.losses import DiceLoss

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import copy 

from . import engine
from . import preprocessing
from . import dataset
from . import dispatcher
from . import metrics

fold = int(os.environ.get('fold'))
MODEL = os.environ.get("MODEL")


device = "cuda"
k_folds = 3
epochs = 500

if __name__ == "__main__":

    train_files = dataset.get_train_files()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state = 42)
    train_ids, val_ids = list(kf.split(train_files))[fold-1]
    print(f"Fold {fold}")
    
    model = dispatcher.MODELS[MODEL]
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True) ## could be  a combination of Dice and cross entropy loss (like sum Ldice + LCE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 3 e-4 in nn U-net paper
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
    
    train_dataset = monai.data.Dataset(data=[train_files[i] for i in train_ids],transform=preprocessing.train_transforms)
    val_dataset = monai.data.Dataset(data=[train_files[i] for i in val_ids],transform=preprocessing.orig_transforms)
    train_loader = DataLoader(train_dataset,batch_size=1)
    val_loader = DataLoader(val_dataset,batch_size=1)

    best_loss = None
    best_model  = None
    best_optimizer = None
    patience = 2
    min_delta = 0 #how big of a change an improvement in performance do we need for it to count
    counter = 0

    
    for epoch in range(epochs):
        
        print(f"Training Epoch: {epoch+1} / {epochs}, Fold: {fold}")
        engine.train(train_dataset, train_loader, model, criterion, optimizer)
        print(f"Validation Epoch: {epoch+1} / {epochs}, Fold: {fold}")
        val_loss = engine.evaluate(val_dataset, val_loader, model)
        print(f"Epoch={epoch+1}, Fold={fold}, Loss={val_loss}")

        if best_loss == None or best_loss >= val_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict()) #need also to save the optimizer!
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"No improvement found in the last{counter} epochs")
            if counter >= patience:
                print(f"Early stopping triggered after {counter} epochs.")
                checkpoint = {"state_dict":best_model,"optimizer":best_optimizer}
                torch.save(checkpoint,f"models/checkpoint_{MODEL}_{fold}.pht.tar")
                print(f"End of fold{fold} with scores {metrics.metric_score(val_dataset, val_loader, model)}")
                break
        scheduler.step(val_loss)
    #calculate the metrics


#in nnunet : . The training was terminated automatically if lvMA did not improve by more than 5 × 10−3 within the last 60 epochs, but not before the learning rate was smaller than 10−6
