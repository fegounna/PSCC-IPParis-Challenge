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


MODEL = os.environ.get("MODEL")
fold = int(os.environ.get("fold"))
if __name__ == "__main__":

    model = dispatcher.MODELS[MODEL]
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint = torch.load(f"/home/ssd/ext-6401/PSCC-IPParis-Challenge/models/checkpoint_{MODEL}_{fold}.pht.tar", map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    num_batches = 1
    test_files = dataset.get_test_files()
    test_ds = monai.data.Dataset(data=test_files,transform=prepocessing.test_transforms)
    test_ldr = DataLoader(test_ds,batch_size=num_batches)
    tk0 = tqdm(test_ldr, total=num_batches)

    model.eval()


    save_directory = f'/home/ssd/ext-6401/PSCC-IPParis-Challenge/predictions/{MODEL}_{fold}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with torch.no_grad():
        for data in tk0:
            inputs = data["image"]
            inputs = inputs.to(device, dtype=torch.float)
            output = model(inputs)

            for prediction,id in zip(output,data["id"]):
                prediction = prediction[0]
                output_np = prediction.cpu().detach().numpy()
                threshold_value = 0.5
                thresholded_output = output_np > threshold_value
                thresholded_output = thresholded_output.astype(np.uint8)
                voxel_size = [0.9765625, 0.9765625, 3.0]
                affine = np.diag(voxel_size + [1])
                nifti_img = nib.Nifti1Image(thresholded_output, affine)
                path = os.path.join(save_directory, f"""{id}.nii.gz""")
                nifti_img.to_filename(path)
        
        tk0.close()
