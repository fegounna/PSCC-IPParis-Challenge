import os
from glob import glob

def get_train_files():
    train_files =[]

    vol_path = "/tsi/data_education/data_challenge/train/volume"
    seg_path = "/tsi/data_education/data_challenge/train/seg"
    for img,label in zip(os.listdir(vol_path), os.listdir(seg_path)):
        image_vol_path = os.path.join(vol_path,img)
        image_seg_path = os.path.join(seg_path, label)
        train_files.append({"image":image_vol_path,"label":image_seg_path})
    return train_files

def get_test_files():
    test_files =[]

    vol_path = "/tsi/data_education/data_challenge/test/volume"
    for img in os.listdir(vol_path):
        image_vol_path = os.path.join(vol_path,img)
        
        image_name = os.path.basename(image_vol_path)
        image_id = image_name.split('.')[0]

        test_files.append({"image":image_vol_path,"id":image_id[:-7]})
    return test_files