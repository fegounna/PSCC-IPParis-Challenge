import os
from glob import glob

def get_train_files():
    train_files =[]

    for i in range(1,292):
        i=str(i)
        i = (3-len(i))*'0'+i

        vol_path = fr"C:\Users\Moakher\Desktop\copy-of-pscc-data-challenge\train\volume\LUNG1-{i}_vol.nii"  
        vol_contents = os.listdir(vol_path)
        image_vol_path = os.path.join(vol_path, vol_contents[0])

        seg_path = fr"C:\Users\Moakher\Desktop\copy-of-pscc-data-challenge\train\seg\LUNG1-{i}_seg.nii"
        seg_contents = os.listdir(seg_path)
        image_seg_path = os.path.join(seg_path, seg_contents[0])

        train_files.append({"image":image_vol_path,"label":image_seg_path})
    return train_files

def get_test_files():
    test_files =[]

    for i in range(1,101):
        i=str(i)
        i = (3-len(i))*'0'+i

        vol_path = fr"C:\Users\Moakher\Desktop\copy-of-pscc-data-challenge\test\volume\LUNG1-{i}_vol.nii"  
        vol_contents = os.listdir(vol_path)
        image_vol_path = os.path.join(vol_path, vol_contents[0])

        image_name = os.path.basename(image_vol_path)
        image_id = image_name.split('.')[0]

        test_files.append({"image":image_vol_path,"id":image_id[:-7]})
    return test_files