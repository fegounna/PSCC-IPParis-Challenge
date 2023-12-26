import monai
from monai.transforms import Transform

from monai.transforms import ( 
    Compose,
    LoadImaged,
    ToTensord,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ScaleIntensityRangePercentilesd,
    RandFlipd,
    RandAffined,
    RandRotated,
    RandAdjustContrast,
    RandGaussianNoised,
    Orientationd,
    NormalizeIntensityd,
)


class CustomCrop(Transform):
    def __call__(self, data):
        d = dict(data)
        d["image"] = d["image"][80:450, 90:410, :100]
        d["label"] = d["label"][80:450, 90:410, :100]
        return d
class AddChannel(Transform):
    def __call__(self,data):
        d = dict(data)
        d["image"] = d["image"][None]
        d["label"] = d["label"][None]
        return d
class AddChannel_test(Transform):
    def __call__(self,data):
        d = dict(data)
        d["image"] = d["image"][None]
        return d




test_transforms = Compose(
[
    LoadImaged(keys = ["image"]),
    AddChannel_test(),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(0.9765625, 0.9765625, 3.0), mode="bilinear"),
    #ScaleIntensityRangePercentilesd(keys ="image",lower = 0.5, upper = 99.5, b_min=0.,b_max=1., clip=True),
    ##ScaleIntensityRanged(keys ="image",a_min =-1000,a_max=1000,b_min=0.,b_max=1.,clip=True),
    CropForegroundd (keys = ["image"],source_key='image',allow_smaller =True),
    Resized(keys = ["image"],spatial_size = (256,256,128), mode='area'),
    NormalizeIntensityd(keys=['image'], subtrahend=None, divisor=None),
    ToTensord(keys = ["image"]),
]
)

orig_transforms = Compose(
[
    LoadImaged(keys = ["image","label"]),
    AddChannel(),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.9765625, 0.9765625, 3.0), mode=("bilinear", "nearest")),
    
    ##ScaleIntensityRanged(keys ="image",a_min =-422.,a_max=1027.,b_min=0.,b_max=1.,clip=True),
    ##ScaleIntensityRanged(keys ="image",a_min =-1000,a_max=1000,b_min=0.,b_max=1.,clip=True),
    #ScaleIntensityRangePercentilesd(keys ="image",lower = 0.5, upper = 99.5, b_min=0.,b_max=1., clip=True),
    CropForegroundd (keys = ["image","label"],source_key='image',allow_smaller =True),
    
    Resized(keys = ["image","label"],spatial_size = (256,256,128), mode=['area','nearest']),
    ##Znormalisation 
    NormalizeIntensityd(keys=['image'], subtrahend=None, divisor=None),
    ToTensord(keys = ["image","label"]),
]
)


train_transforms = Compose(
[
    LoadImaged(keys = ["image","label"]),
    AddChannel(),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.9765625, 0.9765625, 3.0), mode=("bilinear", "nearest")),
    
    ##ScaleIntensityRanged(keys ="image",a_min =-422.,a_max=1027.,b_min=0.,b_max=1.,clip=True),
    ##ScaleIntensityRanged(keys ="image",a_min =-1000,a_max=1000,b_min=0.,b_max=1.,clip=True),
    #ScaleIntensityRangePercentilesd(keys ="image",lower = 0.5, upper = 99.5, b_min=0.,b_max=1., clip=True),
    CropForegroundd (keys = ["image","label"],source_key='image',allow_smaller =True),
    
    Resized(keys = ["image","label"],spatial_size = (256,256,128), mode=['area','nearest']),
    ##Data Augmentation 
    #RandAffined(keys=['image', 'label'], prob=0.3, translate_range=10), 
    #RandRotated(keys=['image', 'label'], prob=0.3, range_z=5.0),

    #RandAffined(keys=['image', 'label'], prob=0.3, translate_range=10,  # Shift by up to 10 pixels #scale_range=(0.9, 1.1),  # Scale by 0.9 to 1.1, rotate_range=0.174533  # Rotate by up to 45 degrees),
    
    #RandGaussianNoised(keys='image', prob=0.3),  
    ##maybe zoom ?
    
    ##Znormalisation 
    NormalizeIntensityd(keys=['image'], subtrahend=None, divisor=None),
    ToTensord(keys = ["image","label"]),
]
)