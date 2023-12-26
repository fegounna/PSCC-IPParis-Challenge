from monai.networks.nets import UNet

MODELS = {
    "3DUnet" : UNet(spatial_dims=3,in_channels=1,out_channels=1,channels=(16, 32, 64, 128, 256) ,strides=(2, 2, 2, 2) )
}