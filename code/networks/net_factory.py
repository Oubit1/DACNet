from networks.VNet import VNet,ACNet3d_V2
from networks.unet import UNet,ACNet2d_v3，ACNet2d_v2
def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "UAMT" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "acnet2d_v3":
        net = ACNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "acnet2d_v2":
        net = ACNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "acnet3d_v2" and mode == "train":
        net = ACNet3d_V2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "acnet3d_v2" and mode == "test":
        net = ACNet3d_V2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
