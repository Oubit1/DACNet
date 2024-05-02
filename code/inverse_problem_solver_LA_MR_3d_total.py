from pathlib import Path
from models import utils as mutils
import sampling
import h5py
import os
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      LangevinCorrectorCS)
from models import ncsnpp
from itertools import islice
from losses import get_optimizer
import datasets
import time
import controllable_generation_TV
from utils import restore_checkpoint, fft2, ifft2, show_samples_gray, get_mask, clear
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
from scipy.io import savemat, loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
import torch.nn.functional as F

###############################################
# Configurations
###############################################
problem = 'Fourier_CS_3d_admm_tv'
config_name = 'fastmri_knee_320_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 95
N = num_scales
root = './data/MRI/BRATS'
vol = 'Brats18_CBICA_AAM_1'
root1 = './data/LA'
vol1 = '2018LA_Seg_Training Set'

if sde.lower() == 'vesde':
  #从configs.ve导入fastmri_knee_320_ncsnpp_continuous作为配置 
  # from configs.ve import fastmri_knee_320_ncsnpp_continuous as configs
  configs = importlib.import_module(f"configs.ve.{config_name}")
  if config_name == 'fastmri_knee_320_ncsnpp_continuous':
    ckpt_filename = f"./exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
  elif config_name == 'ffhq_256_ncsnpp_continuous':
    ckpt_filename = f"exp/ve/{config_name}/checkpoint_48.pth"
  config = configs.get_config()
  config.model.num_scales = num_scales
  #这段代码主要是为了创建一个 SDE（Stochastic Differential Equation）对象，
  # 并设置相关的参数。首先，根据配置文件中的设置，创建了一个 VESDE对象，
  # 使用了一些配置中的参数值。
  # sigma_min 和 sigma_max 是配置文件中 model 部分的参数，用于定义 SDE 模型中的噪声水平的范围。
  # N 是配置文件中 model 部分的另一个参数，表示 SDE 模型中的尺度数目。接下来，将 sde.N 的值设置为 N，可能是为了更新 SDE 对象中的尺度数目。
  #  sampling_eps 变量，并将其赋值为 1e-5，可能是为了设置采样时的一个小的 epsilon 值。
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sde.N = N
  sampling_eps = 1e-5

img_size = 192
batch_size = 1
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

# parameters for Fourier CS recon
mask_type = 'uniform1d'
use_measurement_noise = False#是否使用测量噪声
acc_factor = 2.0
center_fraction = 0.15

# ADMM TV parameters
lamb_list = [0.005]
rho_list = [0.01]

random_seed = 0

sigmas = mutils.get_sigmas(config)#获取一组标准差值
scaler = datasets.get_data_scaler(config)#用于将数据进行缩放或标准化的对象。
inverse_scaler = datasets.get_data_inverse_scaler(config)#将缩放后的数据恢复到原始尺度的对象。
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
ema.copy_to(score_model.parameters())

#这行代码的作用是获取指定路径下所有以.npy 为扩展名的文件，并按字母顺序进行排序。
#Path(root) 创建了一个 Path 对象，表示指定的根目录路径。
#/ vol 将根目录路径与 vol 进行连接，得到指定的文件夹路径。
#glob('*.npy') 使用通配符 *.npy 进行文件匹配，查找所有以.npy为扩展名的文件。
#list(...) 将匹配到的文件路径转换为列表形式。
#sorted(...) 对列表中的文件路径进行排序，按照字母顺序进行升序排列。
#最后将排序后的文件路径列表赋值给变量fname_list。
#这段代码适用于从指定的根目录和文件夹中获取所有以 .npy 为扩展名的文件，并按照字母顺序对文件进行排序。变量 fname_list 将包含排序后的文件路径列表。
fname_list = sorted(list((Path(root) / vol).glob('*.npy')))
folder_list = sorted(list((Path(root1) / vol1).glob('*')))

all_img = []
i=0

# for fname in tqdm(fname_list):
# for fname in tqdm(folder_list):
#     h5f = h5py.File(str(fname) + "/mri_norm2.h5", 'r')
#     for i in range(88):
#       img= h5f['image'][:,:,i]
#       img_tensor = torch.from_numpy(img)
#       # img = np.load(fname)
#       # img = torch.from_numpy(img)
#       img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加额外的维度
#       img_tensor = F.interpolate(img_tensor, size=[192,192], mode='bilinear', align_corners=False)
#       # print("维度:", img_tensor.shape)
#       h, w = img_tensor.shape[2],img_tensor.shape[3]
#       # print("h,w:", h,w)
#       # img_tensor = img_tensor.view(1, 1, h, w)
#       # print("img:", img_tensor)
#       all_img.append(img_tensor)

fname=folder_list[0]
h5f = h5py.File(str(fname) + "/mri_norm2.h5", 'r')
for i in range(88):
    img= h5f['image'][:,:,i]
    img_tensor = torch.from_numpy(img)
      # img = np.load(fname)
      # img = torch.from_numpy(img)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加额外的维度
    img_tensor = F.interpolate(img_tensor, size=[192,192], mode='bilinear', align_corners=False)
      # print("维度:", img_tensor.shape)
    h, w = img_tensor.shape[2],img_tensor.shape[3]
      # print("h,w:", h,w)
      # img_tensor = img_tensor.view(1, 1, h, w)
      # print("img:", img_tensor)
    all_img.append(img_tensor)

all_img = torch.cat(all_img, dim=0)
#将体积正常化在适当的范围内
# normalize the volume to be in proper range
vmax = all_img.max()
all_img /= (vmax + 1e-5)

img = all_img.to(config.device)
b = img.shape[0]

for lamb in lamb_list:
    for rho in rho_list:
        print(f'lambda: {lamb}')
        print(f'rho:    {rho}')
        #指定保存目录以保存生成的样本
        # Specify save directory for saving generated samples
        save_root = Path(f'./results/{config_name}/{problem}/{mask_type}/acc{acc_factor}/lamb{lamb}/rho{rho}/{vol1}/{fname}')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        ###############################################
        # Inference
        ###############################################

        # forward model
        kspace = fft2(img)
        # print("kspace",kspace)
        # generate mask
        mask = get_mask(torch.zeros(1, 1, h, w), img_size, batch_size,
                        type=mask_type, acc_factor=acc_factor, center_fraction=center_fraction)
        mask = mask.to(img.device)
        mask = mask.repeat(b, 1, 1, 1)

        pc_fouriercs = controllable_generation_TV.get_pc_radon_ADMM_TV_mri(sde,
                                                                           predictor, corrector,
                                                                           inverse_scaler,
                                                                           mask=mask,
                                                                           lamb_1=lamb,
                                                                           rho=rho,
                                                                           img_shape=img.shape,
                                                                           snr=snr,
                                                                           n_steps=n_steps,
                                                                           probability_flow=probability_flow,
                                                                           continuous=config.training.continuous)

        # undersampling
        under_kspace = kspace * mask
        under_img = torch.real(ifft2(under_kspace))
        # print("under_img",under_img)

        count = 0
        for i, recon_img in enumerate(under_img):
            plt.imsave(save_root / 'input' / f'{count}.png', clear(under_img[i]), cmap='gray')
            plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
            count += 1
        print("under_img shape:", under_img.shape)
        print("under_kspace shape:", under_kspace.shape)
        x = pc_fouriercs(score_model, scaler(under_img), measurement=under_kspace)

        count = 0
        for i, recon_img in enumerate(x):
            plt.imsave(save_root / 'input' / f'{count}.png', clear(under_img[i]), cmap='gray')
            plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
            plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')
            np.save(str(save_root / 'input' / f'{count}.npy'), clear(under_img[i], normalize=False))
            np.save(str(save_root / 'recon' / f'{count}.npy'), clear(x[i], normalize=False))
            np.save(str(save_root / 'label' / f'{count}.npy'), clear(img[i], normalize=False))
            count += 1
            # 假设每个图像的尺寸是 (height, width)
            height, width = (192,192)
            # 创建一个空的3D数组来存储合成后的图像数据
            depth = 88
            volume = np.zeros((depth, height, width))
            tensor_cpu = x.cpu().numpy()
            volume[i, :, :] = tensor_cpu[i]

# 创建一个新的.h5文件
output_dir = 'results/output/reconstructed_data/data/LA/2018LA_Seg_Training Set'/fname
file_extension = '.h5'
output_file = os.path.join(output_dir, f"{'mri_norm2'}{file_extension}")

# 确保路径中的目录存在
os.makedirs(output_dir, exist_ok=True)

with h5py.File(output_file, 'w') as hf:
    # 将3D图像数据写入.h5文件中的数据集
    hf.create_dataset('volume', data=volume)

print("3D图像数据已保存至", output_file)
