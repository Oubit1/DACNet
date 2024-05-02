# Data Augmentation Strategies for Semi-Supervised Medical Image Segmentation
by Jiahui Wang, Mingfeng Jiang*, Dongsheng Ruan, Yang Li, Zefeng Wang, Yongquan Wu, Tao Tan, Guang Yang, Senior Member, IEEE, Ling Xia
## Introduction
This repository is for our paper '[Data Augmentation Strategies for Semi-SupervisedMedical Image Segmentation]
## Requirements
This repository is based on Pytorch 1.9.1, CUDA11.1 and Python 3.6.5
## Usage
### Install
Clone the repo:
```shell
git clone https://github.com/Cuthbert-Huang/CC-Net.git 
```
### Dataset
We use [the dataset of 2018 Atrial Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/).
We use [the dataset of Pancreas-CT](https://drive.google.com/file/d/1qzFUtkHx-46kFvHE7RAMhjAdo6dmn4iT/view?usp=sharing/).
We use [the dataset of ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html/).
### Preprocess
If you want to process .nrrd data into .h5 data, you can use `code/dataloaders/preprocess.py`.
### Diffusion model
Our code is origin from [score-MRI](https://github.com/HJ-harry/score-MRI)
Follow the class and function definitions involved in this code[inverse_problem_solver_ACDC_2d.py] can be found under that link.
### Pretrained models
The pretrained models were provided in (https://pan.quark.cn/s/a0cbd42de20f)password：DDsn.
### Train
python ./code/train_DACNet_3d.py --dataset_name LA --model acnet3d_v2 --exp DACNet --labelnum 8 --gpu 0 --temperature 0.1 --max_iteration 16000
python ./code/train_DACNet_2d.py --dataset_name ACDC --model acnet2d_v3 --exp DACNet --labelnum 14 --gpu 0 --temperature 0.1 --max_iteration 30000

### Test
python ./code/test_3d.py --dataset_name LA --model acnet3d_v2 --exp DACNet --labelnum 8 --gpu 0
python ./code/test_2d.py --dataset_name ACDC --model acnet2d_v3 --exp DACNet --labelnum 14 --gpu 0

## Acknowledgements
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), and [MC-Net+](https://github.com/ycwu1997/MC-Net),and [CC-Net+](https://github.com/Cuthbert-Huang/CC-Net) Thanks to these authors for their excellent work.
