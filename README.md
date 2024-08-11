## 1 install
```
Python 3.6, Tensorflow 2.6, CUDA 11.4 and cudnn ( /usr/local/cuda-11.4 没有用 conda 的cudatoolkit)
git clone --depth=1 https://github.com/wangzihao77/ASFE-Net.git
conda create -n randlanet python=3.6 
conda activate randlanet
pip install tensorflow-gpu==2.6 -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
pip install -r helper_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
sh compile_op.sh
```
## 2 data-S3DIS
```
ls /home/$USER/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version
python utils/data_prepare_s3dis.py
```

## 3 配置
```
ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None
```
## 4 train

```
python -B main_S3DIS.py --gpu 0 --mode train --test_area 1 表示:
Area2-3-4-5-6共228个ply文件作为训练集
Area1共44个ply文件作为验证集
下面为6折交叉验证：
cat jobs_6_fold_cv_s3dis.sh 
python -B main_S3DIS.py --gpu 0 --mode train --test_area 1  # Area23456作为训练集，area1作为验证集
python -B main_S3DIS.py --gpu 0 --mode test --test_area 1
python -B main_S3DIS.py --gpu 0 --mode train --test_area 2
python -B main_S3DIS.py --gpu 0 --mode test --test_area 2
python -B main_S3DIS.py --gpu 0 --mode train --test_area 3
python -B main_S3DIS.py --gpu 0 --mode test --test_area 3
python -B main_S3DIS.py --gpu 0 --mode train --test_area 4
python -B main_S3DIS.py --gpu 0 --mode test --test_area 4
python -B main_S3DIS.py --gpu 0 --mode train --test_area 5
python -B main_S3DIS.py --gpu 0 --mode test --test_area 5
python -B main_S3DIS.py --gpu 0 --mode train --test_area 6
python -B main_S3DIS.py --gpu 0 --mode test --test_area 6


## 5 vis
label
python  main_S3DIS.py --mode vis --test_area 1
origin
python vis_S3DIS.py 

## 6 code
helper_tf_util.py	封装了一些卷积池化操作代码
helper_tool.py	      有训练时各个数据集所用到的一些参数信息，还有一些预处理数据时的一些模块。
main_*.py	        训练对应数据的主文件
ASFE-Net.py	       定义网络的主题结构
tester_*.py	      测试对应数据的文件，该文件在main_*.py中被调用
utils	            对数据集预处理的模块以及KNN模块。
utils/data_prare_*.py 预处理，把point和label做了grid sampling，并且生成了一个kdtree保存下来


## 7 NET
将特征升维到8
encoder：由4个（dilated_res_block+random_sample）构成，形成特征金字塔
将金字塔尖的特征再次计算以下
decoder：由4个（nearest_interpolation+conv2d_transpose）构成，恢复到point-wise的特征
由point-wise经过一些MLP，得到f_out





	
### (1) Setup
This code has been tested with Python 3.6, Tensorflow 2.6, CUDA 11.4.0 and cuDNN 7.4.1 on Ubuntu 22.4.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/wangzihao77/ASFE-Net.git && cd ASFENet
```
- Setup python environment
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

**Update 03/21/2020, pre-trained models and results are available now.** 
You can download the pre-trained models and results [here](https://drive.google.com/open?id=1iU8yviO3TP87-IexBXsu13g6NklwEkXB).
Note that, please specify the model path in the main function (e.g., `main_S3DIS.py`) if you want to use the pre-trained model and have a quick try of our RandLA-Net.

### (2) S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Start 6-fold cross validation:
```
sh jobs_6_fold_cv_s3dis.sh
```
- Move all the generated results (*.ply) in `/test` folder to `/data/S3DIS/results`, calculate the final mean IoU results:
```
python utils/6_fold_cv.py
```










## Related Repos
1. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
2. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
5. [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000x Fewer Labels](https://github.com/QingyongHu/SQN) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SQN.svg?style=flat&label=Star)


