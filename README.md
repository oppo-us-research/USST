# Egocentric 3D Hand Trajectory Forecasting
[Project]() **|** [ArXiv]() **|** [Demo]()

[Wentao Bao](https://cogito2012.github.io/homepage), 
[Lele Chen](https://www.cs.rochester.edu/u/lchen63), 
[Libing Zeng](https://libingzeng.github.io),
[Zhong Li](https://sites.google.com/site/lizhong19900216),
[Yi Xu](https://scholar.google.com/citations?user=ldanjkUAAAAJ&hl=en),
[Junsong Yuan](https://cse.buffalo.edu/~jsyuan),
[Yu Kong](https://www.egr.msu.edu/~yukong)

This is an official PyTorch implementation of the USST published in ICCV 2023. We release the dataset annotations (H2O-PT and EgoPAT3D-DT), PyTorch codes (training, inference, and demo), and the pretrained model weights.

## Table of Contents
1. [Task Overview](#task-overview)
1. [Installation](#installation)
1. [Datasets](#datasets)
1. [Demo & Testing](#testing)
1. [Training](#training)
1. [Citation](#citation)

## Task Overview
**Egocentric 3D Hand Trajectory Forecasting (Ego3D-HTF)** aims to predict the future 3D hand trajectory (in <span style="color:red">*red*</span> color) given the past observation of an egocentric RGB video and historical trajectory (in <span style="color:blue">*blue*</span> color). Compared to predicting the trajectory in 2D space,
predicting trajectory in global 3D space is practically more valuable to understand human intention for AR/VR applications.

<p align="center">
  <a href="https://www.youtube.com/watch?v=Qj3MXcPN0nY">
    <img src="https://img.youtube.com/vi/Qj3MXcPN0nY/0.jpg" alt="YouTube Video" width=360px height=240px>
  </a>
  <img src="assets/demo.gif"  alt="demo" width = 360px height=240px>
</p>


## Installation

- Create a conda virtual environment with `python 3.7`:
```shell
  conda create -n usst python=3.7
  conda activate usst
```
- Install the latest `PyTorch`:
```shell
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
- Install other python packages:
```shell
  pip install -r requirements.txt
```
- [Optional] Install `ffmpeg` if you need to visualize results by GIF animation:
```shell
  conda install ffmpeg
```

## Datasets

### EgoPAT3D-DT
This dataset can be downloaded here: [Google Drive](), which is collected by re-annotating the raw RGB-D recordings from [EgoPAT3D Dataset](https://github.com/ai4ce/EgoPAT3D). After downloaded, the dataset folder should be structured as follows.
```shell
data/EgoPAT3D/EgoPAT3D-postproc
--|odometry  # visual odometry data, e.g., "1/bathroomCabinet_1/c1_s47_e78_sx636_sy394_ex628_ey437.npy"
--|trajectory_repair  # trajectory data, e.g., "1/bathroomCabinet_1/c1_s47_e78_sx636_sy394_ex628_ey437.pkl"
--|video_clips_hand  # video clips, e.g., "1/bathroomCabinet_1/c1_s47_e78_sx636_sy394_ex628_ey437.mp4"
```

### H2O-PT
This dataset can be downloaded here: [Google Drive](), which is collected by re-annotating the [H2O Dataset](https://taeinkwon.com/projects/h2o/). After downloaded, the dataset folder should be structured as follows.
```shell
data/H2O/Ego3DTraj
--|splits  # training splits, e.g., "train.txt", "val.txt", "test.txt"
--|traj  # trajectory data from pose (PT), e.g., "sub1_h1_0.pkl", ...
--|video  # video clips, e.g., "sub1_h1_0.mp4", ...
```

## Training

a. Train the proposed ViT-based USST model on **EgoPAT3D-DT** dataset using GPU_ID=0 and 8 workers:
```shell
cd exp
nohup bash train.sh 0 8 usst_vit_3d >train_egopat3d.log 2>&1 &
```
b. Train the proposed ViT-based USST model on **H2O-PT** dataset using GPU_ID=0 and 8 workers:
```shell
cd exp
nohup bash trainval_h2o.sh 0 8 h2o/usst_vit_3d train >train_h2o.log 2>&1 &
```
c. This repo contains TensorboardX suport to monitor the training status:
```shell
# open a new terminial
cd output/EgoPAT3D/usst_vit_3d
tensorboard --logdir=./logs
# open the browser with the prompted localhost url.
```
d. Checkout other model variants in the `config/` folder, including the ResNet-18 backbones (`usst_res18_xxx.yml`), 3D/2D trajectory target (`usst_xxx_3d/2d.yml`), and 3D target in local camera reference (`usst_xxx_local3d`).


## Testing

a. Download the pretrained model from here: [Google Drive](), or train the model from scratch following the above [Training](#training) instructions.

b. Test and evaluate a trained model, e.g., usst_vit_3d, on **EgoPAT3D-DT** testing set:
```shell
cd exp
bash test.sh 0 8 usst_vit_3d
```
c. Test and evaluate a trained model, e.g., usst_vit_3d, on **H2O-PT** testing set:
```shell
cd exp
bash trainval_h2o.sh 0 8 usst_vit_3d eval
```
Evaluation results will be cached in `output/[EgoPAT3D|H2O]/usst_vit_3d` and reported on the terminal.

d. [Optional] Show the demos of a testing examples in our paper:
```shell
python demo_paper.py \
  --config config/usst_vit_3d.yml \
  --tag usst_vit_3d
```

## Citation
If you find the code useful in your research, please cite:

    @inproceedings{BaoUSST_ICCV23,
      author = "Wentao Bao and Lele Chen and Libing Zeng and Zhong Li and Yi Xu and Junsong Yuan and Yu Kong",
      title = "Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting",
      booktitle = "International Conference on Computer Vision (ICCV)",
      year = "2023"
    }

## License

See [Apache-2.0 License](/LICENSE).

## Acknowledgement

We sincerely thank the owners of the following source code repos, which contribute to our released codes: [EgoPAT3D](https://github.com/ai4ce/EgoPAT3D/tree/main/preprocessing), [VPT](https://github.com/KMnP/vpt), [torch_videovision](https://github.com/hassony2/torch_videovision),  [FullyConvResNet](https://programmer.group/5ef0376c8e2f2.html), [RAFT](https://github.com/princeton-vl/RAFT), and [pyrgbd](https://github.com/telegie/pyrgbd).