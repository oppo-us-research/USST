# Egocentric 3D Hand Trajectory Forecasting
[Project](https://actionlab-cv.github.io/EgoHandTrajPred) **|** [ArXiv](https://arxiv.org/pdf/2307.08243.pdf) **|** [Demo](assets/demo.gif)

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

<!-- <div class="video-container" style="display: flex;"> -->
  <div class="video" align="center">
    <a href="https://www.youtube.com/watch?v=MYY6GmqZSJA">
      <img src="https://img.youtube.com/vi/MYY6GmqZSJA/0.jpg" alt="Brief Intro." width=540px height=360px>
    </a>
    <p>Brief Intro. [<a href="https://www.youtube.com/watch?v=MYY6GmqZSJA">YouTube</a>]</p>
  </div>
  <!-- <div class="video" align="center">
    <img src="assets/demo.gif"  alt="demo" width = 360px height=240px>
    <p>Annotation Example</p>
  </div>
</div> -->



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
- This dataset can be downloaded here: [OneDrive](https://1drv.ms/f/c/0adef1d3209dfb47/Qkf7nSDT8d4ggAoxFQAAAAAAHkE-BxpcdN_G8g), which is collected by re-annotating the raw RGB-D recordings from [EgoPAT3D Dataset](https://github.com/ai4ce/EgoPAT3D). 

- After downloaded, place the downloaded `tar.gz` file under `data/EgoPAT3D/` and extract it: `tar zxvf EgoPAT3D-postproc.tar.gz`. The dataset folder should be structured as follows.
```shell
  data/EgoPAT3D/EgoPAT3D-postproc
                |-- odometry  # visual odometry data, e.g., "1/*/*.npy"
                |-- trajectory_repair  # trajectory data, e.g., "1/*/*.pkl"
                |-- video_clips_hand  # video clips, e.g., "1/*/*.mp4"
```

### H2O-PT
- This dataset can be downloaded here: [OneDrive](https://1drv.ms/f/c/0adef1d3209dfb47/Qkf7nSDT8d4ggAoyFQAAAAAAHS9hDgAkNfNlGg), which is collected by re-annotating the [H2O Dataset](https://taeinkwon.com/projects/h2o/). 

- After downloaded, place the downloaded `tar.gz` file under `data/H2O/` and extract it: `tar zxvf Ego3DTraj.tar.gz`. The dataset folder should be structured as follows.
```shell
data/H2O/Ego3DTraj
        |-- splits  # training splits ("train.txt", "val.txt", "test.txt")
        |-- traj  # trajectory data from pose (PT), e.g., "*.pkl", ...
        |-- video  # video clips, e.g., "*.mp4", ...
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

a. Test and evaluate a trained model, e.g., usst_vit_3d, on **EgoPAT3D-DT** testing set:
```shell
cd exp
bash test.sh 0 8 usst_vit_3d
```
b. Test and evaluate a trained model, e.g., usst_res18_3d, on **H2O-PT** testing set:
```shell
cd exp
bash trainval_h2o.sh 0 8 usst_res18_3d eval
```
Evaluation results will be cached in `output/[EgoPAT3D|H2O]/usst_vit_3d` and reported on the terminal.

c. To evaluate the 2D trajectory forecasting performance of a pretrained 3D target model, modify the config file `usst_xxx_3d.yml` to set `TEST.eval_space: norm2d`, then run the `test.sh` (or `trainval_h2o.sh`) again.

d. If only doing testing without training, please download our pretrained model from here: [OneDrive](https://1drv.ms/f/c/0adef1d3209dfb47/Qkf7nSDT8d4ggAozFQAAAAAAKF0IcEEpPpWW_g). After downloaded a zip file, place it under the `output/` folder, e.g., `output/EgoPAT3D/usst_res18_3d.zip` and then extract it: `cd output/EgoPAT3D && unzip usst_res18_3d.zip`. Then, run the run the `test.sh` (or `trainval_h2o.sh`).

e. [Optional] Show the demos of a testing examples in our paper:
```shell
python demo_paper.py --config config/usst_vit_3d.yml --tag usst_vit_3d
```

## Citation
If you find the code useful in your research, please cite:

    @inproceedings{BaoUSST_ICCV23,
      author = "Wentao Bao and Lele Chen and Libing Zeng and Zhong Li and Yi Xu and Junsong Yuan and Yu Kong",
      title = "Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting",
      booktitle = "International Conference on Computer Vision (ICCV)",
      year = "2023"
    }

Please also cite the EgoPAT3D paper if you use our EgoPAT3D-DT annotations:

    @InProceedings{Li_2022_CVPR,
      title = {Egocentric Prediction of Action Target in 3D},
      author = {Li, Yiming and Cao, Ziang and Liang, Andrew and Liang, Benjamin and Chen, Luoyao and Zhao, Hang and Feng, Chen},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2022}
    }

and H2O paper if you use our H2O-PT annotations:

    @InProceedings{Kwon_2021_ICCV,
      author = {Kwon, Taein and Tekin, Bugra and St\"uhmer, Jan and Bogo, Federica and Pollefeys, Marc},
      title = {H2O: Two Hands Manipulating Objects for First Person Interaction Recognition},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2021},
      pages = {10138-10148}
    }

## License

- Codes and checkpoints are licensed under the
[Apache 2.0 License](/LICENSE).   
- Dataset EgoPAT3D-DT is licensed following the [EgoPAT3D](https://github.com/ai4ce/EgoPAT3D), which is licensed under the [MIT License](https://github.com/ai4ce/EgoPAT3D/blob/main/LICENSE).   
- Dataset H2O-PT is licensed by following [ETH Zurich H2O Dataset Terms of Use](https://h2odataset.ethz.ch/).


## Acknowledgement

We sincerely thank the owners of the following source code repos, which are referred by our released codes: [EgoPAT3D](https://github.com/ai4ce/EgoPAT3D/tree/main/preprocessing), [hoi_forecast](https://github.com/stevenlsw/hoi-forecast), [pyk4a](https://github.com/etiennedub/pyk4a), [RAFT](https://github.com/princeton-vl/RAFT), and [NewCRFs](https://github.com/aliyun/NeWCRFs).
