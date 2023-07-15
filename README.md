# Egocentric 3D Hand Trajectory Forecasting
[Project]() **|** [ArXiv]() **|** [Demo]()

[Wentao Bao](https://cogito2012.github.io/homepage), 
[Lele Chen](https://www.cs.rochester.edu/u/lchen63), 
[Libing Zeng](https://libingzeng.github.io),
[Zhong Li](https://sites.google.com/site/lizhong19900216),
[Yi Xu](https://scholar.google.com/citations?user=ldanjkUAAAAJ&hl=en),
[Junsong Yuan](https://cse.buffalo.edu/~jsyuan),
[Yu Kong](https://www.egr.msu.edu/~yukong)

This is an official PyTorch implementation of USST model for the Ego3D-HTF task, which aims to forecast human hand trajectory in 3D physical space from an egocentric RGB video.

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
<img src="assets/demo.gif"  alt="demo" width = 480px height=270px>
</p>


## Installation

- Create a conda virtual environment with `python 3.7`:
```shell
  conda create -n usst python=3.7
  conda activate usst
```
- Install the latest `PyTorch`:
```shell
  pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
- Install other pip packages:
```shell
  pip install -r requirements.txt
```
- [Optional] Install `ffmpeg` if you need to visualize results by GIF animation:
```shell
  conda install ffmpeg
```

## Datasets

We released the dataset which can be downloaded here: [Google Drive](). If you are interested in the raw RGB-D recordings, please refer to [EgoPAT3D Dataset](https://github.com/ai4ce/EgoPAT3D).


## Testing

a. Download the pretrained model from [here](), or train the model from scratch following the instructions of the [Training](#training).

b. [Optional] Show the demos of a testing examples in our paper:
```shell
python demo_paper.py \
  --config config/usst_vit_final.yml \
  --tag usst_vit_final
```

c. Test and evaluate a trained model, e.g., usst_vit_final, on full testing set:
```shell
cd exp
bash test.sh 0 8 usst_vit_final
```
Evaluation results will be cached in `output/EgoPAT3D/usst_vit_final` and reported on the terminal.


## Training

a. Train the proposed USST model using GPU_ID=0 and 8 workers:
```shell
cd exp
nohup bash train.sh 0 8 usst_vit_final >train.log 2>&1 &
```
b. Monitor the training status using Tensorboard:
```shell
# open a new terminial
cd output/EgoPAT3D/usst_vit_final
tensorboard --logdir=./logs
# open the browser with the prompted localhost url.
```
c. [Optional] Train other model variants or baselines, e.g., SRNN
```shell
cd exp
nohup bash train_baseline.sh 0 8 baselines/srnn >train_srnn.log 2>&1 &
```


## Citation
If you find the code useful in your research, please cite:

    @inproceedings{BaoUSST2022,
      author = "Wentao Bao and Lele Chen and Libing Zeng and Zhong Li and Yi Xu and Junsong Yuan and Yu Kong",
      title = "Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting",
      year = "2022"
    }

## License

See [Apache-2.0 License](/LICENSE)

## Acknowledgement

We sincerely thank the owners of the following source code repos, which contribute to our released codes: [EgoPAT3D](https://github.com/ai4ce/EgoPAT3D/tree/main/preprocessing), [OCT](https://github.com/stevenlsw/hoi-forecast), [DVAE](https://github.com/XiaoyuBIE1994/DVAE), [VPT](https://github.com/KMnP/vpt), [torch_videovision](https://github.com/hassony2/torch_videovision),  [FullyConvResNet](https://programmer.group/5ef0376c8e2f2.html), [RAFT](https://github.com/princeton-vl/RAFT), and [pyrgbd](https://github.com/telegie/pyrgbd).