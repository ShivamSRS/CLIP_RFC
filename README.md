# CLIP + Residual Feature Connection

Residual Feature Connection(RFC) aims to fuse the task-specific knowledge learned from the target domain and the original knowledge pre-trained from CLIP. We show that RFC can adapt pre-trained CLIP to downstream pathology tasks and achieve good performance with just a few annotated samples. Specifically, RFC achieves over 19% improvement on accuracy when only using 0.1% of labeled data in PCam with only 10 minutes of fine-tuning while running on a single RTX 2080Ti.

## Model Overview
<img width="750" alt="Residual_Feature_Connection_v5" src="https://user-images.githubusercontent.com/40489953/213370883-ed6b540b-de66-44f2-bc66-2d88d58b4f63.png">

## Installation

### Environment Set up

```
$ conda create --name <env_name> --file requirements.txt
```
#### Setup Weights & Biaes
```
$ wandb login
```
### Prepare Dataset
We put all our dataset in the same folder for simplicity

```
datasets/
  DATA/
    - pcamv1
    - MHIST
```

## Running our model
### Training Pcam
```
$ python Pcam/src/script/train_CustomCLIP.py --seed 1 --percent $percent --alpha $alpha
```

### Training MHIST
```
$ python MHIST/src/script/train_CustomCLIP.py --seed 1 --percent $percent --alpha $alpha
```

### Testing
Testing Result including, Acc, recall, precision, f1, and AUC, will automaticaly link to the wandb account you created.

## Evluation

### Draw ROC Curve
```
# PCam
$ python Pcam/src/script/generate_roc.py $percent

# MHIST
$ python MHIST/src/script/generate_roc.py $percent
```

### Draw T-SNE(PCam Only)
```
$ python Pcam/src/script/t_sne.py $alpha
```


