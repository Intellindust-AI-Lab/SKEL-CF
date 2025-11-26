# Environment-setup

## Conda Installation
The code was tested on pytorch 2.0 + cu117, xfromers==0.0.18, python=3.9.23, you may need to carefully adjust the version of torch, xfromers and cuda according to your GPUs.
```
conda env create -f environment.yml
```
```
conda activate skelvit
```
```
pip install --no-deps albucore==0.0.24
```
## Detectron2 Installation
To test images “in the wild,” you need to install the appropriate version of [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) corresponding to your specific GPU, CUDA, and PyTorch configuration.

# Data-preparation

We have uploaded our various datasets to the OneDrive. You can download the required resources from [this link](https://1drv.ms/f/c/89cf0bfd859af8e2/IgCPQSPYHWA1T495bGcBbJCqAVPl4MX0-1-_8G4LdbYRxSA?e=8JHYI7).

## SMPL & SKEL Bodys
For convenience, we provide the body_models folder containing the SKEL and SMPL models. You only need to download **body_models** from OneDrive and place it under the data_inputs/ directory. After extraction, the file structure will be as follows:

    data_inputs/
    └── body_models/
        ├── skel/
        │   ├── Geometry/
        │   ├── sample_motion/
        │   ├── bsm.osim
        │   ├── changelog_v1.1.1.txt
        │   ├── skel_female.pkl
        │   ├── skel_male.pkl
        │   └── tmp.osim
        ├── SMPL/
        │   ├── SMPL_FEMALE.pkl
        │   ├── SMPL_MALE.pkl
        │   └── SMPL_NEUTRAL.pkl
        ├── SMPLX/
        │   ├── SMPLX_FEMALE.npz
        │   ├── SMPLX_MALE.npz
        │   └── SMPLX_NEUTRAL.npz
        ├── train-eval-utils/
        ├── J_regressor_SKEL_mix_MALE.pkl
        ├── J_regressor_SMPL_MALE.pkl
        ├── read_file.py
        ├── smpl_mean_params.npz

## Checkpoints
### Pretrained Camera Model Checkpoint & ViTPose Backbone Checkpoint
Download **backbone** from OneDrive and place it under the data_inputs/ directory. The file structure will be as follows:

    data_inputs/
    └── backbone/
        ├── cam_model_cleaned.ckpt
        └── vitpose_backbone.pth

### SKEL-CF Checkpoint
Download **best.pth** from OneDrive and place it at data_outputs/exp/ (or wherever you like, but rember to adjust the path in eval.sh and rendering bash shells)

## Training Data
We recommend that you follow [4DHuman](https://github.com/shubham-goel/4D-Humans) to download the corresponding datasets. We provide guidance on how these data should be ultimately organized.
### Training labels
Download `skel-training-labels` folder from OneDrive and place it under the `data_inputs/` directory. The file structure should be:

    data_inputs/
    └── skel-training-labels/
        ├── aic-release_skel.npz
        ├── coco-release_skel.npz
        ├── h36m-release_skel.npz
        ├── insta1-release_skel.npz
        ├── insta2-release_skel.npz
        ├── mpi-inf-release_skel.npz
        └── mpii-release_skel.npz
### Training images

    data_inputs/
    └── skel-training-images/
        ├── aic-train
        ├── coco-train
        ├── h36m-train
        ├── insta-train
        ├── mpi-inf-train
        └── mpii-train
## Evaluation Data
We evaluated our method on several benchmark datasets, including COCO, LSP-EXTENDED, 3DPW, EMDB and H36M used by [HMR2.0](https://github.com/shubham-goel/4D-Humans). In addition, we also tested on the yoga action dataset [MOYO](https://moyo.is.tue.mpg.de/), as well as its more challenging subset, MOYO-HARD. Some datasets are relatively difficult to process, so just take your time.

### Evaluation labels
    data_inputs/
    └── skel-evaluation-labels/
        ├── moyo_hard.npz
        ├── moyo_v2.npz
        ├── h36m_val_p2.npz
        ├── coco_val.npz
        ├── spec_test.npz
        └── emdb_test.npz

### Evaluation Images

    data_inputs/
    └── skel-evaluation-data/
        ├── 3dpw
        ├── coco
        ├── emdb
        ├── h36m-select
        ├── moyo
        └── spec-syn
