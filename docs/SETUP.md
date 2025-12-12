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

## body_models Installation
Due to copyright considerations, we cannot directly provide files/folders in [body_models](https://github.com/Intellindust-AI-Lab/SKEL-CF/tree/main/body_models), you should get skel folder comes from [SKEL](https://github.com/MarilynKeller/SKEL/tree/master/skel) and skel_utils folder and the remaining related code from [HSMR](https://github.com/IsshikiHugh/HSMR/tree/main/lib/body_models) and put these folders/files into the corresponding locations as shown below. 

Some external code and libraries include dependencies on and calls to their original repositories, so you may need to adjust these parts yourself (for example, the issue encountered in [issue#5](https://github.com/Intellindust-AI-Lab/SKEL-CF/issues/5)). If you run into any problems during the installation process, please feel free to raise them in an issue.

    body_models/
    ├── skel/
    │   ├── alignment/
    │   │   ├── align_config_joint.py
    │   │   ├── align_config.py
    │   │   ├── aligner.py
    │   │   ├── losses.py
    │   │   └── utils.py
    │   └── fit_osim/
    │       ├── mappings/
    │       ├── Fitting_SKEL_to_osim.md
    │       ├── osim_aug.py
    │       └── osim_fitter.py
    ├── __init__.py
    ├── config.py
    ├── joints_def.py
    ├── kin_skel.py
    ├── osim_rot.py
    ├── skel_model.py
    ├── utils.py
    ├── skel_utils/
    │   ├── augmentation.py
    │   ├── definition.py
    │   ├── limits.py
    │   └── transforms.py
    └── skel_wrapper.py

# Data-preparation

Choose your setup path:
- [Quick Start](#-quick-start): Run the demo in minutes with pre-configured settings
- [Advanced Setup](#-Advanced-setup): Deep dive into the method details and custom configurations

<!-- We have uploaded our various datasets to the OneDrive. You can download the required resources from [this link](https://1drv.ms/f/c/89cf0bfd859af8e2/IgCPQSPYHWA1T495bGcBbJCqAVPl4MX0-1-_8G4LdbYRxSA?e=8JHYI7). -->
## 🚀 Quick start

### Prerequisites: Download Body Models

Before running the code, you need to download these files:

| Model | Description | Download Link |
|-------|-------------|---------------|
| SKEL | Skeleton body model | [Download](https://skel.is.tue.mpg.de/download.php) |
| J_regressor | Auxiliary joint regressor | [Download](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84OWNmMGJmZDg1OWFmOGUyL0lnQ1BRU1BZSFdBMVQ0OTViR2NCYkpDcUFWUGw0TVgwLTEtXzhHNExkYllSeFNBP2U9OEpIWUk3&id=89CF0BFD859AF8E2%21sd823418f601d4f358f796c67016c90aa&cid=89CF0BFD859AF8E2) |
| SMPL | SMPL body model | [Download](https://smpl.is.tue.mpg.de/download.php) |

Once downloaded, organize the files according to the directory structure below:

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
        ├── J_regressor_h36m.pkl
        ├── SMPL_TO_J19.pkl
        ├── J_regressor_SKEL_mix_MALE.pkl
        └── J_regressor_SMPL_MALE.pkl

### SKEL-CF Checkpoint
Download **SKEL-CF.pth** from [SKEL-CF](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84OWNmMGJmZDg1OWFmOGUyL0lnQ1BRU1BZSFdBMVQ0OTViR2NCYkpDcUFWUGw0TVgwLTEtXzhHNExkYllSeFNBP2U9OEpIWUk3&id=89CF0BFD859AF8E2%21sd823418f601d4f358f796c67016c90aa&cid=89CF0BFD859AF8E2) and place it at data_outputs/exp/ (or wherever you like, but rember to adjust the path in eval.sh and rendering bash shells)

## ⚙️ Advanced setup

### Pretrained Camera Model Checkpoint & ViTPose Backbone Checkpoint
You should download the pretrained checkpoint " cam_model_cleaned.ckpt" from the [CameraHMR](https://camerahmr.is.tue.mpg.de), it's called "HumanFov Model" on their website, with a size about 768 MB. Also, you should download vitpose_backbone.pth from [VITPOSE](https://github.com/ViTAE-Transformer/ViTPose) with simple decoder (about 2.35GB), we use vitpose-H as baseline, you can also choose S/B/L model if you don't have enough VRAM, and place them under the data_inputs/backone directory. The file structure will be as follows:

    data_inputs/
    └── backbone/
        ├── cam_model_cleaned.ckpt
        └── vitpose_backbone.pth


### Training images
1. Download the training datasets following the [HSMR](https://github.com/IsshikiHugh/HSMR) instructions.
2. Extract webdataset archives into individual image files.
3. Organize the images according to the directory structure shown below.

```shell
    data_inputs/
    └── skel-training-images/
        ├── aic-train
        ├── coco-train
        ├── h36m-train
        ├── insta-train
        ├── mpi-inf-train
        └── mpii-train
```
### Training labels
Download `skel-training-labels` folder from OneDrive and place it under the `data_inputs/` directory. The file structure should be:

```shell
    data_inputs/
    └── skel-training-labels/
        ├── aic-release_skel.npz
        ├── coco-release_skel.npz
        ├── h36m-release_skel.npz
        ├── insta1-release_skel.npz
        ├── insta2-release_skel.npz
        ├── mpi-inf-release_skel.npz
        └── mpii-release_skel.npz
```

# Evaluation Data
We evaluated our method on several benchmark datasets, including COCO, 3DPW, EMDB, SPEC-SYN and H36M used by [HSMR](https://github.com/IsshikiHugh/HSMR) or [CameraHMR](https://camerahmr.is.tue.mpg.de/). We also tested on the yoga action dataset [MOYO](https://moyo.is.tue.mpg.de/), as well as its more challenging subset, MOYO-HARD. Some datasets are relatively difficult to process, so just take your time.

## Evaluation labels
Download the evaluation labels from [SKEL-CF](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84OWNmMGJmZDg1OWFmOGUyL0lnQ1BRU1BZSFdBMVQ0OTViR2NCYkpDcUFWUGw0TVgwLTEtXzhHNExkYllSeFNBP2U9OEpIWUk3&id=89CF0BFD859AF8E2%21sd823418f601d4f358f796c67016c90aa&cid=89CF0BFD859AF8E2)
```shell
    data_inputs/
    └── skel-evaluation-labels/
        ├── moyo_hard.npz
        ├── moyo_v2.npz
        ├── h36m_val_p2.npz
        ├── coco_val.npz
        ├── spec_test.npz
        ├── 3dpw_test.npz
        └── emdb_test.npz
```    

## Evaluation Images
Download the evaluation images from the following sources:
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
- [EMDB](https://eth-ait.github.io/emdb/)
- [COCO](https://cocodataset.org/#download)
- [SPEC-SYN](https://spec.is.tue.mpg.de/)
```shell
    data_inputs/
    └── skel-evaluation-data/
        ├── 3dpw
        ├── coco
        ├── emdb
        ├── h36m-select
        ├── moyo
        └── spec-syn
```
