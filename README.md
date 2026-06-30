<h2 align="center">
  [ECCV 2026] SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery
</h2>

<p align="center">
    <a href="https://github.com/Intellindust-AI-Lab/SKEL-CF/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2511.20157v6">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2511.20157-red">
    </a>
   <a href="https://pokerman8.github.io/SKEL-CF/">
        <img alt="project webpage" src="https://img.shields.io/badge/Webpage-SkelCF-purple">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/SKEL-CF/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Intellindust-AI-Lab/SKEL-CF">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/SKEL-CF/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Intellindust-AI-Lab/SKEL-CF?color=olive">
    </a>
    <a href="https://github.com/Intellindust-AI-Lab/SKEL-CF">
        <img alt="stars" src="https://img.shields.io/github/stars/Intellindust-AI-Lab/SKEL-CF">
    </a>
    <a href="mailto:shenxi@intellindust.com">
        <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-yellow">
    </a>
</p>

<p align="center">
    TL;DR: SKEL-CF is a coarse-to-fine transformer framework for estimating anatomically accurate SKEL parameters from 3D human data. By converting 4DHuman to SKEL-aligned 4DHuman-SKEL and incorporating camera modeling, it addresses data scarcity and depth/scale ambiguities. SKEL-CF outperforms prior SKEL-based methods on MOYO (85.0 MPJPE / 51.4 PA-MPJPE), offering a scalable, biomechanically faithful solution for human motion analysis.
</p>

---

<div align="center">
  <a href="https://dali936.github.io/">Da Li</a><sup>1,2,*</sup>,&nbsp;&nbsp;
  <a href="https://pokerman8.github.io/">Jiping Jin</a><sup>1,3,*</sup>,&nbsp;&nbsp;
  <a href="https://xuanlong-yu.github.io/">Xuanlong Yu</a><sup>1</sup>,&nbsp;&nbsp;
  Wei Liu<sup>5</sup>,&nbsp;&nbsp;
  <a href="https://vinthony.github.io/academic/">Xiaodong Cun</a><sup>4</sup>,&nbsp;&nbsp;<br/>
  Kai Chen<sup>5</sup>,&nbsp;&nbsp;
  Rui Fan<sup>3</sup>,&nbsp;&nbsp;
  Jiangang Kong<sup>5</sup>,&nbsp;&nbsp;
  <a href="https://xishen0220.github.io">Xi Shen</a><sup>1,†</sup>
</div>

<br/>
<p align="center">
<i>
*Equal contribution &nbsp;&nbsp; †Corresponding author <br>
</i>
</p>

<p align="center">
<sup>1</sup><a href="https://intellindust-ai-lab.github.io">Intellindust AI Lab</a>&nbsp;&nbsp;
<sup>2</sup>ShenZhen University &nbsp;&nbsp;
<sup>3</sup>ShanghaiTech University &nbsp;&nbsp;<br/>
<sup>4</sup>Great Bay University &nbsp;&nbsp;
<sup>5</sup>Didi Chuxing Co.Ltd
</p>

<p align="center">
<strong>If you like our work, please give us a ⭐!</strong>
</p>


<p align="center">
  <img src="https://github.com/Pokerman8/SKEL-CF/blob/main/static/images/teaser.jpg">
</p>

</details>

##  🗓️ Updates
- [x] **\[2025.11.26\]** Release SKEL-CF.
- [x] **\[2025.11.27\]** Release checkpoints and labels on [Hugging Face](https://huggingface.co/Intellindust/SKEL_CF_vitpose_H).

## 🧭 Table of Content
* [1. ⚒️ Setup](#%EF%B8%8F-setup)
* [2. 🚀 Demo & Quick Start](#-demo--quick-start)
* [3. 🧱 Reproducibility](#-reproducibility)
* [4. 👀 Visual Results](#-visual-results)
* [5. 📝 Citation](#-citation)
* [6. 📜 Acknowledgement](#-acknowledgement)
* [7. 🌟 Star History](#-star-history)


## ⚒️ Setup

1. [🌏 Environment Setup](./docs/SETUP.md#environment-setup)
2. [📦 Data Preparation](./docs/SETUP.md#data-preparation)


## 🚀 Demo & Quick Start
Quick start with images:

```shell

bash vis/run_demo.sh
```

Quick start with videos:

```shell

bash vis/run_video.sh
```

## 🧱 Reproducibility

For reproducing the results in the paper, please refer to [`docs/EVAL.md`](./docs/EVAL.md) and [`docs/TRAIN.md`](./docs/TRAIN.md).

## 👀 Visual Results

### Per-Layer Refinement
![Per-Layer Refinement 2](https://raw.githubusercontent.com/Pokerman8/SKEL-CF/main/static/images/per-layer-2.gif)
![Per-Layer Refinement 1](https://raw.githubusercontent.com/Pokerman8/SKEL-CF/main/static/images/per-layer-1.gif)
### Sports Video
<a href="https://raw.githubusercontent.com/Pokerman8/SKEL-CF/main/static/videos/badminton01.mp4">
  <img src="https://img.shields.io/badge/▶️-Badminton%20Video-FF6B6B?style=for-the-badge&logo=video" alt="Badminton Video" />
</a>


<a href="https://raw.githubusercontent.com/Pokerman8/SKEL-CF/main/static/videos/skate01.mp4">
  <img src="https://img.shields.io/badge/▶️-Skate%20Video-4ECDC4?style=for-the-badge&logo=video" alt="Skate Video" />
</a>

> 💡 **Tip**: Click the buttons above to watch videos, or visit our [project page](https://pokerman8.github.io/SKEL-CF/) for more visual results.


## 📝 Citation
If you use `SKEL-CF` or its methods in your work, please cite the following BibTeX entries:

```latex
@inproceedings{li2025skelcf,
  title={SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery},
  author={Li, Da and Jin, Jiping and Yu, Xuanlong and Cun, Xiaodong and Chen, Kai and Fan, Rui and Kong, Jiangang and Shen, Xi},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```
## 📜 Acknowledgement

Parts of the code are adapted from the following repos: [SKEL](https://github.com/MarilynKeller/SKEL), [CameraHMR](https://github.com/pixelite1201/CameraHMR), [HSMR](https://github.com/IsshikiHugh/HSMR), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [Detectron2](https://github.com/facebookresearch/detectron2)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Intellindust-AI-Lab/SKEL-CF&type=date&legend=top-left)](https://www.star-history.com/#Intellindust-AI-Lab/SKEL-CF&type=date&legend=top-left)
