# IJCV (2025): TryOn-Adapter
This repository is the official implementation of [TryOn-Adapter](https://arxiv.org/abs/2404.00878)

> **TryOn-Adapter: Efficient Fine-Grained Clothing Identity Adaptation for High-Fidelity Virtual Try-On**<br>

>
> Jiazheng Xing, Chao Xu, Yijie Qian, Yang Liu, Guang Dai, Baigui Sun, Yong Liu, Jingdong Wang

[[arXiv Paper](https://arxiv.org/abs/2404.00878)]&nbsp;

![teaser](assets/teaser.jpg)&nbsp;

## TODO List
- [x] ~~Release Texture Highlighting Map and Segmentation Map~~
- [x] ~~Release Data Preparation Code~~
- [x] ~~Release Inference Code~~
- [x] ~~Release Model Weights~~ 
- [ ] Release Training Code

## Getting Started
### Installation
1. Clone the repository
```shell
git clone https://github.com/jiazheng-xing/TryOn-Adapter.git
cd TryOn-Adapter
```
2. Install Python dependencies
```shell
conda env create -f environment.yaml
conda activate tryon-adapter
```

### Data Preparation
#### VITON-HD
1. The VITON-HD dataset serves as a benchmark. Download [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset.

2. In addition to above content, some other preprocessed conditions are in use in TryOn-Adapter.  The preprocessed data could be downloaded, respectively. The detail information and code see [data_preparation/README.md](data_preparation/README.md). 

   |Content|Google|Baidu|
   |---|---|---|
   |Segmentation Map|[link](https://drive.google.com/file/d/18KvGWR-3siJ_mt7g4CcEVFi_51E7ZifA/view?usp=sharing)|[link](https://pan.baidu.com/s/1zm3XV34tcrXpYt6uAN4R9Q?pwd=ekyn)|
   |Highlighting Texture Map|[link](https://drive.google.com/file/d/111KBYA8-d9xl9a2aS9yUaTp0edflb7qT/view?usp=sharing)|[link](https://pan.baidu.com/s/1xWnvF7TeKB_2AzlCEbPsAQ?pwd=jnlz)|

3. Generate Warped Cloth and Warped Mask based on the [GP-VTON](https://github.com/xiezhy6/GP-VTON.git).

Once everything is set up, the folders should be organized like this:
```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── image-parse-v3
│   │   │   ├── [000006_00.png | 000008_00.png | ...]
│   │   ├── openpose_img
│   │   │   ├── [000006_00_rendered.png | 000008_00_rendered.png | ...]
│   │   ├── openpose_json
│   │   │   ├── [000006_00_keypoints.json | 000008_00_keypoints.json | ...]
│   │   ├── train_paired/test_(un)paired
│   │   │   ├── mask      [000006_00.png | 000008_00.png | ...]
│   │   │   ├── seg_preds [000006_00.png | 000008_00.png | ...]
│   │   │   ├── warped    [000006_00.png | 000008_00.png | ...]
```

#### DressCode
1. The DressCode dataset serves as a benchmark. Download the [DressCode](https://github.com/aimagelab/dress-code) dataset.

2. In addition to above content, some other preprocessed conditions are in use in TryOn-Adapter. The detail information and code see [data_preparation/README.md](data_preparation/README.md). 

3. Generate Warped Cloth and Warped Mask based on the [GP-VTON](https://github.com/xiezhy6/GP-VTON.git).

Once everything is set up, the folders should be organized like this:
```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
|   ├── train_pairs.txt
|   ├── train_pairs.txt
│   ├── [test_paird | test_unpaird | train_paird]
│   │   ├── [dresses | lower_body | upper_body]
│   │   │   │   ├── mask      [013563_1.png| 013564_1.png | ...]
│   │   │   │   ├── seg_preds [013563_1.png| 013564_1.png | ...]
│   │   │   │   ├── warped    [013563_1.png| 013564_1.png | ...]
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
|   |   ├── train_pairs.txt
│   │   ├── images
│   │   │   ├── [013563_0.jpg | 013563_1.jpg | 013564_0.jpg | 013564_1.jpg | ...]
│   │   ├── masks
│   │   │   ├── [013563_1.png| 013564_1.png | ...]
│   │   ├── keypoints
│   │   │   ├── [013563_2.json | 013564_2.json | ...]
│   │   ├── label_maps
│   │   │   ├── [013563_4.png | 013564_4.png | ...]
│   │   ├── skeletons
│   │   │   ├── [013563_5.jpg | 013564_5.jpg | ...]
│   │   ├── dense
│   │   │   ├── [013563_5.png | 013563_5_uv.npz | 013564_5.png | 013564_5_uv.npz | ...]
```
### Inference
Please download the pretrained model from [HuggingFace](https://huggingface.co/Ockham98/TryOn-Adapter).
To perform inference on the Dress Code or VITON-HD dataset, use the following command:
```shell
python test_viton.py/test_dresscode.py --plms --gpu_id 0 \
--ddim_steps 100 \
--outdir <path> \
--config [configs/viton.yaml | configs/dresscode.yaml] \
--dataroot <path> \
--ckpt <path> \
--ckpt_elbm_path <path> \
--use_T_repaint [True | False] \
--n_samples 1 \
--seed 23 \
--scale 1 \
--H 512 \
--W 512 \
--unpaired
```

```shell
--ddim_steps <int>         sampling steps
--outdir <str>             output direction path
--config <str>             config path of viton-hd/dresscode
--ckpt <str>               diffusion model checkpoint path
--ckpt_elbm_path <str>     elbm module checkpoint dirction path
--use_T_repaint <bool>     whether to use T-Repaint technique
--n_samples <int>          numbers of samples per inference
--unpaired                 whether to use the unpaired setting
```

or just simply run:
```shell
bash test_viton.sh
bash test_dresscode.sh
```


## Acknowledgements
Our code is heavily borrowed from [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example). We also thank [GP-VTON](https://github.com/xiezhy6/GP-VTON.git), our warping garments are generated from it.

## Citation
```
@article{xing2025tryon,
  title={TryOn-Adapter: Efficient Fine-Grained Clothing Identity Adaptation for High-Fidelity Virtual Try-On},
  author={Xing, Jiazheng and Xu, Chao and Qian, Yijie and Liu, Yang and Dai, Guang and Sun, Baigui and Liu, Yong and Wang, Jingdong},
  journal={International Journal of Computer Vision},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```