[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-learning-of-generalized-game/image-classification-on-sports10)](https://paperswithcode.com/sota/image-classification-on-sports10?p=contrastive-learning-of-generalized-game)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-learning-of-generalized-game/representation-learning-on-sports10)](https://paperswithcode.com/sota/representation-learning-on-sports10?p=contrastive-learning-of-generalized-game)

# Contrastive Learning of Generalized Game Representations

### [IEEE CoG 2021 Paper](https://arxiv.org/abs/2106.10060) | [*Sports10* Dataset](https://drive.google.com/drive/folders/137Byy_ngEp_dFnzKpnCK1wxyzYxnhamE?usp=sharing) | [Pre-Trained Models]()

This repository is the official code implementation of the paper **"Contrastive Learning of Generalized Game Representations"** by [Chintan Trivedi](deepgamingai.com), [Antonios Liapis](http://antoniosliapis.com/) and [Georgios Yannakakis](https://yannakakis.net/).

<img src='./imgs/t-SNE Results.png'/>


## Installation
```
git clone git@github.com:ChintanTrivedi/contrastive-game-representations.git
cd contrastive-game-representations
pip install -r requirements.txt
```
Note that this code has been developed/tested with a single NVIDIA GPU using [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [Tensorflow 2.3](https://www.tensorflow.org/install) on Windows platform. It is expected to work as-is on Linux/Colab platforms (untested).


## Download *Sports10* Dataset
We present a new dataset containing ***100,000 Gameplay Images*** of ***175 Video Games*** across ***10 Sports Genres***. The games are also divided into three visual styling categories: ***RETRO*** (arcade-style, 1990s and earlier), ***MODERN*** (roughly 2000s) and ***PHOTOREAL*** (roughly late 2010s).

<img src='./datasets/Sports10 Banner Image.png'/>

- Download the full dataset (~10GB) from [google drive](https://drive.google.com/drive/folders/137Byy_ngEp_dFnzKpnCK1wxyzYxnhamE?usp=sharing) and extract the zip file's contents to the ```datasets``` directory in the project.
- Meta-data is also available with the full [list of games](https://drive.google.com/file/d/1OywBuQjjEjxBAKFL7QzE3QRoVQVON4wO/view?usp=sharing).
- For more information on the dataset, please refer our [paper](https://arxiv.org/abs/2106.10060).


## Training models from scratch
After downloading and unzipping the dataset, learn game representations with a ResNet50 encoder by running the following commands on your terminal.

**1. Plain Supervised Learning**
```
python train_fulsup.py --dataset_directory="./datasets/Sports10" --epochs=10 --img_shape=224 --batch_size=64
```
**2. Supervised Contrastive Learning**
```
python train_supcon.py --dataset_directory="./datasets/Sports10" --epochs=10 --img_shape=224 --batch_size=64
```
For additional training arguments, check [train_fulsup.py](./train_fulsup.py) or [train_supcon.py](./train_supcon.py)

## Download Pre-Trained Model
Alternatively, you can download our pre-trained models from this [google drive link](). Models can be loaded using ```tensorflow.keras.models.load_model('$MODEL_FILENAME.h5')``` and fine-tuned for any downstream task involving RL or GANs.

## Citation
```
@inproceedings{trivedi2021contrastive,
  title={Contrastive Learning of Generalized Game Representations},
  author={Trivedi, Chintan and Liapis, Antonios and Yannakakis, Georgios N},
  booktitle={2021 IEEE Conference on Games (CoG)},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgements
1. [Improving Image Classifiers With Supervised Contrastive Learning](https://wandb.ai/authors/scl/reports/Improving-Image-Classifiers-With-Supervised-Contrastive-Learning--VmlldzoxMzQwNzE) by [Sayak Paul](https://github.com/sayakpaul)
2. [Contrastive Loss Functions](https://github.com/wangz10/contrastive_loss) by [Zichen Wang](https://github.com/wangz10)
