# Contrastive Learning of Generalized Game Representations

### [IEEE CoG 2021 Paper]() | [*Sports10* Dataset](https://drive.google.com/drive/folders/137Byy_ngEp_dFnzKpnCK1wxyzYxnhamE?usp=sharing) | [Pre-Trained Models]()

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

<img src='./datasets/Sports10 Dataset Preview.png'/>

Download the full dataset (~10GB) from [google drive](https://drive.google.com/drive/folders/137Byy_ngEp_dFnzKpnCK1wxyzYxnhamE?usp=sharing).
Meta-data is also available with the full [list of games](https://drive.google.com/file/d/1OywBuQjjEjxBAKFL7QzE3QRoVQVON4wO/view?usp=sharing).
For more information on the dataset, please refer our [paper]().


## Run Fully Supervised Training
```
python train_fulsup.py
```

## Run Supervised Contrastive Training
```
python train_supcon.py
```

## Download Pre-Trained Model


## Citation


## Acknowledgements
