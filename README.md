# 4D Recontruction
TU-Berlin HTCV-Project 2024 SS

This implementation of 4D Reconstruction is based on [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians), which utilizes submodules such as differential-gaussian-rasterization, the same as those in 4D Gaussian Splatting.

## Environmental Setups
```python
git clone https://github.com/guanxiangwenyyds/4dreconstruction
cd 4dreconstruction
conda env create -f environment.yml
conda activate 4dreconstruction

pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
In our environment, we use cuda = 11.8

## Data Preparation
Currently, the project is only compatible with two datasets.

One is the dataset used in [D-Nerf](https://github.com/albertpumarola/D-NeRF), which can be downloaded from [dropbox](https://www.dropbox.com/scl/fi/5oyd2uop62yw1ttlw1x5v/logs.zip?rlkey=5ko5sf3njkjv3vssonk1jmruy&e=1&dl=0).

另The other is the [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians?tab=readme-ov-file) dataset used in [PanopticaSport](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip)数据集.

Cloning the code will provide a subset of each dataset mentioned above.

Datasets should be organized as follows, for instance, we use the subset 'mutant' from the D-Nerf dataset and the 'basketball' subset from the Panoptic dataset.
```python
├─ data
│   |mutant
│   |basketball
│   |other dataset
```

## Experiment
Different training methods are required for different datasets.

### Panoptic Dataset
For example, the repository provides an example of the Panoptic dataset for 'basketball', and to differentiate experiments on the same dataset, you may also specify the 'identifier' parameter:
```python
python train_panoptic.py --dataset "basketball" --identifier "001"
```
After training, you can evaluate the trained models on the test set, calculating SSIM, PSNR, and L1 metrics, while also generating a series of images.
```python
python evl_panoptic.py --dataset "basketball" --identifier "001"
```
Once the above code has been executed, you should find the corresponding experiment results in the 'output/result/' directory based on your dataset parameter and identifier parameter.

### D-Nerf Dataset
The repository also provides an example of the D-Nerf Dataset "mutant", where you can also specify the 'identifier' parameter.
```python
python train_dnerf.py --dataset "mutant" --identifier "001"
```
Evaluating on the test set.
```python
python evl_dnerf.py --dataset "mutant" --identifier "001"
```
After the training is complete, you can also use the obtained model for animation synthesis, including 1.0X videos, as well as 0.5X videos synthesized through interpolation.
```python
python animation.py --dataset "mutant" --identifier "001"
```
You can also use the obtained model for mesh extraction.
```python
python mesh_extract.py --dataset "mutant" --identifier "001"
```
All results from the above experiments are saved in 'output/result/[dataset][identifier]'.

Animation synthesis and mesh extraction are only applicable to the D-Nerf dataset!

In addition, for settings on command line parameters, you can refer to the content in [arguments/__init__.py](https://github.com/guanxiangwenyyds/4dreconstruction/blob/main/arguments/__init__.py) for further experimentation.