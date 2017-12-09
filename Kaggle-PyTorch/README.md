
# PyTorch Model Ensembler + Convolutional Neural Networks (CNN's)

![curve](logo.png)


# Introduction
Here, we investigate the effect of PyTorch model ensembles by combining the top-N single models crafted during the training phase. 
The results demonstrate that model ensembles may significantly outperform conventional single model approaches. Moreover, 
the method constructs an ensemble of deep CNN models with different architectures that are complementary to each other.

## Ensemble learning: 
Ensemble learning is a technique of using several models or for solving a particular classification problem. Ensemble methods seek to **promote diversity** among the models they combine and reduce the problem related to overfitting of the training data-sets.
The outputs of the individual models of the ensemble are combined (e.g. by averaging) to form the final prediction

**During inference, the responses of the individual ConvNets of the ensemble are averaged to form the final classification.**
Both of these are **well studied techniques in the machine learning community** and relate to model averaging and **over-fitting prevention.** 

If you want to investigate image classification by ensembling models, this is a repository that will help you out in doing so. 
It shows how to perform **CNN ensembling** in PyTorch with publicly available data sets. It is based on many hours of debugging and a bunch of of official pytorch tutorials/examples. 

I felt that it was not exactly super trivial to perform ensembling in PyTorch, and so I thought I'd release my code as a tutorial which I wrote originally for my Kaggle.

I Highly encourage you to run this on a **existing data sets** (read main-binary.py to know which format to store your data in), but for a sample dataset to start with, you can download a simple 2 class dataset from here - https://download.pytorch.org/tutorial/hymenoptera_data.zip

All Torch and PyTorch specific details have been explained in detail in the file main-binary.py.

Hope this tutorial helps you out! :)

Relevant Kaggle post: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/44849
    
![curve](curve.png)

# Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org) version **2 and up**
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

## Setup and Installation

Guides for downloading and installing PyTorch using Docker can be found [here](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/tree/master/docker).

## Python Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)

## Progress

- [x] Binary Classification on the Statoil Data Set
- [x] SENet
- [x] Curve generation, ACC/LOSS in PNG format
- [x] Multi-Class Classification on the Seedlings Data set  
- [ ] Multi-Class Classification on the TF Audio commands Data set  

### Networks Used (See *models* folder for details)
- [x] SENet - [Squeeze-and-Excitation Networks]  
- [x] Wide ResNet ([paper](https://arxiv.org/abs/1605.07146)) ([code](https://github.com/szagoruyko/wide-residual-networks))
- [x] DenseNet ([paper](https://arxiv.org/abs/1608.06993)) ([code](https://github.com/liuzhuang13/DenseNet)) see [Andreas Veit's implementation](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py) or [Brandon Amos's implementation](https://github.com/bamos/densenet.pytorch/blob/master/densenet.py) )
- [ ] AlexNet ([paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks))
- [ ] VGGNet ([paper](https://arxiv.org/abs/1409.1556))
- [ ] SqueezeNet ([paper](https://arxiv.org/abs/1602.07360)) ([code](https://github.com/DeepScale/SqueezeNet))
- [x] ResNet ([paper](https://arxiv.org/abs/1512.03385)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] Pre-ResNet ([paper](https://arxiv.org/abs/1603.05027)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] Pre-ResNet with Stochastic Depth
- [ ] PyramidalNet ([paper](https://arxiv.org/abs/1610.02915))([code](https://github.com/jhkim89/PyramidNet))
- [ ] PyramidalNet with Separated Stochastic Depth ([paper](https://arxiv.org/abs/1612.01230))([code](https://github.com/AkTgWrNsKnKPP/PyramidNet_with_Stochastic_Depth))
- [ ] ResNeXt ([paper](https://arxiv.org/abs/1611.05431)) ([code](https://github.com/facebookresearch/ResNeXt))
- [ ] MSDNet ([paper](https://arxiv.org/abs/1703.09844)) ([code](https://github.com/gaohuang/MSDNet))
- [ ] Steerable CNN ([paper](https://arxiv.org/abs/1612.08498))

#### Upcoming 

* [MaskRCNN](https://arxiv.org/abs/1703.06870) ?

### DataLoaders implemented

* [Statoil Satelites classification]()
* [Seedlings Classification]()
* [TF Audio commands classification]()


# Material
The material consists of several competitions and data sets.

## Data Sets in PyTorch 
Enter the absolute path of the dataset folder below. Keep in mind that this code expects data to be in same format as Imagenet. I encourage you to use your own dataset. In that case you need to organize your data such that your dataset folder has EXACTLY two folders. Name these 'train' and 'val'

**The 'train' folder contains training set and 'val' fodler contains validation set on which accuracy is measured.**  

The structure within 'train' and 'val' folders will be the same. They both contain **one folder per class**. All the images of that class are inside the folder named by class name; this is crucial in PyTorch. 

### More on Data Sets
If your dataset has 2 classes like in the Kaggle Statoil set, and you're trying to classify between pictures of 1) ships 2) Icebergs, 
say you name your dataset folder 'data_directory'. Then inside 'data_directory' will be 'train' and 'test'. Further, Inside 'train' will be 2 folders - 'ships', 'icebergs'. 

## So, the structure looks like this: 

![curve](dataset.png)

```
|-  data_dir
       |- train 
             |- ships
                  |- ship_image_1
                  |- ship_image_2
                         .....

             |- ice
                  |- ice_image_1
                  |- ice_image_1
                         .....
       |- val
             |- ships
             |- ice
```

For a full example refer to: https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/kdataset/seedings.py 


## [Competition 1 -  Statoil/C-CORE Iceberg Classifier Challenge]( https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)
![statoil](statoil.png)


### Single model Log loss 

| network               | dropout | preprocess | GPU       | params  | training time | Loss   |
|:----------------------|:-------:|:----------:|:---------:|:-------:|:-------------:|:------:|
| Lecun-Network         |    -    |   meanstd  | GTX1080  |          |         |        |
| Residual-Network50    |    -    |   meanstd  | GTX1080  |          |    |        |
| DenseNet-100x12       |    -    |   meanstd  | GTX1080  |          |    |        |
| ResNeXt-4x64d         |    -    |   meanstd  | GTX1080  |          |    |        |
| SENet(ResNeXt-4x64d)  |    -    |   meanstd  | GTX1080  |          |  -            |   -    |


### 50 models **ensemble** Log loss 

![curve](pytorch-ensembler.png)


### Usage

Launch [visdom](https://github.com/facebookresearch/visdom#launch) by running (in a separate terminal window)

```
python -m visdom.server
```

**To train the model :**

```
```

**To validate the model :**

```
```

**To test the model w.r.t. a dataset on custom images(s):**

```
```


## About ResNeXt & DenseNet

https://github.com/liuzhuang13/DenseNet
https://github.com/prlz77/ResNeXt.pytorch
https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#computer-vision
https://github.com/zhunzhong07/Random-Erasing
https://github.com/lim0606/pytorch-geometric-gan
  
## Contributors   
Credits: Shlomo Kashani and many others. 

[Shlomo Kashani](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/) 
