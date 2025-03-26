# Ada-DF

This repository is the official implementation of the ICASSP 2023 paper *A Dual-branch Adaptive Distribution Fusion Framework for Real-world Facial Expression Recognition*.

## Abstract

![](https://raw.githubusercontent.com/taylor-xy0827/images/main/202302201825499.jpg)

Facial expression recognition (FER) plays a significant role in our daily life. However, due to the ambiguity existing in and out of FER datasets, the performance of FER is greatly hindered. To address this, we propose the double-branch framework Ada-DF. Firstly, the auxiliary branch is constructed to obtain the label distributions of samples. Secondly, the class distributions of emotions are computed through the label distributions of each emotion to replace the single labels for label distribution learning. Finally, we propose the adaptive distribution fusion module to fuse the previous two distributions according to attention weights for utilizing their pros and discarding their cons. We evaluate our method on three real-world datasets RAF-DB, AffectNet, and SFEW, which show that our Ada-DF outperforms the previous methods and achieves SOTA on all three datasets. 

## Requirements

### Dependencies

We train Ada-DF on CUDA 10.2, PyTorch 1.12.0, torchvision 0.13.0, and Python 3.9. We recommend using [Anaconda](https://www.anaconda.com/) to set up the environment:

~~~python
# create the virtual environment
conda create -n ada-df python=3.9

# activate the environment
conda activate ada-df

# install dependencies
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=10.2 -c pytorch
pip install pandas
pip install timm
pip install tensorboard
~~~

**Note: Other versions of dependencies may be compatible either.**

### Datasets

We do not provide FER datasets in our repository. Please download the datasets yourselves:

- Download the [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset and extract the `raf-basic` dir to `./datasets`. 
- Download the [AffectNet](https://mohammadmahoor.com/pages/databases/affectnet/) dataset and extract the `affectnet` dir to `./datasets`. 
- Download the [SFEW](https://cs.anu.edu.au/few/AFEW.html) dataset and extract the `sfew` dir to `./datasets`. 

Please refer to [README](https://github.com/taylor-xy0827/Ada-DF/tree/main/datasets) for detailed folder structure of `./datasets`.

**Note: Except for the RAF-DB dataset, all other datasets do not provide aligned facial images. We have aligned all facial images via [MTCNN](https://github.com/serengil/deepface).**

### Pre-trained backbone

We do not provide the pre-trained backbone in our repository. Please download the pre-trained ResNet18 from [Google Drive](https://drive.google.com/file/d/1ByvxPD9QkmWZDWtTmDQ5ta1MiAkXt22T/view?usp=sharing) and put it under `./pretrain`. 

## Training

We provide the training code for RAF-DB, AffectNet, and SFEW.

You should activate the virtual environment before training:

~~~python
conda activate ada-df
~~~

For the RAF-DB dataset, run:

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'raf' --data_path './datasets/raf-basic/' --batch_size 64 --num_classes 7 --threshold 0.7 --beta 3 --max_weight 1.0 --min_weight 0.2
~~~

For the AffectNet-7 dataset, run:

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'affectnet7' --data_path './datasets/affectnet/' --batch_size 64 --num_classes 7 --threshold 0.5 --beta 5 --max_weight 1.0 --min_weight 0.2
~~~

For the SFEW dataset, run:

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'sfew' --data_path './datasets/sfew/' --batch_size 16 --num_classes 7 --threshold 0.5 --beta 5 --max_weight 1.0 --min_weight 0.2
~~~

## Results

Our Ada-DF outperforms the previous work with 90.04%, 65.34%, and 60.46% on RAF-DB, AffectNet, and SFEW.

![image.png](https://raw.githubusercontent.com/taylor-xy0827/images/main/202303302209662.png)

## Citation

If you find our code useful, please consider citing our paper:

```shell
@inproceedings{liu2023dual,
  title={A Dual-Branch Adaptive Distribution Fusion Framework for Real-World Facial Expression Recognition},
  author={Liu, Shu and Xu, Yan and Wan, Tongming and Kui, Xiaoyan},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
