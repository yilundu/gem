# Learning Signal-Agnostic Manifolds of Neural Fields

This is the uncleaned code for the paper [Learning Signal-Agnostic Manifolds of Neural Fields](https://arxiv.org/abs/2111.06387). The cleaned code will be cleaned shortly.

## Setup Instructions

Please install packages in the list requirements.txt file:

```
pip install -r requirements.txt
```

Next to install the datasets/models used in the paper, download the tar.gz file in the following [link](https://www.dropbox.com/s/0yi8nqtxrssljdj/gem_data_release_2.tar.gz?dl=0) and extract it in the root directory of the repo. Please download
and extract the all\_vox256\_img\* hdf5 files for ShapeNet from [here](https://github.com/czq142857/IM-NET).


## Demo
## Training Different Signal Manifolds

Please utilize the following command to train an image manifold with the linear_lle loss

```
python experiment_scripts/train_autodecoder_multiscale.py --experiment_name=celeba --type linear_lle
```

Utilize the following command  to train with just mse

```
python experiment_scripts/train_autodecoder_multiscale.py --experiment_name=celeba --type none
```

## Citing our Paper

If you find our code useful for your research, please consider citing 

``` 
@inproceedings{du2021gem,
  title={Learing Signal-Agnostic Manifolds of Neural Fields},
  author={Du, Yilun and Collins, M. Katherine and and Tenenbaum, B. Joshua
  and Sitzmann, Vincent},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
