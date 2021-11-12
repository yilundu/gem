# Learning Signal-Agnostic Manifolds of Neural Fields

This is the uncleaned code for the paper Learning Signal-Agnostic Manifolds of Neural Fields. The
code will be cleaned shortly.


## Downloading Data

Please utilize the following link to download the underlying models and data used in the paper.


## Demo

The underying audiovisual manifold illustrated in the paper may be constructed by utilizing the following command

```
python experiment_scripts/audiovisual_manifold_interpolate.py --experiment_name=audiovis_demo --checkpoint_path log_root/audiovis_demo/checkpoints/model_70000.pth
```

## Training Different Signal Manifolds

Please utilize the following command to train an image manifold

```
python experiment_scripts/train_autodecoder_multiscale.py --experiment_name=celeba 
```

Please utilize the following command to train a 3D shape manifold

```
python experiment_scripts/train_imnet_autodecoder.py --experiment_name=imnet 
```

Please utilize the following command to train an audio manifold

```
python experiment_scripts/train_audio_autodecoder.py --experiment_name=audio 
```

Please utilize the following command to train an audiovisual manifold

```
python experiment_scripts/train_audiovisual_autodecoder.py --experiment_name=audiovisual
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
