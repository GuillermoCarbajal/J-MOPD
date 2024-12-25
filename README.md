# J-MOPD

Official Pytorch Implementation  of *J-MOPD*. 

## Offsets Prediction Network Architecture
<p align="center">
<img width="900" src="imgs/offsets_network.png?raw=true">
  </p>
  
## Quick Demo


* <a href="https://colab.research.google.com/github/GuillermoCarbajal/J-MOPD/blob/main/J-MOPD_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation
### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/J-MOPD.git
```


### Download deblurring models

[Offset Prediction Network Model (256)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop256.pkl)           
[Restoration Network (256)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop256_G.pkl)     
[Offset Prediction Network Model (320)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop320.pkl)              
[Restoration Network (320)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop320_G.pkl)      
[Offset Prediction Network Model (GoPro 256)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop256_GoPro.pkl)               
[Restoration Network (GoPro 256)](https://iie.fing.edu.uy/~carbajal/J-MOPD/crop256/crop256_GoPro_G.pkl)      

### Deblur an image or a list of images
```
python test_J-MOPD.py -b blurry_img_path --reblur_model reblur_model_path --nimbusr_model restoration_model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_images`: may be a singe image path or a .txt with a list of images.
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--gamma_factor`: gamma correction factor. By default is assummed `gamma_factor=2.2`. For Kohler dataset images `gamma_factor=1.0`.
  




<p align="center">
<img width="900" src="imgs/J-MOPD_example.png?raw=true">
  </p>

## Aknowledgments 
We thank the authors of [Deep Model-Based Super-Resolution with Non-Uniform Blur](https://arxiv.org/abs/2204.10109) for the Non-Blind Deconvolution Network provided in https://github.com/claroche-r/DMBSR 


Guillermo Carbajal was supported partially by Agencia Nacional de Investigacion e Innovación (ANII, Uruguay) `grant POS FCE 2018 1 1007783`. The experiments presented in this paper were carried out using ClusterUY (site: https://cluster.uy).
