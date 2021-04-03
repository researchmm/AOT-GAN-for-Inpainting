# AOT-GAN for High-Resolution Image Inpainting
![aotgan](https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/docs/aotgan.PNG?raw=true)
### [Arxiv Paper](https://github.com/researchmm/AOT-GAN-for-Inpainting) | 

AOT-GAN: Aggregated Contextual Transformations for High-Resolution Image Inpainting<br>
[Yanhong Zeng](https://sites.google.com/view/1900zyh),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](https://scholar.google.com/citations?user=qnbpG6gAAAAJ&hl),  and [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).<br>


<!-- ------------------------------------------------ -->
## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
```


<!-- ---------------------------------------------------- -->
## Introduction 
Despite some promising results, it remains challenging for existing image inpainting approaches to fill in large missing regions in high resolution images (e.g., 512x512). We analyze that the difﬁculties mainly drive from simultaneously inferring missing contents and synthesizing fine-grained textures for a extremely large missing region. 
We propose a GAN-based model that improves performance by,
1) **Enhancing context reasoning by AOT Block in the generator.** The AOT blocks aggregate contextual transformations with different receptive fields, allowing to capture both informative distant contexts and rich patterns of interest for context reasoning. 
2) **Enhancing texture synthesis by SoftGAN in the discriminator.**  We improve the training of the discriminator by a tailored mask-prediction task. The enhanced discriminator is optimized to distinguish the detailed appearance of real and synthesized patches, which can in turn facilitate the generator to synthesize more realistic textures.


<!-- ------------------------------------------------ -->
## Results
![face_object](https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/docs/face_object.PNG?raw=true)
![logo](https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/docs/logo.PNG?raw=true)


<!-- -------------------------------- -->
## Prerequisites 
* python 3.8.8
* [pytorch](https://pytorch.org/) (tested on Release 1.8.1)

<!-- --------------------------------- -->
## Installation 

Clone this repo.

```
git clone git@github.com:researchmm/AOT-GAN-for-Inpainting.git
cd AOT-GAN-for-Inpainting/
```

For the full set of required Python packages, we suggest create a Conda environment from the provided YAML, e.g.

```
conda env create -f environment.yml 
conda activate inpainting
```

<!-- --------------------------------- -->
## Datasets 




<!-- -------------------------------------------------------- -->
## Getting Started

1. Training:
    * Prepare training images filelist [[our split]](https://drive.google.com/open?id=1_j51UEiZluWz07qTGtJ7Pbfeyp1-aZBg)
    * Modify [celebahq.json](configs/celebahq.json) to set path to data, iterations, and other parameters.
    * Our codes are built upon distributed training with Pytorch.  
    * Run `python train.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. 
    * For example, `python train.py -c configs/celebahq.json -n pennet -m pconv -s 512 `
2. Resume training:
    * Run `python train.py -n pennet -m pconv -s 512 `.
3. Testing:
    * Run `python test.py -c [config_file] -n [model_name] -m [mask_type] -s [image_size] `. 
    * For example, `python test.py -c configs/celebahq.json -n pennet -m pconv -s 512 `
4. Evaluating:
    * Run `python eval.py -r [result_path]`

<!-- ------------------------------------------------------------------- -->
## Pretrained models
[CELEBA-HQ](https://drive.google.com/open?id=1d7JsTXxrF9vn-2abB63FQtnPJw6FpLm8) |
[Places2](https://drive.google.com/open?id=19u5qfnp42o7ojSMeJhjnqbenTKx3i2TP) 

Download the model dirs and put it under `experiments/`


<!-- ------------------------ -->
## TensorBoard
Visualization on TensorBoard for training is supported. 

Run `tensorboard --logdir [log_fold] --bind_all` and open browser to view training progress. 


### License
Licensed under an MIT license.
