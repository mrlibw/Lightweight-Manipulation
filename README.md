# Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation
Pytorch implementation for Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation. The goal is to introduce a lightweight generative adversarial network for efficient image manipulation using natural language descriptions.

### Overview
<img src="archi.jpg" width="940px" height="230px"/>

**[Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation](https://proceedings.neurips.cc/paper/2020/file/fae0b27c451c728867a567e8c1bb4e53-Paper.pdf).**  
[Bowen Li](https://mrlibw.github.io/), [Xiaojuan Qi](https://xjqi.github.io/), [Philip H. S. Torr](http://www.robots.ox.ac.uk/~phst/), [Thomas Lukasiewicz](http://www.cs.ox.ac.uk/people/thomas.lukasiewicz/).<br> University of Oxford, University of Hong Kong <br> NeurIPS 2020 <br>

### Data

1. Download the preprocessed metadata for [bird](https://drive.google.com/file/d/1D87x3JAt0w9ymlKElh7ArpAviKUqkNbN/view?usp=sharing) and [coco](https://drive.google.com/file/d/1hNEsFDj7S0aG1tXFvJy1DXtQWSl2Z9gZ/view?usp=sharing), and save both into `data/`
2. Download [bird](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset and extract the images to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`

### Training
All code was developed and tested on CentOS 7 with Python 3.7 (Anaconda) and PyTorch 1.1.

#### [DAMSM](https://github.com/taoxugit/AttnGAN) model includes a text encoder and an image encoder
- Pre-train DAMSM model for bird dataset:
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
```
- Pre-train DAMSM model for coco dataset: 
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 1
```
#### Our Model
- Train the model for bird dataset:
```
python main.py --cfg cfg/train_bird.yml --gpu 2
```
- Train the model for coco dataset: 
```
python main.py --cfg cfg/train_coco.yml --gpu 3
```

`*.yml` files include configuration for training and testing. To reduce the number of parameters used in the model, please edit DF_DIM and/or GF_DIM values in the corresponding `*.yml` files.

#### Pretrained DAMSM Model
- [DAMSM for bird](https://drive.google.com/file/d/1n-qKR7K4V-4oVC1GaGeIHLTQfIzPsTsE/view?usp=sharing). Download and save it to `DAMSMencoders/`
- [DAMSM for coco](https://drive.google.com/file/d/1GnXhzMKtFM-RK_ATsfU1tomta1Ko72vr/view?usp=sharing). Download and save it to `DAMSMencoders/`
#### Pretrained Lightweight Model 
- [Bird](https://drive.google.com/file/d/1ojDzj4zak0-L9tG48hSfN9FxibwjsS6V/view?usp=sharing). Download and save it to `models/`

- [COCO](https://drive.google.com/file/d/1fhGtqEF2FRZNq8-wNrDUgD6paMGEC-SW/view?usp=sharing). Download and save it to `models/`

### Testing
- Test our model on bird dataset:
```
python main.py --cfg cfg/eval_bird.yml --gpu 4
```
- Test our model on coco dataset: 
```
python main.py --cfg cfg/eval_coco.yml --gpu 5
```
### Evaluation

- To generate images for all captions in the testing dataset, change B_VALIDATION to `True` in the `eval_*.yml`. 
- [Fréchet Inception Distance](https://github.com/mseitzer/pytorch-fid).

### Code Structure
- code/main.py: the entry point for training and testing.
- code/trainer.py: creates the networks, harnesses and reports the progress of training.
- code/model.py: defines the architecture.
- code/attention.py: defines the spatial and channel-wise attentions.
- code/VGGFeatureLoss.py: defines the architecture of the VGG-16.
- code/datasets.py: defines the class for loading images and captions.
- code/pretrain_DAMSM.py: trains the text and image encoders, harnesses and reports the progress of training. 
- code/miscc/losses.py: defines and computes the losses.
- code/miscc/config.py: creates the option list.
- code/miscc/utils.py: additional functions.

### Citation

If you find this useful for your research, please use the following.

```
@article{li2020lightweight,
  title={Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation},
  author={Li, Bowen and Qi, Xiaojuan and Torr, Philip and Lukasiewicz, Thomas},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

### Acknowledgements
This code borrows heavily from [ManiGAN](https://github.com/mrlibw/ManiGAN) and [ControlGAN](https://github.com/mrlibw/ControlGAN) repositories. Many thanks.
