## Face age classification
PyTorch implementation of [DTGN[ICCV2015]](https://ieeexplore.ieee.org/document/7410698/) to recognize face expression on dataset CK+

<br/>

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)

<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/HayeonLee/face-expression-classification.git
$ cd face-expression-classification/
```

### 2. Downloading & Preprocessing the dataset
(1) Download [Cohn-Kanade (CK+) dataset](http://www.consortium.ri.cmu.edu/ckagree/) <br/>
(2) Move *cohn-kanade-images* directory under *face-expression-classification/data/* <br/>
(3) Perform preprocessing to crop and align images
```bash
$ python preprocessing/face_alignment.py
```
*cohn-kanade-images_processed* directory will be generated under *data* folder

### 3. Downloading pretrained model
Download [the pretrained model checkpoint](https://drive.google.com/open?id=1F8zDsrGumdPHJdrZvEvPxM2A1qUCatGJ) to test the model as 10 cross-fold validation
```bash
$ unzip ckplus -d face-expression-classification/data
```

### 4. Testing
Test your own model </br>

Test pretrained model using chekpoints
```bash
$ python main.py --mode valid --image_dir cacd2000_224 --crop_size 224 --image_size 224 \
                 --gpu_ids 0 --batch_size 64 --pretrain False --log_dir cacd2000_cls/logs \
                 --model_save_dir cacd2000_cls/models --result_dir cacd2000_cls/results \
                 --resume_iters 104000
```

<br/>

### 5. Training
```bash
$ python main.py --mode valid --image_dir cacd2000_224 --crop_size 224 --image_size 224 \
                 --gpu_ids 0 --batch_size 64 --pretrain True --log_dir cacd2000_cls/logs \
                 --model_save_dir cacd2000_cls/models --result_dir cacd2000_cls/results
```



## Results

<p align="center"><img width="70%" src="png/125.png"/></p>

<p align="center"><img width="70%" src="png/107.png"/></p>


<br/>

## Reference
I refer code and readme template of [StarGAN](https://github.com/yunjey/StarGAN.git)<br/>
[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) <br/>
Yunjey Choi et al. IEEE Conference on Computer Vision and Pattern Recognition, 2018

<br/>
