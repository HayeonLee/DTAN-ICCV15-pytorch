## Face age classification
PyTorch implementation of [DTGN(ICCV2015)](https://ieeexplore.ieee.org/document/7410698/) to recognize face expression on dataset CK+

<br/>

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [OpenCV](https://opencv.org/)

<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/HayeonLee/fer.git
$ cd fer
```

### 2. Downloading & Preprocessing the dataset
(1) Download [Cohn-Kanade (CK+) dataset](http://www.consortium.ri.cmu.edu/ckagree/) <br/>
(2) Move *cohn-kanade-images* directory to *fer/data/cohn-kanade-images* <br/>
```bash
$ mv cohn-kanade-images fer/data/cohn-kanade-images
```
(3) Move *Emotion* directory to *fer/data/Emotion* <br/>
```bash
$ mv Emotion fer/data/Emotion
```
(4) Perform preprocessing to crop and align images
```bash
$ python preprocessing/face_alignment.py
```
*ck_align* directory will be generated under *data* folder

### 3. Downloading pretrained model
Download [the pretrained model checkpoint](https://drive.google.com/open?id=1F8zDsrGumdPHJdrZvEvPxM2A1qUCatGJ) to test the model as 10 cross-fold validation
```bash
$ unzip ckplus -d fer/ckplus
```

### 4. Testing
```bash
$ python main.py --mode valid --main_path fer --image_dir fer/ckplus/data/ck_align \
                 --emotion_dir fer/ckplus/data/Emotion \
                 --model_name Nthfold --ithfold N\ 
                 --crop_size 64 --image_size 64 --batch_size 32 \
                 --restore true --use_visdom False
```
N: 0~9
result images will be saved under *fer/ckplus/models/Nthfold/results*

### 5. Training
```bash
$ python main.py --mode train --main_path fer --image_dir fer/ckplus/data/ck_align \
                 --emotion_dir fer/ckplus/data/Emotion \
                 --model_name MODEL_NAME --ithfold N\ 
                 --crop_size 64 --image_size 64 --batch_size 32 \
                 --use_visdom True
```

<br/>

## Results

<p align="center"><img width="70%" src="png/125.png"/></p>

<p align="center"><img width="70%" src="png/107.png"/></p>


<br/>

## Reference
* [[DTAN] Joint fine-tuning in deep neural networks for facial expression recognition](https://ieeexplore.ieee.org/document/7410698/) <br/> JUNG, Heechul, et al. In: Proceedings of the IEEE International Conference on Computer Vision. 2015. p. 2983-2991.<br/>
* [Face alignment code](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)

<br/>
