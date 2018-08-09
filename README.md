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

<br/>

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

<br/>

### 3. Downloading pretrained model
Download [the pretrained model checkpoint](https://drive.google.com/open?id=1F8zDsrGumdPHJdrZvEvPxM2A1qUCatGJ) to test the model as 10 cross-fold validation
```bash
$ unzip ckplus -d fer/ckplus
```

<br/>

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

<br/>

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
### CK+ Dataset
#### Accuracy
  
| fold    | 0     | 1      | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | mean    |
| :-----: |:-----:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| acc(%)  | 94.29 |  100   | 97.14 | 100   |  100  |  100  |  100  | 90.63 | 86.67 |  100  |  96.87  |

</br>

#### Confusion matrix
  
| True\Pred| Anger | Contempt|Disgust| Fear | Happy| Sadness| Surprise |
| :-----: |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Anger | 97.78|    |    |     |     |     | 2.22 |     | 
| Contempt |      |  100   |     |      |     |     |     |     | 
| Disgust  |     |     |100 |     |     |     |     |    | 
| Fear |     |        | 4.00 | 88.00 |      | 4.00  | 4.00 | 
| Happy  |      |  | 1.45 | | 98.55|  |    | | 
| Sadness  |     |  3.57  |  |  ||  92.86  | 3.57 | 
| Surprise  |      |  1.20   |  |  |  1.20  | 1.20  |  96.39 |

<br/>

#### Qualitative results
<img width="70%" src="png/sample.png"/>

<br/>

### Oulu-Casia Dataset
#### Accuracy
  
| fold    | 0     | 1      | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | mean     |
| :-----: |:-----:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| acc(%)  | 87.50 |  79.16  | 75.00 | 89.58  |  85.42  |  81.25  |  83.33  | 83.33 | 81.25 |  81.25  | 82.67 |

</br>

#### Confusion matrix
  
| True\Pred| Anger | Contempt|Disgust| Fear | Happy| Sadness| Surprise |
| :-----: |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Anger | 97.78|    |    |     |     |     | 2.22 |     | 
| Contempt |      |  100   |     |      |     |     |     |     | 
| Disgust  |     |     |100 |     |     |     |     |    | 
| Fear |     |        | 4.00 | 88.00 |      | 4.00  | 4.00 | 
| Happy  |      |  | 1.45 | | 98.55|  |    | | 
| Sadness  |     |  3.57  |  |  ||  92.86  | 3.57 | 
| Surprise  |      |  1.20   |  |  |  1.20  | 1.20  |  96.39 |

<br/>

#### Qualitative results
<img width="70%" src="png/sample.png"/>

<br/>

## Reference
* [[DTAN] Joint fine-tuning in deep neural networks for facial expression recognition](https://ieeexplore.ieee.org/document/7410698/) <br/> JUNG, Heechul, et al. In: Proceedings of the IEEE International Conference on Computer Vision. 2015. p. 2983-2991.<br/>
* [Face alignment code](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)

<br/>
