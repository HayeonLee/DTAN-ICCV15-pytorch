import os
import random
from random import randint
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from utils import get_data_list

class CKplus(data.Dataset):
    def __init__(self, data_list, transform, mode):
        '''
        1. Under Emotion directory, read all emotion file name
            ex) Emotion/S005/001/S005_001_00000011_emotion.txt
        2. Under extended-cohn-kanade-images, read all image file name corresponding to files of 1.
            ex) cohn-kanade-images/S005/001/S005_001_00000011.png
        '''
        self.transform = transform
        self.dataset = data_list
        self.mode = mode
        random.seed(1234)
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        label, img_dirname = self.dataset[index]
        filenames = sorted(os.listdir(img_dirname))
        degree = np.random.randint(-20, 20)
        seed = np.random.randint(2147483647) # make a seed with numpy generator

        imgs = self._stack_frames(0, degree, img_dirname, filenames, seed)
        img = self._stack_frames(1, degree, img_dirname, filenames, seed)
        imgs = torch.cat((imgs, img), 0)
        img = self._stack_frames(2, degree, img_dirname, filenames, seed)

        imgs = torch.cat((imgs, img), 0)
        label = torch.LongTensor([label])
        return imgs, label

    def __len__(self):
        return len(self.dataset)

    def _stack_frames(self, nf, degree, img_dirname, filenames, seed):
        img = Image.open(os.path.join(img_dirname, filenames[nf])).convert('L')
        if self.mode == 'train':
            img = TF.rotate(img, degree)
        random.seed(seed)
        img = self.transform(img)
        return img

def get_loader(config):
    train_list, valid_list, num_data = get_data_list(config.emotion_dir,
                                                     config.image_dir,
                                                     config.cls,
                                                     config.kfold,
                                                     config.ithfold)
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(config.image_size))
    transform.append(T.RandomCrop(config.crop_size, 4))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    transform = T.Compose(transform)

    transform_valid = []
    transform_valid.append(T.Resize(config.image_size))
    transform_valid.append(T.CenterCrop(config.crop_size))
    transform_valid.append(T.ToTensor())
    transform_valid.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    transform_valid = T.Compose(transform_valid)

    if config.dataset_name == 'ckplus':
        print('CKplus dataset for train and validation are created...')
        train_dataset = CKplus(train_list, transform, config.mode)
        valid_dataset = CKplus(valid_list, transform_valid, 'valid')
    elif config.dataset_name == 'oulu':
        # train_dataset = Oulu(train_list, transform, config.video_normal)
        # valid_dataset = Oulu(valid_list, transform_valid, config.video_normal)
        print('Oulu dataset for train and validation are created...')

    if config.mode == 'train':
        print('The number of train_dataset(before augmentation): {} '.format(len(train_dataset)))
        print('The number of valid_dataset: {}'.format(len(valid_dataset)))
        trainloader = data.DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=config.num_workers)
        validloader = data.DataLoader(dataset=valid_dataset,
                                      batch_size=len(valid_dataset),
                                      shuffle=False,
                                      num_workers=config.num_workers)
        return trainloader, validloader, num_data
    if config.mode == 'valid':
        print('The number of valid_dataset: {}'.format(len(valid_dataset)))
        validloader = data.DataLoader(dataset=valid_dataset,
                                      batch_size=len(valid_dataset),
                                      shuffle=False,
                                      num_workers=config.num_workers)
        return None, validloader, num_data
