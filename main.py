import os
import argparse
import torch
from torch.backends import cudnn
from dataloader import get_loader
from solver import Solver

def str2bool(v):
    return v.lower() in ('true')

def str2list(v):
    return [int(x) for x in v.split(',')]

def main(config):
  cudnn.benchmark = True
  model_path = os.path.join(config.main_path, 'models', config.model_name)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if not os.path.exists(os.path.join(model_path, 'results')):
    os.makedirs(os.path.join(model_path, 'results'))
  if not os.path.exists(os.path.join(model_path, 'checkpoint')):
    os.makedirs(os.path.join(model_path, 'checkpoint'))

  trainloader, validloader, num_data = get_loader(config)

  solver = Solver(trainloader, validloader, config, num_data)

  if config.mode == 'train':
    solver.train()
  elif config.mode == 'valid':
    solver.valid()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--main_path', type=str, default='/data/hylee-fer', help='main path to save model and results')
  parser.add_argument('--emotion_dir', type=str, default='/data/Emotion', help='emotion data directory.')
  parser.add_argument('--image_dir', type=str, default='/data/cohn-kanade-images_processed', help='image data directory')
  parser.add_argument('--batch_size',type=int, default=32, help='number of sequences to train on in parallel')
  parser.add_argument('--crop_size',type=int, default=64)
  parser.add_argument('--image_size',type=int, default=64)
  parser.add_argument('--num_workers', type=int, default=1)
  parser.add_argument('--model_name', type=str, default='fer1',help='model name')
  parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--max_epoch', type=int, default=300, help='iterations')
  parser.add_argument('--log_step', type=int, default=5, help='log step')
  parser.add_argument('--acc_step', type=int, default=1, help='accuracy print step')
  parser.add_argument('--draw_step', type=int, default=1, help='draw step')
  parser.add_argument('--lr_update_step', type=str2list, default='700')
  parser.add_argument('--lr_decay', type=int, default=10)
  parser.add_argument('--cls', type=int, default=7)
  parser.add_argument('--kfold', type=int, default=10)
  parser.add_argument('--ithfold', type=int, default=0)
  parser.add_argument('--mode', type=str, default='train', help='train|valid')
  parser.add_argument('--dataset_name', type=str, default='ckplus', help='ckplus|oulu')
  parser.add_argument('--degree', type=int, default=20)
  parser.add_argument('--restore', type=str2bool, default='False')
  parser.add_argument('--model_save_from',type=int, default=1,help='step size of iterations to save checkpoint(model)')
  parser.add_argument('--use_visdom', type=str2bool, default='True')
  parser.add_argument('--momentum',type=float, default=0.9,help='momentum')
  parser.add_argument('--weight_decay',type=float, default=0.0001,help='weight_decay')
  parser.add_argument('--nest',type=str2bool, default=False,help='weight_decay')
  config = parser.parse_args()
  print(config)

  main(config)
