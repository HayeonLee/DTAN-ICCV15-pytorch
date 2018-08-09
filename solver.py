import torch
import torch.optim as optim
import torch.nn as nn
from models import DTAN
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from visdom import Visdom

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


class Solver(object):
    def __init__(self, trainloader, validloader, config, num_data):
        self.trainloader = trainloader
        self.validloader = validloader
        self.max_epoch = config.max_epoch
        self.lr = config.lr
        self.log_step = config.log_step
        self.acc_step = config.acc_step
        self.draw_step = config.draw_step
        self.lr_update_step = config.lr_update_step
        self.lr_decay = config.lr_decay
        self.cls = config.cls
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.model_name = config.model_name
        self.ithfold = config.ithfold
        self.restore= config.restore
        self.model_save_from = config.model_save_from
        self.num_data = num_data
        self.use_visdom = config.use_visdom
        self.model_path = os.path.join(config.main_path, 'models', config.model_name)
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.nest = config.nest
        self.mode = config.mode
        if self.cls == 7:
            self.dict = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        else:
            self.dict = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()

        if self.use_visdom:
            self.viz = Visdom()
            self.loss_plot = self.viz.line(Y=torch.Tensor([0.]),
                                           X=torch.Tensor([0.]),
                                           opts = dict(title = 'Loss for ' + self.model_name,
                                                        xlabel='epoch',
                                                        xtickmin=0,
                                                        xtickmax=self.max_epoch,
                                                        ylabel='Loss',
                                                        ytickmin=0,
                                                        ytickmax=5,
                                                        ),)
            self.acc_plot = self.viz.line(Y=torch.Tensor([0.]),
                                          X=torch.Tensor([0.]),
                                          opts = dict(title = 'Test accuracy for ' + self.model_name,
                                                      xlabel= 'epoch',
                                                      xtickmin=0,
                                                      xtickmax=self.max_epoch,
                                                      ylabel='Accuracy',
                                                      ytickmin=0,
                                                      ytickmax=100,),)

    def build_model(self):
        # Restore model or
        # Create model
        self.model = DTAN(self.image_size, in_ch=3, out_ch=64, num_block=2, num_cls=self.cls)
        # Create optimizer
        self.optimizer = optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nest) #momentum=0.9, weight_decay=0.0001
        # Create loss function
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.num_data).to(self.device))

        # Restore model..
        if self.restore:
            self.restore_model()

        # Set GPU mode
        self.model.to(self.device)

    def update_lr(self):
        self.lr = self.lr / self.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    # def label2onehot(self, labels, dim):
    #     """Convert label indices to one-hot vectors."""
    #     batch_size = labels.size(0)
    #     out = torch.zeros(batch_size, dim)
    #     out[np.arange(batch_size), labels.long()] = 1
    #     return out.type(torch.LongTensor)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0,1)

    def restore_model(self):
        print('Loading the trained models from the checkpoint')
        model_path = os.path.join(self.model_path, 'checkpoint', '{}th_fold.ckpt'.format(self.ithfold))
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def save_model(self):
        save_path = os.path.join(
            self.model_path,
            'checkpoint',
            '{}th_fold.ckpt'.format(self.ithfold))
        torch.save(self.model.state_dict(), save_path)
        print('Save model checkpoints into {}...'.format(os.path.join(self.model_path, 'checkpoint')))

    def save_images(self, images, labels, output_indice, ith, k, best_valid_acc):
        plt.clf()
        fig, ax = plt.subplots(6, 6)
        nrow = int(images.size(0) / 6)
        for i in range(nrow):
            for j in range(6):
                ax[i][j].imshow(np.transpose(self.denorm(images[6*i + j]).numpy(), (1,2,0)))
                ax[i][j].axis('off')
                ax[i][j].set_title('T:{}  P:{}'.format(self.dict[labels[6*i + j]], self.dict[output_indice[6*i + j]]))
                ax[i][j].axis('off')
        for j in range(images.size(0) % 6):
            ax[nrow][j].imshow(np.transpose(self.denorm(images[6*nrow + j]).numpy(), (1,2,0)))
            ax[nrow][j].axis('off')
            ax[nrow][j].set_title('T:{}  P:{}'.format(self.dict[labels[6*nrow + j]], self.dict[output_indice[6*nrow + j]]))
            ax[nrow][j].axis('off')
        # if mode == 'train':
        # plt.savefig(os.path.join(self.model_path, 'results', '{:6d}-{}-fold-{:d}-acc-{:d}'.format(ith, self.ithfold, k, best_valid_acc)))
        # else:
        plt.savefig(os.path.join(self.model_path, 'results', '{:6d}-{}-fold-{:d}-acc{:d}'.format(ith, self.ithfold, k, int(best_valid_acc.item()))))
        plt.close()
        print('save {} th image... in {}'.format(k, os.path.join(self.model_path, 'results')))


    # def set_ax(self, ax, i, j, img, label, pred):
    #     ax[i][j].imshow(np.transpose(self.denorm(img).numpy(), (1,2,0)))
    #     ax[i][j].axis('off')
    #     ax[i][j].set_title('T:{}  P:{}'.format(self.dict[label], self.dict[pred]))
    #     ax[i][j].axis('off')
    #     # return ax

    def save_results(self, images, labels, output_indice, ith, mode='train', best_valid_acc=0):
        qu = int(images.size(0) / 36)
        for k in range(qu):
            self.save_images(images[k*36:(k+1)*36], labels[k*36:(k+1)*36], output_indice[k*36:(k+1)*36], ith, k, best_valid_acc)
        if images.size(0) % 36:
            self.save_images(images[qu*36:], labels[qu*36:], output_indice[qu*36:], ith, qu, best_valid_acc)

    def train(self):

        data_iter = iter(self.trainloader)
        iters = len(data_iter)
        total = 0
        correct = 0
        best_acc = 0
        best_valid_acc = 0
        mean_loss = 0

        for i in range(self.max_epoch):
            for j in range(iters):
                try:
                    img_seqs, label = next(data_iter)
                except:
                    data_iter = iter(self.trainloader)
                    img_seqs, label = next(data_iter)

                img_seqs = img_seqs.to(self.device)
                label = label[:,0].to(self.device)

                # Forward pass
                predict = self.model(img_seqs)
                # Compute loss
                loss = self.criterion(predict, label)
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # For train acc.
                _, output_index = torch.max(predict, 1)
                total += label.size(0)
                correct += (output_index == label).sum().float()
                mean_loss += loss
                # Print logs
                if (j+1) % self.log_step == 0:
                    print('iteration:[{}/{}], epoch:[{}], loss:[{:.4f}], lr:[{}], best train acc:[{:.4f}], best valid acc:[{:.4f}]'.format(
                           i * iters +  (j+1)*self.batch_size, self.max_epoch * self.batch_size * iters , i, loss, self.lr, best_acc, best_valid_acc))
                    # self.save_results(img_seqs.cpu(), label.cpu(), output_index.cpu(), j)

            if (i+1) % self.acc_step == 0:
                valid_acc, valid_loss = self.valid(i, best_valid_acc)
                train_acc = 100 * (correct/total)
                mean_loss = mean_loss / iters
                print('train acc: {:.4f}, valid acc: {:.4f}, mean loss: {:.4f}, valid loss: {:.4f}'.format(train_acc, valid_acc, mean_loss, valid_loss))
                correct = 0
                total = 0
                mean_loss = 0
                if best_acc < train_acc:
                    best_acc = train_acc
                if best_valid_acc < valid_acc:
                    if (i+1) > self.model_save_from:
                        self.save_model()
                    best_valid_acc = valid_acc
            # Draw lines
            if self.use_visdom and i % self.draw_step == 0:
                self.viz.line(Y=torch.Tensor([train_acc]),
                              X=torch.Tensor([i+1]),
                              win=self.acc_plot,
                              name='train',
                              update='append',)
                self.viz.line(Y=torch.Tensor([valid_acc]),
                              X=torch.Tensor([i+1]),
                              win=self.acc_plot,
                              name='test',
                              update='append',)
                self.viz.line(Y=torch.Tensor([mean_loss]),
                              X=torch.Tensor([i+1]),
                              win=self.loss_plot,
                              name='train loss',
                              update='append',)
                self.viz.line(Y=torch.Tensor([valid_loss]),
                              X=torch.Tensor([i+1]),
                              win=self.loss_plot,
                              name='test loss',
                              update='append',)
            # Decay learning rates.
            if (i+1) in self.lr_update_step:
                self.update_lr()
                print ('    Decayed learning rates, lr: {}'.format(self.lr))

    def valid(self, i=0, best_valid_acc=0):
        data_iter = iter(self.validloader)
        total = 0
        correct = 0
        mean_loss = 0
        self.model.eval()
        for img_seqs, label in data_iter:
            # with torch.no_grad():
            img_seqs = img_seqs.to(self.device)
            label = label[:,0].to(self.device)
            predict = self.model(img_seqs)
            loss = self.criterion(predict, label)
            _, output_index = torch.max(predict, 1)
            total += label.size(0)
            correct += (output_index == label).sum().float()
            mean_loss += loss
                # if best_valid_acc:
                #     self.save_results(img_seqs.cpu(), label.cpu(), output_index.cpu(), i, 'test', best_valid_acc)
        valid_acc = correct / total * 100
        if best_valid_acc < valid_acc:
            self.save_results(img_seqs.cpu(), label.cpu(), output_index.cpu(), i, 'valid', valid_acc)
        mean_loss = mean_loss / len(data_iter)
        if self.mode == 'valid':
            print('{}th fold accuarcy: {}'.format(self.ithfold, valid_acc))
        return valid_acc, mean_loss

