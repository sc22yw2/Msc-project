import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os
from tqdm import tqdm
import wandb

from utils import utils


class Trainer():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, train_data_loader, \
                valid_data_loader=None, metric=None, start_epoch=0, num_epochs=25, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.metric = metric
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.is_debug = is_debug

        self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.logger = logger
        self.writer = writer
        self.wandb_dict = {}
    def fit(self):
        for epoch in range(0, self.start_epoch):
            self.lr_schedule.step()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.append('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.append('-' * 60)
            self.cur_epoch = epoch
            self.lr_schedule.step()
            if self.is_debug:
                self._dump_infos()
            self._train()
            self._valid()
            self._save_best_model()
            print()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('best loss: %f' % (self.best_loss))
        self.logger.append('------------------------------------------------------------')

    def _train(self):
        self.model.train()  # Set model to training mode
        losses = []

        if self.metric is not None:
            self.metric[0].reset()
            self.metric[1].reset()

        for i, (inputs, age,sex) in enumerate(tqdm(self.train_data_loader)):              # Notice

            if self.is_use_cuda:
                inputs, age,sex = inputs.cuda(), age.cuda(),sex.cuda()
                age = age.squeeze()
                sex = sex.squeeze()

            else:
                age = age.squeeze()
                sex = sex.squeeze()

            self.optimizer.zero_grad()
            pred_sex,pred_age = self.model(inputs)            # Notice
            loss = self.loss_fn[0](pred_sex, sex)
            loss += self.loss_fn[1](pred_age,age)
            if self.metric is not None:
                prob_sex     = F.softmax(pred_sex, dim=1).data.cpu()
                predicted_probability, predicted = torch.max(pred_age, 1)
                self.metric[0].add(prob_sex, sex.data.cpu())
                self.metric[1].add(predicted.data.cpu(),age.data.cpu())

            loss.backward()


            self.optimizer.step()


            losses.append(loss.item())       # Notice
            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                batch_mean_loss  = np.mean(losses)
                print_str = '[%s]\tTraining Batch[%d/%d]\t  Loss: %.4f\t\n'           \
                            % (local_time_str, i, len(self.train_data_loader) - 1, batch_mean_loss)
                if i == len(self.train_data_loader) - 1 and self.metric is not None:
                    top1_acc_score = self.metric[0].value()[0]
                    age_mean_score = self.metric[1].value().item()
                    print_str += '@Top-1 Score: %.4f\t\n' % (top1_acc_score)
                    print_str += '@age_mean Score: %.4f\t' % (age_mean_score)

                self.logger.append(print_str)
        self.wandb_dict['loss'] = batch_mean_loss
        self.wandb_dict['top1_acc_score'] = top1_acc_score
        self.wandb_dict['age_mse'] = age_mean_score
        wandb.log(utils.add_prefix(self.wandb_dict, f'train'), step=self.cur_epoch, commit=False)

        # self.writer.add_scalar('loss/loss_c', batch_mean_loss, self.cur_epoch)

    def _valid(self):
        self.model.eval()
        losses = []
        acc_rate = 0.
        if self.metric is not None:
            self.metric[0].reset()
            self.metric[1].reset()

        with torch.no_grad():              # Notice
            for i, (inputs, age,sex) in enumerate(tqdm(self.valid_data_loader)):

                if self.is_use_cuda:
                    inputs, age, sex = inputs.cuda(), age.cuda(), sex.cuda()
                    age = age.squeeze()
                    sex = sex.squeeze()

                else:
                    age = age.squeeze()
                    sex = sex.squeeze()

                self.optimizer.zero_grad()
                pred_sex, pred_age = self.model(inputs)  # Notice
                loss = self.loss_fn[0](pred_sex, sex)
                loss += self.loss_fn[1](pred_age, age) / 100
                if self.metric is not None:
                    prob_sex = F.softmax(pred_sex, dim=1).data.cpu()
                    predicted_probability, predicted = torch.max(pred_age, 1)
                    self.metric[0].add(prob_sex, sex.data.cpu())
                    self.metric[1].add(predicted.data.cpu(), age.data.cpu())

                losses.append(loss.item())
            
        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #self.logger.append(losses)
        batch_mean_loss = np.mean(losses)
        print_str = '[%s]\tValidation: \t  Loss: %.4f\t\n'     \
                    % (local_time_str, batch_mean_loss)
        if  self.metric is not None:
            top1_acc_score = self.metric[0].value()[0]
            age_mean_score = self.metric[1].value().item()
            print_str += '@Top-1 Score: %.4f\t\n' % (top1_acc_score)
            print_str += '@age_mean Score: %.4f\t' % (age_mean_score)
        self.logger.append(print_str)
        if top1_acc_score >= self.best_acc:
            self.best_acc = top1_acc_score
            self.best_loss = batch_mean_loss
        self.wandb_dict['loss'] = batch_mean_loss
        self.wandb_dict['top1_acc_score'] = top1_acc_score
        self.wandb_dict['age_mse'] = age_mean_score
        wandb.log(utils.add_prefix(self.wandb_dict, f'val'), step=self.cur_epoch, commit=True)

    def _save_best_model(self):
        # Save Model
        self.logger.append('Saving Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch,
            'num_epochs': self.num_epochs
        }
        if not os.path.isdir('./checkpoint/' + self.model_type):
            os.makedirs('./checkpoint/' + self.model_type)
        torch.save(state, './checkpoint/' + self.model_type + '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')   # Notice