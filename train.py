#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: caixia_dong
@license: (C) Copyright 2020-2023, Medical Artificial Intelligence, XJTU.
@contact: caixia_dong@xjtu.edu.cn
@software: MedAI
@file: train.py
@time: 2022/7/22 14:49
@version:
@desc:
'''
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import datetime
import numpy as np
from torch.optim import lr_scheduler

from model.csnet_3d import CSNet3D
from model.unet3d import UNet3D
from model.PE_Net import PE_Net
from model.unetr import UNETR
from model.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from model.csnet_3d import CSNet3D
from dataloader.npy_3d_Loader import Data
from monai.losses import DiceCELoss
from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
import matplotlib.pyplot as plt
import sys
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = {
    'data_path': '/home/zk2/SCI/XPX/PE_Net/data/npy',
    'epochs': 600,
    'input_shape': (128,128,128),
    'snapshot': 50,
    'test_step': 1,
    'model_path': './save_models_randomcrop',
    'batch_size': 2,  # VNet 1 other 2
    'folder': 'folder2',
    'model_name': 'CSNet3D',
    'p_feature_path': './p_feature_map',
    'train_loss_path': './csv'
}

Test_Model = {'CSNet3D': CSNet3D,
              'UNet3D': UNet3D,
              'PE_Net': PE_Net
              'UNETR': UNETR,
              'CSNet3D': CSNet3D,
             ,
              }

best_score = [0]
ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_ckpt(net, iter):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    # torch.save(model.state_dict(),PATH)
    torch.save(net.state_dict(), os.path.join(ckpt_path, date + iter + '.pkl'))
    print("{} Saved model to:{}".format("\u2714", ckpt_path))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loss_plot(csv_path,epoch,folder):
    df = pd.read_csv(csv_path)
    loss_data = df['train_loss'].values
    epoches = np.arange(1,len(loss_data)+1)

    plt.plot(epoches,loss_data,'b',label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    # plt.legend()
    plt.savefig(str(args['model_name'])+'_train_loss_'+str(folder)+'.png')


def hot_map(pred_,num):   #pred 2 2 88 128 128
    # pred = torch.argmax(pred_, dim=1)
    # pred=pred_[0]
    # pred=pred.squeeze().cpu()
    # # pred=np.array(pred)
    # pred=pred[50,:,:]
    # prob_map = pred.squeeze().detach().numpy()
    # plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)  
    # plt.colorbar()  
    # path = os.path.join(args['p_feature_path'],'{}.png'.format(num))
    # plt.savefig(path)
    # plt.clf()
    # plt.close()
    pred_=pred_[1,:,:,:,:]
    pred_=pred_[0].mean(dim=0)
    # print(pred_.shape)
    plt.imshow(pred_.cpu().detach().numpy(), cmap='jet', vmin=0, vmax=1)  #cmap='viridis'
    plt.colorbar()  # 
    path = os.path.join(args['p_feature_path'],'{}.png'.format(num))
    plt.savefig(path)
    plt.clf()
    plt.close()

def train():
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    net = Test_Model[args['model_name']](2, 1).cuda()
    # net = UNETR(in_channels=1, out_channels=2, img_size=(224,256,256), pos_embed='conv', norm_name='instance').cuda()
    # net = nn.DataParallel(net, device_ids=[0]).cuda()
    logfile = os.path.join(ckpt_path,
                           '{}_{}_{}.txt'.format('IMA', str(args['model_name']), '90_128_128'))  # 训练日志保存地址
    sys.stdout = Logger(logfile)
    print("------------------------------------------")
    print("Network Architecture of Model unet")
    num_para = 0
    for name, param in net.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    # print(net)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, str(args['model_name'])))
    print("------------------------------------------")

    # load train dataset
    train_data = Data(args['data_path'], args['folder'], args['input_shape'], train=True)
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=4, shuffle=True)

    critrion2 = WeightedCrossEntropyLoss().cuda()
    critrion = nn.CrossEntropyLoss().cuda()
    critrion3 = DiceLoss().cuda()
    # critrion4 =DiceCELoss(
    #     to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6
    # ).cuda()
    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    iters = 1
   
    data_list = []  #存放每一个epoch的所有  batch的平均loss
    for epoch in range(args['epochs']):
        scheduler.step()
        net.train()
        loss_list = []  #存放一个epoch中的  每一个batch 的loss
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            print(f'************{pred.shape}**************')
            print(label.size())
            # hot_map(dec1,iters)
            loss_dice = critrion3(pred, label)
            label = label.squeeze(1)
            loss_ce = critrion(pred, label)
            loss_wce = critrion2(pred, label)
            loss = (loss_ce + 0.6 * loss_wce + 0.4 * loss_dice) / 3
            # loss = critrion4(pred, label)
            # print(loss)    #  tensor(0.2413, device='cuda:0', grad_fn=<DivBackward0>)
        
            loss_list.append(loss.item())
            loss.backward()
            
            optimizer.step()
            tp, fn, fp, iou = metrics3d(pred, label, pred.shape[0])
            if (epoch % 2) == 0:
                print(
                    '\033[1;36m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tIoU:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        iou / pred.shape[0]))
            else:
                print(
                    '\033[1;32m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tIoU:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        iou / pred.shape[0]))

            iters += 1

        data_list.append([np.mean(loss_list)/2])
        print(f'-----------{data_list}-----------')
        save = pd.DataFrame(data_list, columns=['train_loss'])
        save_csv_path=os.path.join(args['train_loss_path'], args['model_name']+'_train_loss_'+args['folder']+'.csv')
        save.to_csv(save_csv_path, index=False, header=True)
        
        loss_plot(save_csv_path,epoch,args['folder'])

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_tp, test_fn, test_fp, test_iou = model_eval(net)
            print("Average TP:{0:.4f}, average FN:{1:.4f},  average FP:{2:.4f},  average IOU:{3:.4f}".format(test_tp,
                                                                                                             test_fn,
                                                                                                             test_fp,
                                                                                                             test_iou))
            if test_iou > max(best_score):
                best_score.append(test_iou)
                print(best_score)
                modelname = ckpt_path + '/' + 'best_score' + '_checkpoint.pkl'
                print('the best model will be saved at {}'.format(modelname))
                torch.save(net.state_dict(), modelname)
    print("------------------the best score of model is--------------- :", best_score)
   



def model_eval(net):
    print("\033[1;30;43m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
    test_data = Data(args['data_path'], args['folder'], args['input_shape'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)
    net.eval()
    TP, FN, FP, IoU = [], [], [], []
    file_num = 0
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            pred_val = net(image)
            label = batch[1].cuda()
            label = label.squeeze(1)
            tp, fn, fp, iou = metrics3d(pred_val, label, pred_val.shape[0])
            print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test IoU:{3:.4f}".format(tp, fn, fp, iou))
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            IoU.append(iou)
            file_num += 1
        return np.mean(TP), np.mean(FN), np.mean(FP), np.mean(IoU)


if __name__ == '__main__':
    print("______________________")
    train()
