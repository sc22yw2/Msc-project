import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from data_loader.ImageNet_datasets import ImageNetData
import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter,MSEMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import wandb
from losses import LDAMLoss,LogitAdjust,FocalLoss


def loss_combination(sex_loss, age_loss,cls_num):
    age_cls_num = cls_num["age_cls_num_list"]
    sex_cls_num = cls_num["sex_cls_num_list"]
    age_loss_dict = {'CE':nn.CrossEntropyLoss(), 'LDAM':LDAMLoss(cls_num_list=age_cls_num), 'Focal':FocalLoss(class_num=age_cls_num), 'LogitAdjust':LogitAdjust(cls_num_list=age_cls_num)}
    sex_loss_dict = {'CE':nn.CrossEntropyLoss(), 'LDAM':LDAMLoss(cls_num_list=sex_cls_num), 'Focal':FocalLoss(class_num=sex_cls_num), 'LogitAdjust':LogitAdjust(cls_num_list=sex_cls_num)}
    return [sex_loss_dict[sex_loss],age_loss_dict[age_loss]]



def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def main(args):
    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('./logs/'+args.model+'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomResizedCrop((224, 224)),

            transforms.RandomApply([
                transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=45),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_datasets = ImageNetData(img_root=args.data_root,img_file='train', transform= data_transforms['train'])
    val_datasets   = ImageNetData(img_root=args.data_root,img_file='test',  transform=data_transforms['val'])
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size*len(gpus), drop_last=True, shuffle=True, num_workers=4)
    val_dataloaders   = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=4)



    if args.debug:
        x, y =next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    if  'resnet50' == args.model.split('_')[0]:
        my_model = resnet_cbam.MyResNet50(pretrained=False,num_classes=2)
    elif 'resnet50-cbam' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50_cbam(pretrained=False,num_classes=2)

    else:
        raise ModuleNotFoundError


    #my_model.apply(fc_init)
    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    # loss_fn = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss()]
    loss_fn = loss_combination(args.sex_loss_function, args.age_loss_function,train_datasets.cls_num)
    optimizer = optim.AdamW(my_model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)           #

    metric = [ClassErrorMeter([1], True),MSEMeter()]
    start_epoch = 0
    num_epochs  = 90

    my_trainer = Trainer(my_model, args.model, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_dataloaders, \
                        val_dataloaders, metric, start_epoch, num_epochs, args.debug, logger, writer)
    my_trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-d', '--data_root', default='./data',
                         type=str, help='data root')
    parser.add_argument('-sex_loss', '--sex_loss_function', default='CE',
                        type=str, choices=['CE', 'LDAM','Focal','LogitAdjust'],help='loss function')
    parser.add_argument('-age_loss', '--age_loss_function', default='CE',
                        type=str, choices=['CE', 'LDAM', 'Focal', 'LogitAdjust'], help='loss function')
    parser.add_argument('-m', '--model', default='resnet50-cbam',
                         type=str, help='resnet50-cbam or resnet50')
    parser.add_argument('--batch_size', default=32,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')

    args = parser.parse_args()
    wandb_name = str(args.model)+'_sex_'+str(args.sex_loss_function)+'_age_'+str(args.age_loss_function)

    wandb.init(dir=os.path.abspath("wandb"),config=args,
               project="CBAM",
               name=wandb_name,)
    main(args)
