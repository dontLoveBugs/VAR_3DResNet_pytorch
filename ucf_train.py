# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/20 20:24
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import argparse
from datetime import datetime
import os
import shutil
import socket

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler

import utils
from dataset import get_training_set, get_test_set, get_validation_set
from mean import get_mean, get_std
from models.resnext import get_fine_tuning_parameters
from target_transforms import ClassLabel, VideoID
from temporal_transforms import TemporalRandomCrop, LoopPadding, TemporalCenterCrop
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from models import resnet, resnext, wide_resnet, pre_act_resnet, densenet
from train import train_epoch
from validation import val_epoch


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='/home/data/UnsupervisedDepth/wangixn/UCF-101/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='jpg_video_directory',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default='anotation/ucf101.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='ucf101',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument(
        '--n_classes',
        default=101,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,  # 0.1 ---> 0.001 when finetune
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-5, type=float, help='Weight Decay')  # when finetuning, 1e-3 --> 1e-5
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch Size')  # according to GPU
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=1,     # 不进行采样
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', default='./pretrained_models/resnext-101-kinetics-ucf101_split1.pth', type=str,
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=3,   # finetune conv5_x and fc
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.add_argument(
        '--n_threads',
        default=4,  # 4-->1
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnext',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=101,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    args = parser.parse_args()

    return args


def create_dataloader(args):
    if args.root_path != '':
        args.video_path = os.path.join(args.root_path, args.video_path)
        args.annotation_path = os.path.join(args.root_path, args.annotation_path)
        args.result_path = os.path.join(args.root_path, args.result_path)
        if args.resume_path:
            args.resume_path = os.path.join(args.root_path, args.resume_path)
        if args.pretrain_path:
            # args.pretrain_path = os.path.join(args.root_path, args.pretrain_path)
            args.pretrain_path = os.path.abspath(args.pretrain_path)
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)

    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)

    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(args.mean, [1, 1, 1])
    else:
        norm_method = Normalize(args.mean, args.std)

    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            args.scales, args.sample_size, crop_positions=['c'])

    spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(args.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(args.sample_duration)
    target_transform = ClassLabel()
    training_data = get_training_set(args, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=True)

    spatial_transform = Compose([
        # Scale(args.sample_size),
        Scale(int(args.sample_size / args.scale_in_test)),
        # CenterCrop(args.sample_size),
        CornerCrop(args.sample_size, args.crop_position_in_test),
        ToTensor(args.norm_value), norm_method
    ])
    temporal_transform = TemporalCenterCrop(args.sample_duration)
    target_transform = ClassLabel()
    validation_data = get_validation_set(
        args, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True)

    return train_loader, val_loader


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 默认使用GPU 0


def main():
    args = parse_command()
    # 如果有多GPU 使用多GPU训练
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use", torch.cuda.current_device())

    train_loader, val_loader = create_dataloader(args)

    print('train size:', len(train_loader))
    print('val size:', len(val_loader))

    # create results folder, if not already exists
    output_directory = utils.get_output_directory_run(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    torch.manual_seed(args.manual_seed)

    # 定义模型
    model = resnext.resnet101(
        num_classes=args.n_classes,
        shortcut_type=args.resnet_shortcut,
        cardinality=args.resnext_cardinality,
        sample_size=args.sample_size,
        sample_duration=args.sample_duration)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    if args.pretrain_path:
        print('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
        del pretrain # 清理缓存

    # parameters = get_fine_tuning_parameters(model, args.ft_begin_index)
    train_params = [{'params': resnext.get_1x_lr_params(model), 'lr': args.lr},
                    {'params': resnext.get_10x_lr_params(model), 'lr': args.lr * 10}]

    # loss函数
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # 优化器
    args.nesterov = False
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening
    optimizer = optim.SGD(
        train_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        dampening=dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    print('run')
    for i in range(args.begin_epoch, args.n_epochs + 1):

        train_epoch(i, train_loader, model, criterion, optimizer, logger)

        validation_loss = val_epoch(i, val_loader, model, criterion, output_directory, logger)

        if i % args.checkpoint == 0:
            save_file_path = os.path.join(output_directory,
                                          'save_{}.pth'.format(i))
            states = {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

        scheduler.step(validation_loss)


if __name__ == '__main__':
    main()
