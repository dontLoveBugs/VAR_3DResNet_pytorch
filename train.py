import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, logger = None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if torch.cuda.is_available():
            targets = targets.cuda(async=True)

            inputs = inputs.cuda()
            targets = targets.cuda()

        # inputs = Variable(inputs)
        # targets = Variable(targets)
        torch.cuda.synchronize()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # batch_logger.log({
        #     'epoch': epoch,
        #     'batch': i + 1,
        #     'iter': (epoch - 1) * len(data_loader) + (i + 1),
        #     'loss': losses.val,
        #     'acc': accuracies.val,
        #     'lr': optimizer.param_groups[0]['lr']
        # })

        print('Train: Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

        if logger is not None:
            current_step = epoch * len(data_loader) + i
            logger.add_scalar('Train/Loss', losses.avg, current_step)
            logger.add_scalar('Train/Acc', accuracies.avg, current_step)

    # epoch_logger.log({
    #     'epoch': epoch,
    #     'loss': losses.avg,
    #     'acc': accuracies.avg,
    #     'lr': optimizer.param_groups[0]['lr']
    # })


