import csv
import os

import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, output_directory, logger = None, write_to_file=True):
    print('validation at epoch {}'.format(epoch))

    fieldnames = ['video', 'label']

    if write_to_file:
        filename = 'test-' + str(epoch) + '.csv'
        test_csv = os.path.join(output_directory, filename)
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, paths) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # if not opt.no_cuda:
        #     targets = targets.cuda(async=True)
        # inputs = Variable(inputs, volatile=True)
        # targets = Variable(targets, volatile=True)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        torch.cuda.synchronize()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        torch.cuda.synchronize()

        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Test: Epoch: [{0}][{1}/{2}]\t'
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

        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        print('pred: ', pred.item())

        if write_to_file:
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'video':paths[0], 'label':pred.item() + 1})

    # logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
    if logger is not None:
        logger.add_scalar('Test/loss', losses.avg, epoch)
        logger.add_scalar('Test/Acc', accuracies.avg, epoch)
    return losses.avg
