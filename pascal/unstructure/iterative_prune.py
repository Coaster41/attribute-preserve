import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import *
import logging
from dataset import get_loader
from utils import set_random_seed, compute_mAP, AverageMeter
import argparse
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Unstructured Pruning Pascal Training')

parser.add_argument('--gpu', action='store_true')
parser.add_argument('--seed', default=None, type=int, 
                    help="Random Seed")
parser.add_argument('--arch', default='VGG16_VOC', type=str,
                    help='Model architecture')
parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                    help="Data path, here for multi-label classification")
parser.add_argument('--exp_name', default='None', type=str,
                    help='Name of Current Running')

parser.add_argument('--epochs', default=30, type=int, 
                    help='number of total epochs to train')
parser.add_argument('--learning_rate', default=0.001, type=float, 
                    help='initial learning rate')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='Optimizer Momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, 
                    help='For Optimizer, weight decay')
parser.add_argument('--print_freq', default=30, type=int, 
                    help='print frequency')

##Pruning Arguments
parser.add_argument('--model_path', default='', type=str, 
                    help='Initialize model path & Teacher path')
parser.add_argument('--percent', default=0.2, type=float,
                    help='percentage of weight to prune')
parser.add_argument('--iter', default=16, type=int,
                    help='How many times to repeat')

def main():

    global args
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    # create model
    model = VGG('VGG16_VOC', input_dims=(3,227,227))

    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    _, mAP_full = load_model(args.model_path, model)

    train_loader = get_loader(data_name='VOC2012',
                            data_path=args.data_path,
                            split='train',
                            batch_size= args.batch_size)
    val_loader = get_loader(data_name='VOC2012',
                            data_path=args.data_path,
                            split='val',
                            batch_size= args.batch_size)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()

    print('Full Network Starting mAP : {}'.format(mAP_full))

    ##For each iteration
    for it in range(1, args.iter+1):
        best_mAP = 0
        prune_percent  = 1 - ((1 - args.percent)**it)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        file_handler = logging.FileHandler('./loggers/{}_{:.3f}.txt'.format(args.exp_name, prune_percent))
        logger.addHandler(file_handler)
        logger.addHandler(streamHandler)

        logger.info('Start Iterative Pruning -- iter : {} (prune_rate : {:.3f})'.format(it, prune_percent))
        save_file = './weights/{}_{:.3f}'.format(args.exp_name, prune_percent)

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        model, test_mAP = prune(prune_percent, model, criterion, val_loader, save_file, logger)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        logger.info('Start Finetuning model with prune_rate with mAP {:.3f}'.format(test_mAP))

        for epoch in range(0, args.epochs):

            #####################################################################################################
            num_parameters = get_conv_zero_param(model)
            logger.info('Zero parameters: {}'.format(num_parameters))
            num_parameters = sum([param.nelement() for param in model.parameters()])
            logger.info('Parameters: {}'.format(num_parameters))
            #####################################################################################################

            #Train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, logger)

            #Evaluate on validation set
            valid_mAP = validate(val_loader, model, criterion, epoch, logger)

            #Remember best prec@1 and save checkpoint
            is_best = valid_mAP > best_mAP
            best_mAP = max(valid_mAP, best_mAP)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args,
                'p_rate': prune_percent,
                'state_dict': model.state_dict(),
                'best_mAP': best_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best,checkpoint=save_file)
            logger.info('Saved best mAP for prune {:.3f} : {:.3f}'.format(prune_percent, best_mAP))

        with open(args.exp_name+'.txt', 'a') as f:
            f.write('{:.3f}\t{:.3f}\n'.format((1-prune_percent), best_mAP))
        del logger
    return

def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total

def train(train_loader, model, criterion, optimizer, epoch, logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    train_mAP = AverageMeter()

    #Switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu:
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        # compute output
        output, _, _ = model(input)

        loss = criterion(output, target)
        train_losses.update(loss.item(), input.size(0))

        # measure accuracy and record loss
        train_mAP.update(100*compute_mAP(target.data, output.data), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'mAP {mAP.val:.3f} ({mAP.avg:.3f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=train_losses, mAP=train_mAP,))

def validate(val_loader, model, criterion, epoch, logger):
    valid_losses = AverageMeter()
    outputs = []
    targets = []

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            # compute output
            output, _, _ = model(images, mode='eval')
            loss = criterion(output, target)
            outputs.append(output)
            targets.append(target)
            valid_losses.update(loss.item(), images.size(0))

        mAP = 100*compute_mAP(torch.cat(targets, dim=0).data, torch.cat(outputs, dim=0).data)
        logger.info('Epoch[{ep}] : val_loss {loss.avg:.3f} mAP {map_: .3f}'
            .format(ep=epoch, loss=valid_losses, map_=mAP))

    return mAP

def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    filepath = os.path.join(checkpoint, 'checkpoint_'+filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_'+filename))

def load_model(file_path, model, optimizer=None):
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_mAP']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
    return start_epoch, best_acc

def prune(percent, model, criterion, val_loader, savefile, logger):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    conv_weights = torch.zeros(total).cuda()
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True

    logger.info('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    test_mAP = validate(val_loader, model, criterion, 0, logger)

    save_checkpoint({
        'epoch': 0,
        'args': args,
        'p_rate':pruned/total,
        'state_dict': model.state_dict(),
        'best_mAP': test_mAP,
        }, False, savefile, filename='beforefinetune.pth.tar')

    with open(os.path.join(savefile, 'result.txt'), 'w') as f:
        f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        f.write('After Pruning: Test mAP:  %.2f\n' % (test_mAP))

        if zero_flag:
            f.write("There exists a layer with 0 parameters left.")
    return model, test_mAP

if __name__ == '__main__':
    main()