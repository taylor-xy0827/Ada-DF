import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import create_model
from utils import set_random_seed, Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training')
# train configs
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_classes', default=7, type=int)
parser.add_argument('--num_samples', default=30000, type=int)
# method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', default=False, type=bool)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--tops', default=0.7, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
# common configs
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--dataset', default='raf', type=str)
parser.add_argument('--data_path', default='./datasets/raf-basic', type=str)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

best_acc=0
best_epoch = 0

# set device
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

# set random seed
if args.seed is not None:
    set_random_seed(args.seed)

def main():
    global best_acc
    global best_epoch
    global device

    # log file
    logger = Logger('./results/log-'+time.strftime('%b%d_%H-%M-%S')+'.txt')
    # logger = Logger('./results/log-'+str(args.beta)+'-'+str(args.threshold)+'-'+time.strftime('%b%d_%H-%M-%S')+'.txt')
    logger.info(args)

    # TensorBoard writer
    writer = SummaryWriter()

    # initialization
    LD = torch.zeros(args.num_classes, args.num_classes).to(device)
    for i in range(args.num_classes):
        LD[i] = torch.zeros(args.num_classes).fill_((1-args.threshold)/(args.num_classes-1)).scatter_(0, torch.tensor(i), args.threshold)
    if args.sharpen == True:
        LD = torch.pow(LD, 1/args.T) / torch.sum(torch.pow(LD, 1/args.T), dim=1)
    LD_max = torch.max(LD, dim=1)
    LD_sum = LD

    nan = float('nan')
    weights_avg = [nan for i in range(args.num_classes)]
    weights_max = [nan for i in range(args.num_classes)]
    weights_min = [nan for i in range(args.num_classes)]

    # model
    logger.info('Load model...')
    model = create_model(args.num_classes, args.drop_rate).to(device)   

    # dataloaders
    train_loader, test_loader = get_dataloaders(args.dataset, args.data_path, args.batch_size, args.num_workers, args.num_samples)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
    criterion_kld = nn.KLDivLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    logger.info('Start training.')

    for epoch in range(1, args.epochs+1):
        logger.info('----------------------------------------------------------')
        logger.info('Epoch: %d, Learning Rate: %f', epoch, optimizer.param_groups[0]['lr'])
        logger.info(f'Maximums of LD: {[round(LD_max[0].cpu().tolist()[i], 4) for i in range(args.num_classes)]}')
        logger.info(f'Average weights: {[round(weights_avg[i], 4) for i in range(args.num_classes)]}')
        logger.info(f'Max weights: {[round(weights_max[i], 4) for i in range(args.num_classes)]}')
        logger.info(f'Min weights: {[round(weights_min[i], 4) for i in range(args.num_classes)]}')

        # train
        train_loss, train_loss_ce, train_loss_kld, alpha_1, alpha_2 = train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch)
        _, train_acc, outputs_new, targets_new, weights_new = validate(train_loader, model, criterion, epoch, phase='train')

        LD = generate_adaptive_LD(outputs_new, targets_new, args.num_classes, args.threshold, args.sharpen, args.T)
        LD_max = torch.max(LD, dim=1)

        weights_avg, weights_max, weights_min = generate_average_weights(weights_new, targets_new, args.num_classes, args.max_weight, args.min_weight)

        # test
        test_loss, test_acc, _, _, _ = validate(test_loader, model, criterion, epoch, phase='test')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch    

        logger.info('')
        logger.info('Alpha_1, Alpha_2: %.2f, %.2f Beta: %.2f', alpha_1, alpha_2, args.beta)
        logger.info('Train Acc: %.2f', train_acc)
        logger.info('Test Acc: %.2f', test_acc)
        logger.info('Best Acc: %.2f (%d)', best_acc, best_epoch)

        is_best = (best_epoch==epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'class_distributions': LD.detach(),
            }, 
            is_best)

        scheduler.step()

def train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch):
    if args.alpha is not None:
        alpha_1 = args.alpha
        alpha_2 = 1 - args.alpha
    else:
        if epoch <= args.beta:
            alpha_1 = math.exp(-(1-epoch/args.beta)**2)
            alpha_2 = 1
        else:
            alpha_1 = 1
            alpha_2 = math.exp(-(1-args.beta/epoch)**2)
    
    # losses
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_kld = AverageMeter()
    losses_rr = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')

    # training
    model.train()
    for i, (images, labels, indexes) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if args.dataset == 'sfew':
            batch_size, ncrops, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

        outputs_1, outputs_2, attention_weights = model(images)

        tops = int(args.batch_size * args.tops)

        # Rank Regularization
        _, top_idx = torch.topk(attention_weights.squeeze(), tops)
        _, down_idx = torch.topk(attention_weights.squeeze(), args.batch_size - tops, largest = False)

        high_group = attention_weights[top_idx]
        low_group = attention_weights[down_idx]
        high_mean = torch.mean(high_group)
        low_mean = torch.mean(low_group)
        diff  = low_mean - high_mean + args.margin_1

        if diff > 0:
            RR_loss = diff
        else:
            RR_loss = torch.from_numpy(np.array(0))

        loss_ce = criterion(outputs_1, labels).mean()

        # label fusion
        attention_weights = attention_weights.squeeze(1)
        attention_weights = ((attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())) * (args.max_weight-args.min_weight) + args.min_weight 

        attention_weights = attention_weights.unsqueeze(1)

        labels_onehot = F.one_hot(labels, args.num_classes)
        targets = (1-attention_weights) * F.softmax(outputs_1, dim=1) + attention_weights * LD[labels]

        loss_kld = criterion_kld(F.log_softmax(outputs_2, dim=1), targets).sum() / args.batch_size

        loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss

        # record loss
        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_kld.update(loss_kld.item(), images.size(0))
        losses_rr.update(RR_loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=losses.avg, loss_ce=losses_ce.avg, loss_kld=losses.avg, loss_rr = losses_rr.avg)

    return losses.avg, losses_ce.avg, losses_kld.avg, alpha_1, alpha_2

def validate(test_loader, model, criterion, epoch, phase='train'):
    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()

    outputs_new = torch.ones(1, args.num_classes).to(device)
    targets_new = torch.ones(1).long().to(device)
    weights_new = torch.ones(1, 1).float().to(device)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')

    with torch.no_grad():
        for i, (inputs, targets, indexes) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if args.dataset == 'sfew':
                batch_size, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

            if phase == 'train':
                if args.dataset == 'sfew':
                    targets = torch.repeat_interleave(targets, repeats=ncrops, dim=0)
                outputs, _, attention_weights = model(inputs)
            else:
                _, outputs, attention_weights = model(inputs)
                if args.dataset == 'sfew':
                    outputs = outputs.view(batch_size, ncrops, -1)
                    outputs = torch.sum(outputs, dim=1) / ncrops

            loss = criterion(outputs, targets).mean()

            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets), dim=0)
            weights_new = torch.cat((weights_new, attention_weights), dim=0)

            top1, _ = get_accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            accs.update(top1.item(), inputs.size(0))

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

    return (losses.avg, accs.avg, outputs_new, targets_new, weights_new) 

if __name__ == '__main__':
    main()