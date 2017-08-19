import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import model
import numpy as np
import scipy.io as scio
import cPickle
import json

from ipdb import set_trace as st

import torch.optim as optim
from datashape.coretypes import int64, float32

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nConv1Out', type=int, default=512,
                    help='conv1 output size')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--decaysteps', type=float, default=50000,
                    help='decaysteps')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


ntokens_with_NAN = 546 + 1
sentence_max_length = 206

###############################################################################
# Build the model
###############################################################################


model = model.RNNModel(args.model, ntokens_with_NAN, args.emsize, args.nhid, args.nlayers, args.nConv1Out, args.dropout)
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss(size_average=False)


###############################################################################
# Training code
###############################################################################


def accuracy(output, target, topk=(1,), ori_label=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    # write to local
    if len(ori_label) > 1:
        f = open('./compare.dat', 'a')
        p_0_o_0 = 0
        p_1_o_0 = 0
        p_0_o_1 = 0
        p_1_o_1 = 0
        for p, o in zip(pred.tolist()[0], ori_label):
            if p == 0 and o == 0:
                p_0_o_0 += 1
            elif p == 1 and o == 0:
                p_1_o_0 += 1
            elif p == 0 and o == 1:
                p_0_o_1 += 1
            elif p == 1 and o == 1:
                p_1_o_1 += 1
        f.write('p_0_o_0:' + str(p_0_o_0) + ',' + 'p_0_o_1:' + str(p_0_o_1) + '\n')
        f.write('p_1_o_0:' + str(p_1_o_0) + ',' + 'p_1_o_1:' + str(p_1_o_1) + '\n\n')
        f.close()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_data(x_path, y_path):
    # load X data
    x_map = cPickle.load(open(x_path, 'rb'))
    ques_fealist = []
    length_list = []
    for info in x_map:
        length_list.append(len(info['data']))
        for i in range(len(info['data']), sentence_max_length):
            info['data'].append(ntokens_with_NAN - 1)
        ques_fealist.append(info['data'])
    # load manually Y label data
    label_list = []
    index_label = {int(k): v for k, v in json.load(open(y_path, 'r')).items()}
    for k in sorted(index_label.keys()):
        label = 1 if ('1' in index_label[k]['label']) else 0
        label_list.append(label)
    # shuffled data
    ques_fealist_Shu = []
    label_list_Shu = []
    length_list_Shu = []
    idxs = range(len(ques_fealist))
    np.random.shuffle(idxs)
    for i in range(len(idxs)):
        ques_fealist_Shu.append(ques_fealist[idxs[i]])
        label_list_Shu.append(label_list[idxs[i]])
        length_list_Shu.append(length_list[idxs[i]])
    return ques_fealist_Shu, label_list_Shu, length_list_Shu


def train(epoch, optimizer, quesfeaShu, labelShu, lengthShu):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    for i in range(0, len(quesfeaShu) / args.batch_size):
        if i == len(quesfeaShu) / args.batch_size - 1:
            batchend = len(quesfeaShu)
        else:
            batchend = (i + 1) * (args.batch_size)
        batchstart = i * (args.batch_size)
        batch_size = batchend - batchstart

        quesfeabatch = []
        labelbatch = []
        lengthbatch = []

        quesfeaOri = quesfeaShu[batchstart:batchend]
        labelOri = labelShu[batchstart:batchend]
        lengthOri = lengthShu[batchstart:batchend]

        idxbatch = sorted(range(len(lengthOri)), key=lambda x: lengthOri[x], reverse=True)
        for j in range(len(idxbatch)):
            quesfeabatch.append(quesfeaOri[idxbatch[j]])
            labelbatch.append(labelOri[idxbatch[j]])
            lengthbatch.append(lengthOri[idxbatch[j]])

        questrainarray = np.asarray(quesfeabatch)
        labeltrainarray = np.asarray(labelbatch)
        lengthtrainarray = np.asarray(lengthbatch)

        tmp = [questrainarray, labeltrainarray, lengthtrainarray]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
        trques, trlabel, length = tmp
        if args.cuda:
            trlabel.cuda()

        output = model(trques, length)
        loss = criterion(output, trlabel) / (batch_size)
        prec1, = accuracy(output.data, trlabel.data, topk=(1,))

        losses.update(loss.data[0], batch_size)
        top1.update(prec1[0], batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        print str(top1.avg) + ' ' + str(top1.val) + ' ' + str(loss.data[0]) + ' ' + 'batch ' + str(i)
    print str(top1.avg) + ' ' + str(top1.val) + ' ' + str(loss.data[0]) + ' ' + 'epoch ' + str(epoch)


###############################################################################
# Validating code
###############################################################################


def valid(epoch, quesfeaShu, labelShu, lengthShu):
    top1 = AverageMeter()

    model.eval()

    for i in range(0, len(quesfeaShu) / args.batch_size):
        if i == len(quesfeaShu) / args.batch_size - 1:
            batchend = len(quesfeaShu)
        else:
            batchend = (i + 1) * (args.batch_size)
        # print batchend
        batchstart = i * (args.batch_size)
        batch_size = batchend - batchstart

        quesfeabatch = []
        labelbatch = []
        lengthbatch = []

        quesfeaOri = quesfeaShu[batchstart:batchend]
        labelOri = labelShu[batchstart:batchend]
        lengthOri = lengthShu[batchstart:batchend]
        idxbatch = sorted(range(len(lengthOri)), key=lambda x: lengthOri[x], reverse=True)
        for j in range(len(idxbatch)):
            quesfeabatch.append(quesfeaOri[idxbatch[j]])
            labelbatch.append(labelOri[idxbatch[j]])
            lengthbatch.append(lengthOri[idxbatch[j]])

        questrainarray = np.asarray(quesfeabatch)
        labeltrainarray = np.asarray(labelbatch)
        lengthtrainarray = np.asarray(lengthbatch)

        tmp = [questrainarray, labeltrainarray, lengthtrainarray]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
        trques, trlabel, length = tmp
        if args.cuda:
            trlabel.cuda()

        output = model(trques, length)
        loss = criterion(output, trlabel) / (batch_size)

        prec1, = accuracy(output.data, trlabel.data, topk=(1,))
        top1.update(prec1[0], batch_size)
        print str(top1.avg) + ' ' + str(loss.data[0]) + ' ' + 'batch_valid ' + str(i)
    global best_score
    if top1.avg > best_score:
        torch.save(model, args.save)
        print 'save model'
        best_score = top1.avg
    print str(top1.avg) + ' ' + str(loss.data[0]) + ' ' + 'epoch_valid ' + str(epoch)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# st(context=21)
questrainfealistShu, labeltrainlistShu, lengthtrainlistShu = \
    load_data('../4lstm/lstm.pkl', '../4lstm/mri_label.json')

questrainfealistShu_valid, labeltrainlistShu_valid, lengthtrainlistShu_valid = \
    load_data('../4lstm/lstm.pkl', '../4lstm/mri_label.json')

# At any point you can hit Ctrl + C to break out of training early.
best_score = 0
try:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(epoch, optimizer, questrainfealistShu, labeltrainlistShu, lengthtrainlistShu)
        valid(epoch, questrainfealistShu_valid, labeltrainlistShu_valid, lengthtrainlistShu_valid)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
