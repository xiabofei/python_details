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
parser.add_argument('--epochs', type=int, default=64,
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


# print corpus.train.size()

ntokens = 605
###############################################################################
# Build the model
###############################################################################


model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.nConv1Out, args.dropout)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3 ,1.0]), size_average=False)


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
    if len(ori_label)>1:
        f = open('./compare.dat', 'a')
        p_0_o_0 = 0
        p_1_o_0 = 0
        p_0_o_1 = 0
        p_1_o_1 = 0
        for p,o in zip(pred.tolist()[0], ori_label):
            if p==0 and o==0:
                p_0_o_0 += 1
            elif p==1 and o==0:
                p_1_o_0 += 1
            elif p==0 and o==1:
                p_0_o_1 += 1
            elif p==1 and o==1:
                p_1_o_1 += 1
        f.write('p_0_o_0:'+str(p_0_o_0)+','+'p_0_o_1:'+str(p_0_o_1)+'\n')
        f.write('p_1_o_0:'+str(p_1_o_0)+','+'p_1_o_1:'+str(p_1_o_1)+'\n\n')
        f.close()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


trainmap = cPickle.load(open('data4lstm_train', 'rb'))

questrainfealist = []
labeltrainlist = []
lengthtrainlist = []

questrainfealistShu = []
labeltrainlistShu = []
lengthtrainlistShu = []
for info in trainmap:
    labeltrainlist.append(info['label'])
    lengthtrainlist.append(len(info['data']))
    for i in range(len(info['data']), 235):
        info['data'].append(604)
    questrainfealist.append(info['data'])
idxs = range(len(questrainfealist))

np.random.shuffle(idxs)

for i in range(len(idxs)):
    questrainfealistShu.append(questrainfealist[idxs[i]])
    labeltrainlistShu.append(labeltrainlist[idxs[i]])
    lengthtrainlistShu.append(lengthtrainlist[idxs[i]])


def train(epoch, optimizer, quesfeaShu, labelShu, lengthShu):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    total_loss = 0
    start_time = time.time()
    # st(context=27)
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

validmap = cPickle.load(open('data4lstm_valid', 'rb'))

questrainfealist_valid = []
labeltrainlist_valid = []
lengthtrainlist_valid = []

questrainfealistShu_valid = []
labeltrainlistShu_valid = []
lengthtrainlistShu_valid = []
for info in validmap:
    labeltrainlist_valid.append(info['label'])
    lengthtrainlist_valid.append(len(info['data']))
    for i in range(len(info['data']), 235):
        info['data'].append(604)
    questrainfealist_valid.append(info['data'])
idxs_valid = range(len(questrainfealist_valid))

np.random.shuffle(idxs_valid)

for i in range(len(idxs_valid)):
    questrainfealistShu_valid.append(questrainfealist_valid[idxs_valid[i]])
    labeltrainlistShu_valid.append(labeltrainlist_valid[idxs_valid[i]])
    lengthtrainlistShu_valid.append(lengthtrainlist_valid[idxs_valid[i]])


def valid(epoch, quesfeaShu, labelShu, lengthShu):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    start_time = time.time()
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
        # print output
        loss = criterion(output, trlabel) / (batch_size)
        prec1, = accuracy(output.data, trlabel.data, topk=(1,), ori_label=labeltrainarray)
        # label 0 or 1
        losses.update(loss.data[0], batch_size)
        top1.update(prec1[0], batch_size)

        # loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        print str(top1.avg) + ' ' + str(loss.data[0]) + ' ' + 'batch_valid ' + str(i)
    # update better performance model
    global best_score
    if top1.avg > best_score:
        torch.save(model, args.save)
        print 'save model'
        best_score = top1.avg
    print str(top1.avg) + ' ' + str(loss.data[0]) + ' ' + 'epoch_valid ' + str(epoch)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
st(context=27)
best_score = 0
try:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(epoch, optimizer, questrainfealistShu, labeltrainlistShu, lengthtrainlistShu)
        valid(epoch, questrainfealistShu_valid, labeltrainlistShu_valid, lengthtrainlistShu_valid)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


def test(model, quesfeaShu, labelShu, lengthShu):

    model.eval()

    idx = sorted(range(len(lengthShu)), key=lambda x: lengthShu[x], reverse=True)

    _quesfeaShu = []
    _labelShu = []
    _lengthShu = []

    for j in range(len(idx)):
        _quesfeaShu.append(quesfeaShu[idx[j]])
        _labelShu.append(labelShu[idx[j]])
        _lengthShu.append(lengthShu[idx[j]])

    questrainarray = np.asarray(_quesfeaShu)
    labeltrainarray = np.asarray(_labelShu)
    lengthtrainarray = np.asarray(_lengthShu)

    tmp = [questrainarray, labeltrainarray, lengthtrainarray]
    tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
    trques, trlabel, length = tmp
    if args.cuda:
        trlabel.cuda()
    output = model(trques, length)
    # st(context=27)
    print("precesion 1 : %s" % accuracy(output.data, trlabel.data, topk=(1,), ori_label=labeltrainarray))


# Load the best saved model.
with open(args.save, 'rb') as f:
    test(torch.load(f), questrainfealistShu_valid, labeltrainlist_valid, lengthtrainlist_valid)
