import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.utils.rnn as trnn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,nconv1Out, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout,batch_first = True, bidirectional=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        
        self.init_weights()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.conv1 = nn.Conv2d(2048+2*nhid,nconv1Out,1)
        self.conv2 = nn.Conv2d(nconv1Out,1,1)
        self.softmax = nn.Softmax()
        self.poolatt = nn.AvgPool2d(14)
        self.fc1 = nn.Linear(2*nhid,2)
        self.fc2 = nn.Linear(1024,3000)
        
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, question,length):
        length = list(length.data.cpu().numpy())

        emb = self.drop(self.encoder(question))
        emb = self.tanh(emb)
        #emb = emb.contiguous
        #print input.size()
        hidden = self.init_hidden(len(length))
        seqs = trnn.pack_padded_sequence(emb, length, batch_first=True)

        seqs, hidden = self.rnn(seqs, hidden)
        #out,_ = trnn.pad_packed_sequence(seqs, batch_first=True)
        
        bilstmout = torch.cat([hidden[0][0],hidden[0][1]],-1)

        #print bilstmout.unsqueeze(-1).unsqueeze(-1).size()

        fc1fea = self.fc1(bilstmout)

        return fc1fea

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(2*self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(2*self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
