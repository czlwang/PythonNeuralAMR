from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from tqdm import tqdm as tqdm

import json
import torch.utils.data as data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 40

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS and UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(src_path, targ_path):
    input_lang = Lang("sen")
    output_lang = Lang("amr")
    pairs = []

    sen_lines = open('data/training-dfs-linear_targ.txt', encoding='utf-8').\
        read().strip().split('\n')
    amr_lines = open('data/training-dfs-linear_src.txt', encoding='ascii').\
        read().strip().split('\n')
    #giga_lines = open('data/gigaword.txt.anonymized', encoding='ascii').\
    #    read().strip().split('\n')

    #for line in giga_lines:
    #    input_lang.addSentence(line)
    #    output_lang.addSentence(line)
        
    pairs = [list(x) for x in zip(sen_lines, amr_lines)]
    
    pairs = filterPairs(pairs)

    print("Read %s sentence pairs" % len(pairs))
    #print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    #print('.\n' in input_lang.word2index)
    return input_lang, output_lang, pairs

def prepareSelfTrainData(encoder, decoder, sentences, max_length=MAX_LENGTH):
    pairs = []
    
    for sen in sentences:
        if len(sen.split(' ')) < max_length:
            amr = " ".join(evaluate(encoder, decoder, sen)[0])
            pairs.append([sen, amr])
    #print(pairs)
    pairs = filterPairs(pairs)
    
    return pairs
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        #print("input shape encoder", input.shape)
        #print("embedded shape encoder", embedded.shape)
        output = embedded
        output, hidden  = self.lstm(output, hidden)
        #print("output shape encoder", output.shape)
        #print("hidden shape encoder", hidden[0].shape)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2*self.num_layers, 1, self.hidden_size, device=device), torch.zeros(2*self.num_layers, 1, self.hidden_size, device=device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size + self.num_layers*self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print("input attnDecoder shape", input.shape)
        #print("hidden attnDecoder shape", hidden[0].shape)
        # 1x1xhidden_size (1x1x256)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        #print("embedded attnDecoder shape", embedded.shape)
        #print("embedded[0] attnDecoder shape", embedded[0].shape)
        #print("hidden[0].view(1,-1) attnDecoder shape", hidden[0].view(1,-1).shape)
        #print("hidden[0] attnDecoder shape", hidden[0].shape)
        #print("encoder_outputs shape", encoder_outputs.shape)
        #print("encoder_outputs shape unsqueeze", encoder_outputs.unsqueeze(0).shape)
        #encoder_outputs is seq_length x 2*hidden_size (bi-directional)

        # hidden[0].view(1,-1) is
        # (1x(hidden_size*num_layers)) (1x1024)
        # embedded[0] is 1xhidden_size 1x256
        # attn_weights is 1xmax_length 1x10
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0].view(1,-1)), 1)), dim=1)
        #print("attn_weights.shape", attn_weights.shape)
        # attn_weights.unsqueeze(0) is 1x1x10
        # encoder_outputs.unsqueeze(0) is 1 x max_length x 2*hidden_size or 1x10x512
        # attn_applied is 1 x 1 x 512: intuitively - which word to focus on
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #print("attn_applied.shape", attn_applied.shape)
        # output is concat of 1 x hidden_size and 1 x hidden_size. 
        # in other words, the concat of input and attention.
        # Here, 1 is batch size (?)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #print("output.shape", output.shape)
        output = self.attn_combine(output).unsqueeze(0)
        #print("attn_combine.shape", output.shape)
        
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        #output is 1x1x2*hidden_size

        #print("lstm output.shape", output.shape)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device), torch.zeros(self.num_layers, 1, self.hidden_size, device=device))

def indexesFromSentence(lang, sentence):
    sentence = sentence.rstrip()
    return [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = (encoder_hidden[0][-encoder.num_layers:], encoder_hidden[1][-encoder.num_layers:])#only take first two layers (no bidirection in decoder). Need cell and hidden, so give a tuple.
    #print("encoder_hidden[0].shape ", encoder_hidden[0].shape)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_epochs, pairs, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_epoch = 0  # Reset every print_every
    plot_loss_epoch = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(len(pairs)]
    training_pairs = list(map(tensorsFromPair, pairs))
    #criterion = nn.NLLLoss()
    criterion = nn.functional.cross_entropy

    #print("training pairs len", len(training_pairs))
    for iter in range(n_epochs):
        for i in tqdm(range(len(training_pairs))):
            training_pair = training_pairs[i]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            #print("input tensor shape", input_tensor.shape)
            #print("target tensor shape", target_tensor.shape)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_epoch += loss
            plot_loss_epoch += loss

        print("epoch loss", print_loss_epoch)
        print_loss_epoch = 0

        plot_losses.append(plot_loss_epoch)
        plot_loss_epoch = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    decoder_hidden = (encoder_hidden[0][-encoder.num_layers:], encoder_hidden[1][-encoder.num_layers:])#only take first two layers (no bidirection in decoder). Need cell and hidden, so give a tuple.

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length-1):#Hacky workaround TODO
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs,  n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig('attention.png', bbox_inches='tight')


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

plt.switch_backend('agg')

input_lang, output_lang, train_pairs = prepareData("data/training-dfs-linear_targ.txt", "data/training-dfs-linear_src.txt")
#print(random.choice(pairs))

num_layers = 2
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 10, train_pairs)

torch.save(encoder1, "saved_models/encoder1.pt")
torch.save(attn_decoder1, "saved_models/attn_decoder1.pt")

self_train_iter = 3
self_train_sample = 3

f3 = open("data/gigaword_sample/gigaword.txt.anonymized","r", encoding='ascii')
giga_lines = list(f3.readlines())
f3.close()
for i in range(self_train_iter):
    print("self_train_iter", i)
    self_train_pairs = [x for x in giga_lines if len(x.split(' ')) < MAX_LENGTH][:self_train_sample*10**i] 
    #print(len(self_train_pairs))
    #print([len(x.split(' ')) for x in giga_lines])
    self_train_pairs = prepareSelfTrainData(encoder1, attn_decoder1, self_train_pairs)
    print("self training on ", len(self_train_pairs),  " pairs")
    trainIters(encoder1, attn_decoder1, 5, self_train_pairs)
    print("fine tuning on ", len(train_pairs),  " pairs")
    trainIters(encoder1, attn_decoder1, 5, train_pairs)
    torch.save(encoder1, "saved_models/encoder1.pt")
    torch.save(attn_decoder1, "saved_models/attn_decoder1.pt")

print("evaluate on train pairs")
evaluateRandomly(encoder1, attn_decoder1, train_pairs)

#print("evaluate on self train pairs")
#evaluateRandomly(encoder1, attn_decoder1, self_train_pairs)

#evaluateAndShowAttention("it is an order .")
