from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import subprocess
import os

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
import smatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 180

dev_file = "data/little-prince/dev-dfs-linear_targ.txt"
dev_predict = "model_out/dev-predict.txt"

train_x_file = "data/little-prince/training-dfs-linear_targ.txt"
train_y_file = "data/little-prince/training-dfs-linear_src.txt"

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

    sen_lines = open(src_path, encoding='utf-8').\
        read().strip().split('\n')
    amr_lines = open(targ_path, encoding='ascii').\
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

def prepareSelfTrainData(encoder, decoder, sentences, input_lang, output_lang, max_length=MAX_LENGTH):
    pairs = []
    
    for sen in sentences:
        if len(sen.split(' ')) < max_length:
            amr = " ".join(evaluate(encoder, decoder, sen, input_lang, output_lang)[0])
            pairs.append([sen, amr])
    #print(pairs)
    pairs = filterPairs(pairs)
    
    return pairs
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        print("input_size", input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, input, hidden):
        #print("encoderRNN input size", input.shape)
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
        print("output_size", output_size)
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


def tensorsFromPair(pair, input_lang, output_lang):
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
            #print(decoder_output, target_tensor[di])
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

def inflate_amr(amrs):
    '''
    amrs is a list of stripped amrs
    return list of inflated amrs
    '''
    script_path = '/storage/czw/NeuralAmr/'
    results = []
    for line in amrs:
        if len(line) > 0:
            line = line.lstrip().rstrip()
            result = subprocess.run([os.path.join(script_path, "anonDeAnon_java.sh"), 'deAnonymizeAmr', 'false', line], stdout=subprocess.PIPE)
            inflated = result.stdout.decode('utf-8')
            inflated = inflated.split("#")[0]
            if inflated == "FAILED_TO_PARSE":
                results.append("(y / yes)")
            else:
                results.append(inflated)
    return results

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainEpochs(encoder, decoder, n_epochs, pairs, input_lang, output_lang, iter_count, loss_scalar, learning_rate=0.01):
    start = time.time()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(len(pairs)]
    training_pairs = [tensorsFromPair(pair, input_lang, output_lang) for pair in pairs]
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
            loss_scalar += loss
            iter_count += 1

                    
            if iter_count%1000 == 0:
                print(iter_count)
                evaluateFile(dev_file, dev_predict, encoder1, attn_decoder1, input_lang, output_lang)
                smatch_epoch = smatch.smatch_score("data/little-prince/gold_inflated.txt", "model_out/dev-predict.txt")
                print("smatch ", smatch_epoch)
                writer.add_scalar('data/dev smatch', smatch_epoch, iter_count)
            if iter_count%100 == 0:
                writer.add_scalar('data/loss', loss_scalar, iter_count)
                print("100 loss", loss_scalar, iter_count)
                loss_scalar = 0
    return iter_count

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
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

    for di in range(max_length-1):#Hacky workaround TODO, otherwise the predictions will be max_length, and we'll automatically filter them when building the self training data set.
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

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateFile(in_file, out_file, encoder1, attn_decoder1, input_lang, output_lang):
    '''
       in_file is the name of a file of nl sentences
       out_file is the name of a file which will contain amr predictions separated by new lines. The amrs are properly formatted. bad amrs are replaced with a filler (y \ yes)
    '''
    #f4 = open("data/dev-dfs-linear_targ.txt", encoding='ascii')
    f4 = open(in_file, encoding='ascii')
    f5 = open(out_file, "w")
    f6 = open("model_out/tmp_amrs.txt", "w")
    dev_lines = list(f4.readlines())
    #print(len(dev_lines))
    amrs = []
    print("evaluating dev set")
    for sen in tqdm(dev_lines):
        if len(sen.split(' ')) < MAX_LENGTH:
            amr = " ".join(evaluate(encoder1, attn_decoder1, sen, input_lang, output_lang)[0])
            amrs.append(amr)
        else:
            print("DEV SENTENCE TOO LONG")
    f4.close()

    for amr in amrs: 
        f6.write(amr)
        f6.write("\n\n")
    f6.close()
    
    inflated_amrs = inflate_amr(amrs)
    print("inflating amr predictions")
    for amr in tqdm(inflated_amrs):
        f5.write(amr)
        f5.write("\n\n")
    f5.close()

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
        encoder1, attn_decoder1, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

def writeModels(num_layers, hidden_size, input_lang, output_lang, writer):
    dummy_encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers)
    dummy_input = tensorFromSentence(input_lang, "yes").cpu()[0]#only get first of the sequence
    dummy_encoder1_hidden = [dummy_encoder1.initHidden()[0].cpu(), dummy_encoder1.initHidden()[1].cpu()] 
    writer.add_graph(dummy_encoder1, (dummy_input, dummy_encoder1_hidden), verbose=True)

    dummy_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1)
    dummy_output = torch.zeros(MAX_LENGTH, dummy_encoder1.hidden_size*2)
    dummy_decoder_input = torch.tensor([[SOS_token]]) 
    dummy_decoder_hidden = [dummy_decoder.initHidden()[0].cpu(), dummy_decoder.initHidden()[1].cpu()] 
    print(dummy_decoder_input.size(), dummy_decoder_hidden[0].size(), dummy_output[0].size())
    writer.add_graph(dummy_decoder, (dummy_decoder_input, dummy_decoder_hidden, dummy_output), verbose=True)
    #

plt.switch_backend('agg')
input_lang, output_lang, train_pairs = prepareData(train_x_file, train_y_file)
num_layers = 2
hidden_size = 256

writer = SummaryWriter()

writeModels(num_layers, hidden_size, input_lang, output_lang, writer)

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1).to(device)

#evaluateFile("data/dev-dfs-linear_targ.txt", "model_out/dev-predict.txt", encoder1, attn_decoder1, input_lang, output_lang)
#print("smatch ", smatch.smatch_score("data/gold_inflated.txt", "data/gold_inflated.txt"))
#print("smatch ", smatch.smatch_score("data/dev_gold_format.txt", "model_out/dev-predict.txt"))
loss_scalar = 0
iter_count = 0
iter_count = trainEpochs(encoder1, attn_decoder1, 20, train_pairs, input_lang, output_lang, iter_count, loss_scalar)

torch.save(encoder1, "saved_models/encoder1.pt")
torch.save(attn_decoder1, "saved_models/attn_decoder1.pt")

self_train_iter = 3
self_train_sample = 3

f3 = open("data/gigaword_sample/gigaword.txt.anonymized","r", encoding='ascii')
giga_lines = list(f3.readlines())
f3.close()
for i in range(self_train_iter):
    print("self_train_iter", i)
    short_lines = [x for x in giga_lines if len(x.split(' ')) < MAX_LENGTH][:self_train_sample*10**i] 
    #print(len(self_train_pairs))
    #print([len(x.split(' ')) for x in giga_lines])
    #self_train_pairs = prepareSelfTrainData(encoder1, attn_decoder1, short_lines, input_lang, output_lang)
    #print("self training on ", len(self_train_pairs),  " pairs")
    #iter_count = trainEpochs(encoder1, attn_decoder1, 5, self_train_pairs, input_lang, output_lang, iter_count, loss_scalar)
    print("fine tuning on ", len(train_pairs),  " pairs")
    iter_count = trainEpochs(encoder1, attn_decoder1, 5, train_pairs, input_lang, output_lang, iter_count, loss_scalar)
    torch.save(encoder1, "saved_models/encoder1.pt")
    torch.save(attn_decoder1, "saved_models/attn_decoder1.pt")

print("evaluate on train pairs")
evaluateRandomly(encoder1, attn_decoder1, train_pairs, input_lang, output_lang)

#print("evaluate on self train pairs")
#evaluateRandomly(encoder1, attn_decoder1, self_train_pairs)

#evaluateAndShowAttention("it is an order .")
f3.close()


writer.close()
