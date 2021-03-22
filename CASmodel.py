import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50

import random

#This is to define device for ImagetoSeq, consider passing in device instead for better practices
if not torch.cuda.is_available():
	#print("Yo dawg, you dont have CUDA set up, you probably really want to have CUDA set up")
	device = torch.device('cpu')
else:
	#print("running with CUDA")
	device = torch.device('cuda')


class ResNetEncoder(nn.Module):
    def __init__(self, value_size=128, key_size=128):
        super(ResNetEncoder, self).__init__()
        #this is cheap i know,TODO: make this better in the future
        self.pre_net = nn.Conv2d(1,3,(66,132),(2,4))
        self.net = resnet50(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        #self.net[0].in_channels = 1
        #print(self.net)
        #2048 is the ResNet out channels (resnet out is [Batch, 2048, 7, 7])
        #height and width should rescale if we can afford the memory for larger images lol
        self.key_net = nn.Linear(2048, key_size)
        self.value_net = nn.Linear(2048, value_size)


    def forward( self, x, transform=None):
        x = self.pre_net(x)
        x = self.net(x)
        x = x.permute(0,2,3,1)
        x = x.view(x.size(0), -1, x.size(-1))
        #x is [Batch, height*width, channels]
        keys = self.key_net(x)
        values = self.value_net(x)
        return keys, values

class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()

	def forward(self, query, key, value, lens):

		energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
		#mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
		#mask = mask.to(device)
		#energy.masked_fill_(mask, -1e9)
		attention = nn.functional.softmax(energy, dim=1)
		out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
		return out, attention

#Decoder taken from Listen Attend Spell implementation, may need some modifacations 
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        #if (isAttended == True):
        self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        if hidden_dim == (value_size + key_size):
            print('tying weights')
            self.character_prob.weight = self.embedding.weight
        else:
            print('could not tie weights, sizes do not match')

    def forward(self, key, values, lens, tf=0.1, text=None, isTrain=True):
        batch_size = key.shape[0]
        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 400
        preds = []
        hidden_state = [None, None]
        prediction = torch.zeros(batch_size,1).to(device)
        #print(embeddings.shape)
        context = torch.zeros(batch_size,key.shape[2]).to(device)
        is_tf = random.randrange(100) > (100 * tf)
        emb = embeddings[:,0,:]
        for i in range(max_len):
            if i == 0:
                emb = embeddings[:,0,:]
            else:
                is_tf = random.randrange(100) > (100 * tf)
                #print(is_tf)
                if is_tf:
                    emb = embeddings[:,i,:]
                else:
                    emb = self.embedding(prediction.argmax(dim=-1))
            #print(emb.shape)
            inp = torch.cat([emb, context], dim=1)
            hidden_state[0] = self.lstm1(inp, hidden_state[0])
            inp_2 = hidden_state[0][0]
            hidden_state[1] = self.lstm2(inp_2, hidden_state[1])
            output = hidden_state[1][0]
            if self.isAttended:
                context,_ = self.attention(output,key,values, lens)
            inp = torch.cat([output, context], dim=1)
            prediction = self.character_prob(inp)
            #prediction = nn.functional.gumbel_softmax(prediction, dim=1)
            #print(prediction.shape)
            preds.append(prediction.unsqueeze(1))
            #break
        return torch.cat(preds, dim=1)

    def generate(self, key, values, lens, vocab):

        batch_size = key.shape[0]
        max_len = 400
        preds = []
        hidden_state = [None, None]
        prediction = torch.zeros(batch_size,1).to(device)
        context = torch.zeros(batch_size,key.shape[2]).to(device)
        start = (torch.ones((batch_size,1))* vocab.vocab['<sos>']).long().to(device)
        #print(start.shape)
        emb = self.embedding(start).squeeze(1)
        #print(emb.shape)
        #print(context.shape)
        for i in range(max_len):
            #print(emb.shape)
            inp = torch.cat([emb, context], dim=1)
            hidden_state[0] = self.lstm1(inp, hidden_state[0])
            inp_2 = hidden_state[0][0]
            hidden_state[1] = self.lstm2(inp_2, hidden_state[1])
            output = hidden_state[1][0]
            if self.isAttended:
                context,_ = self.attention(output,key,values, lens)
            inp = torch.cat([output, context], dim=1)
            prediction = self.character_prob(inp)
            prediction = nn.functional.gumbel_softmax(prediction, dim=1)
            preds.append(prediction.argmax(dim=-1).unsqueeze(1))
            emb = self.embedding( preds[-1].squeeze(1))
        #print(preds[-1].shape)
        return torch.cat(preds, dim=1)

#Wrapper  model
class Image2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for the encoder and decoder.
    '''
    def __init__(self,  vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Image2Seq, self).__init__()
        self.encoder = ResNetEncoder(value_size=128, key_size=128)
        self.decoder = Decoder(vocab_size, hidden_dim, isAttended=isAttended)

    #speech_len is a relic of when this was an Seq2Seq model, simply pass None for now
    #TODO remove speech_len
    def forward(self, speech_input, speech_len, batch_size=1,vocab=None, text_input=None, isTrain=True, tf=.1):
        if self.decoder.isAttended:
            key, value= self.encoder(speech_input)
            lens = None
        else:
            if isTrain:
                batch_size=text_input.shape[0]
            key = torch.zeros(batch_size, 1, 128).to(device)
            value = None
            lens = None
        if (isTrain == True):
            predictions = self.decoder(key, value, lens, text=text_input, tf=tf)
        else:
            predictions = self.decoder.generate(key, value, lens, vocab)
        return predictions