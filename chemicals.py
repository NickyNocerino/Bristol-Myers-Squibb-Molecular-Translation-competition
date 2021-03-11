import numpy as np
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import time
import random
import os
import csv
import pandas
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from CASmodel import ResNetEncoder

from Levenshtein import distance as lev_dist

if not torch.cuda.is_available():
	print("Yo dawg, you dont have CUDA set up, you probably really want to have CUDA set up")
	device = torch.device('cpu')
else:
	print("running with CUDA")
	device = torch.device('cuda')

prefix = "InChI=1S/"
train_dir = '../../train'
test_dir = '../../test'

batch_size = 16

class Dict:
	def __init__(self):
		self.vocab = {}
		self.map = []
		self.next_token = 0
		#<pad> = 0, <sos> = 1, <eos> = 2, ' ' = 3
		self.add_single('<pad>')
		self.add_single('<sos>')
		self.add_single('<eos>')
		#self.add_single(' ')
	def add_single(self, token):
		if token not in self.vocab.keys():
			self.vocab[token] = self.next_token
			self.map.append(token)
			self.next_token +=1
			return True
		else:
			return False

	def __len__(self):
		return self.next_token

	def load_vocab(self, voc):
		for char in voc:
			self.add_single(char)

	def encode(self, input):
		output = []
		temp = input
		while len(temp) > 0:
			if len(temp) > 1:
				if temp[0]+temp[1] in self.vocab.keys():
					output.append(self.vocab[temp[:2]])
					temp = temp[2:]
				elif temp[0] in self.vocab.keys():
					output.append(self.vocab[temp[0]])
					temp = temp[1:]
				else:
					print("NEW SYMBOL FOUND")
					print(temp[0])
					print(input)
					self.add_single(temp[0])
					temp = temp[1:]
					return False
			elif temp[0] in self.vocab.keys():
				output.append(self.vocab[temp[0]])
				temp = temp[1:]
			else:
				print("NEW SYMBOL FOUND")
				print(temp[0])
				print(input)
				self.add_single(temp[0])
				temp = temp[1:]
				return False
		output.append(self.vocab['<eos>'])
		return output

	def decode(self, input):
		output=""
		for sybmol in input:
			if sybmol = self.vocab['<eos>']:
				break
			output += self.map[sybmol]
		return output

class Chem_Dataset(Dataset):
	def __init__(self, root_dir, labels, vocab, prefix, transform=None):
		self.root_dir = root_dir
		self.prefix = prefix
		self.vocab = vocab
		self.data = labels
		self.transform = transform
		#seems like vectorization wont give us a speedup here unforunately, if annoying store and load the database after paths are added
		self.data['path'] = self.data.apply(lambda row: root_dir+'/'+row.image_id[0]+'/'+row.image_id[1]+'/'+row.image_id[2]+'/'+row.image_id+'.png', axis=1)
		#print(self.data['path'])
	
	def __len__(self):
		return len(self.data.index)

	def __getitem__(self, idx):
		path = self.data.iloc[idx]['path'][:]
		# consider Denoising
		im = Image.open(path)
		#orientation = random.choice([0,90,180,270])
		#im = im.rotate(orientation, expand=True)
		if( im.size[0] < im.size[1] ):
			im.rotate(-90, expand=True)
		#print(im.size)
		if self.transform:
			im = self.transform(im)
		#im.show()
		#print(self.data.iloc[idx]['InChI'][len(self.prefix):])
		label = self.vocab.encode(self.data.iloc[idx]['InChI'][len(self.prefix):])
		return im.float(), torch.LongTensor(label), len(label)

def collate_fn(batch):
	X = [param[0] for param in batch]
	Y = [param[1] for param in batch]
	Y_lens = [param[2] for param in batch]
	X = torch.stack(X, dim=0)
	Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=vocab.vocab['<pad>'])
	return X, Y, Y_lens



train_labels = pandas.read_csv('../../train_labels.csv')
#print(train_labels.iloc[2]['InChI'][len(prefix):])

vocab = Dict()
vocab.load_vocab(
	["c", "h", "b", "t", "m", "s", "i","N", "Br", "I",
	 "S", "Cl", "H", "C", "P", "O", "Si", "F", "B","D",
	 "T","(",")",",","-","/","1","2","3","4","5","6",
	 "7","8","9","0","+"])

check = True


data_augs = transforms.Compose([
	transforms.Grayscale(),
	#transforms.Resize((512,1024)),
	transforms.Resize((224,224)),
	#transforms.RandomRotation(5),
	transforms.ToTensor(),
])

full_dataset = Chem_Dataset(train_dir, train_labels, vocab, prefix, transform=data_augs)
#consider num_workers= 4 or 8 for speed
train_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

print("loaders created")
encoder = ResNetEncoder(512,1024).to(device)

for i, batched in enumerate(train_dataloader):
	data, labels, label_lens = batched
	print(data.shape)
	keys, vals = encoder(data.to(device))
	print(keys.shape)
	break
