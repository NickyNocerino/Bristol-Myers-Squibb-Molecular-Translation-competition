import numpy as np
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import time
import random
import os
import csv
import pandas
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import CASmodel 

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
pre_train = True

#TODO move vocab and data set/loader to its own file

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
		output = [self.vocab['<sos>']]
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
		for symbol in input:
			if symbol == self.vocab['<eos>']:
				break
			output += self.map[symbol]
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

print("Vocab Defined")

check = True


data_augs = transforms.Compose([
	transforms.Grayscale(),
	#transforms.Resize((512,1024)),
	transforms.Resize((224,224)),
	#transforms.RandomRotation(5),
	transforms.ToTensor(),
])
full_dataset = Chem_Dataset(train_dir, train_labels, vocab, prefix, transform=data_augs)
val_size = int(len(full_dataset)*.1)
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - val_size, val_size ])
#consider num_workers= 4 or 8 for speed
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)


print("loaders created")
model = CASmodel.Image2Seq(len(vocab), 256, isAttended=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
criterion =  nn.CrossEntropyLoss( ignore_index=vocab.vocab['<pad>'])

print("model initialized")

if pre_train:
	pre_train_epochs = 5
	print("Beggining model pre-training")
	for e in range(pre_train_epochs):
		epoch_loss = 0
		model.train()
		for i, batched in enumerate(tqdm(train_data_loader)):
			data, labels, label_lens = batched
			optimizer.zero_grad()
			#print(labels.shape[0])
			pred2 = model(None, None, text_input=labels.to(device))
			loss = criterion(pred2[:,:-1,:].permute(0,2,1),labels[:,1:].to(device))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
			optimizer.step()
			epoch_loss+= loss.item()
			#print("Epoch: "+str(e+1)+"- "+str(i/len(train_data_loader)*100)+"% "+"Loss: "+str(loss.item()), end="\r", flush=True)
		model.eval()
		tot_dist = 0
		count = 0
		for i, batched in enumerate(tqdm(val_data_loader)):
			data, labels, label_lens = batched
			pred2 = model(None, None, text_input=labels.to(device))
			for j in range(pred2.shape[0]):
				sent_a = vocab.decode(labels[j][1:])
				#encode = pred2[j][:-1]
				#print(encode.shape)
				#encode = torch.argmax(encode, dim=1)
				sent_b = vocab.decode(torch.argmax(pred2[j][:-1],dim=1))
				tot_dist +=  lev_dist(sent_a,sent_b)
				count += 1
		print(sent_a)
		print(sent_b)
		print("Lev Dist: "+str(tot_dist/count))
		#print("Epoch: " + str(e+1)+ " Lev Dist: "+str(tot_dist/count)+" Loss: "+ str(epoch_loss))
		#pred3 = model(None, None, vocab=vocab, isTrain=False)
		#print(vocab.clean_encode_to_readable(pred3[0]))
		#print(vocab.clean_encode_to_readable(labels[0][1:]))
	torch.save(model.state_dict(), 'pre_trained.model')

else:
	#load the model
	print("Loading pre-trained model")
	model.load_state_dict(torch.load('pre_trained.model'), strict=False)
