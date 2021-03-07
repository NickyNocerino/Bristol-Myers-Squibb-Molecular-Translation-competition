import numpy as np
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import time
import random
import os
import csv
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from Levenshtein import distance as lev_dist

if not torch.cuda.is_available():
	print("Yo dawg, you dont have CUDA set up, you probably really want to have CUDA set up")
	device = torch.device('cpu')
else:
	print("running with CUDA")
	device = torch.device('cuda')

prefix = "InChI=1S/"

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
			if temp[0] in self.vocab.keys():
				output.append(self.vocab[temp[0]])
				temp = temp[1:]
			elif temp[0]+temp[1] in self.vocab.keys():
				output.append(self.vocab[temp[:2]])
				temp = temp[2:]
			else:
				print("NEW SYMBOL FOUND")
				print(temp[0])
				self.add_single(temp[0])
				temp = temp[1:]
		return output

	def decode(self, input):
		output=""
		for sybmol in input:
			output += self.map[sybmol]
		return output
		


train_labels = pandas.read_csv('../../train_labels.csv')
#print(train_labels.iloc[2]['InChI'][len(prefix):])

vocab = Dict()
vocab.load_vocab(["c", "h", "b", "t", "m", "s", "i","N", "Br", "I", "S", "Cl", "H", "C", "P", "O", "Si", "F", "B","(",")",",","-","/","1","2","3","4","5","6","7","8","9","0","+"])
print(len(vocab))
original = train_labels.iloc()[50]['InChI'][len(prefix):]
print(original)
encoded = vocab.encode(original)
print(encoded)
decoded = vocab.decode(encoded)
print(decoded)

if original == decoded:
	print("horray! it worked")

