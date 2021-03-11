import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50

class ResNetEncoder(nn.Module):
    def __init__(self, height, width, value_size=128, key_size=128):
        super(ResNetEncoder, self).__init__()
        #this is cheap i know,TODO: make this better in the future
        self.pre = nn.Conv2d(1,3,(1,1),(1,1))
        self.net = resnet50(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        #self.net[0].in_channels = 1
        #print(self.net)
        #2048 is the ResNet out channels (resnet out is [Batch, 2048, 7, 7])
        #height and width should rescale if we can afford the memory for larger images lol
        self.key_net = nn.Linear(2048, key_size)
        self.value_net = nn.Linear(2048, value_size)


    def forward( self, x, transform=None):
        x = self.pre(x)
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