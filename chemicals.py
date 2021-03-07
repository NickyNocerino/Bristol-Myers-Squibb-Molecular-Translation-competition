import numpy as np
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import time
import random
import os
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

