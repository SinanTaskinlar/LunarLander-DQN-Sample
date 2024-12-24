# Ortak kullanılan importlar
from Models import A3C
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import gymnasium as gym
from collections import deque
from multiprocessing import Manager, Process
from torch.distributions import Categorical