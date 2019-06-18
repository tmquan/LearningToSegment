# Template for training pytorch with dataflow and tensorboard

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

import cv2
import numpy as np 

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import tensorwatch as tw
import tensorpack.dataflow as df

import tqdm

#
# Global configuration
#
BATCH = 32
EPOCH = 500
SHAPE = 256
NF = 64

#
# Create the data flow
#

#
# Create the model
#

#
# Perform sample
#

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='the image directory')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--sample', action='store_true', help='run inference')
    args = parser.parse_args()
    
    # Choose the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    if args.sample:
        sample()    
    else
        #
        # Train from scratch or load the pretrained network
        #
        # Initialize the program
        writer = SummaryWriter()
        use_cuda = torch.cuda.is_available()
        step = 0
        
        # TODO: Callbacks
        # Save the model
        # Log the loss
        # Log the image to tensorboard

```