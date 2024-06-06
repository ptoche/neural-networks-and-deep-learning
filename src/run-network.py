#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# execute code from file
import os
os.chdir("/Users/patricktoche/Python/machine-learning/neural-networks-and-deep-learning/src")
exec(open("network.py").read())

# instantiate the model
# from network import Network

import network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
#net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
