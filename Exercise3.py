# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:12:41 2022

@author: Jungyu Lee, 301236221
Exercise 3
"""
import numpy as np
import neurolab as nl

#1, 2. Create input data
input_Jungyu = np.random.uniform(-0.6, 0.6, (100,2))

#3. Create output data
output_Jungyu = input_Jungyu.sum(axis=1).reshape(100, 1)

#4. Set the seed = 1
np.random.seed(1)

#5. Create a neural network 
dim_1 = [-0.6, 0.6]
dim_2 = [-0.6, 0.6]
nn = nl.net.newff([dim_1, dim_2], [6, 1])

#6, 7 Train the network
error_progress = nn.train(input_Jungyu, output_Jungyu, show=15, goal=0.00001)

data_test = [[0.1, 0.2]]
sim = nn.sim(data_test)
print('\nSimulation Result:', sim)
